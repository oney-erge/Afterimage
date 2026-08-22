"""Critical-path tracing and replay for the streaming runtime.

Component timers cannot simply be added when disk reads, host work, PCIe
copies and GPU kernels overlap.  This module records the dependency graph and
answers the useful counterfactual question: which operation's removal would
actually shorten the sweep?

The recorder is intentionally independent of CUDA profiling.  Callers can
record wall-clock spans today and later substitute CUDA-event durations while
keeping the graph, profile and planner contracts unchanged.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import threading
import time
from collections import defaultdict, deque
from typing import Iterator


@dataclasses.dataclass(frozen=True)
class TraceEvent:
    id: str
    kind: str
    resource: str
    start_s: float
    end_s: float
    dependencies: tuple[str, ...] = ()
    tensor_key: str | None = None
    nbytes: int = 0
    metadata: dict = dataclasses.field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclasses.dataclass(frozen=True)
class CriticalPathReport:
    duration_s: float
    event_ids: tuple[str, ...]
    earliest_start: dict[str, float]
    latest_start: dict[str, float]
    slack_s: dict[str, float]

    def is_critical(self, event_id: str, tolerance_s: float = 1e-6) -> bool:
        return self.slack_s[event_id] <= tolerance_s


def critical_path(events: list[TraceEvent],
                  duration_overrides: dict[str, float] | None = None) -> CriticalPathReport:
    """Return the longest dependency path through ``events``.

    Event timestamps are evidence, but dependency duration is the replay
    model.  This makes duration overrides useful for placement and scheduling
    counterfactuals without pretending independent timers add together.
    """
    by_id = {event.id: event for event in events}
    if len(by_id) != len(events):
        raise ValueError("trace event ids must be unique")
    overrides = duration_overrides or {}
    children: dict[str, list[str]] = defaultdict(list)
    indegree = {event.id: 0 for event in events}
    for event in events:
        for parent in event.dependencies:
            if parent not in by_id:
                raise ValueError("event %s depends on missing event %s" % (event.id, parent))
            children[parent].append(event.id)
            indegree[event.id] += 1

    queue = deque(event_id for event_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while queue:
        event_id = queue.popleft()
        order.append(event_id)
        for child in children[event_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(events):
        raise ValueError("trace dependency graph contains a cycle")

    duration = {event.id: max(0.0, float(overrides.get(event.id, event.duration_s)))
                for event in events}
    earliest: dict[str, float] = {}
    predecessor: dict[str, str | None] = {}
    finish: dict[str, float] = {}
    for event_id in order:
        deps = by_id[event_id].dependencies
        if deps:
            parent = max(deps, key=lambda dep: finish[dep])
            earliest[event_id] = finish[parent]
            predecessor[event_id] = parent
        else:
            earliest[event_id] = 0.0
            predecessor[event_id] = None
        finish[event_id] = earliest[event_id] + duration[event_id]

    makespan = max(finish.values(), default=0.0)
    last = max(finish, key=finish.get) if finish else None
    path: list[str] = []
    while last is not None:
        path.append(last)
        last = predecessor[last]
    path.reverse()

    latest_finish = {event_id: makespan for event_id in order}
    latest_start: dict[str, float] = {}
    for event_id in reversed(order):
        if children[event_id]:
            latest_finish[event_id] = min(latest_start[child]
                                          for child in children[event_id])
        latest_start[event_id] = latest_finish[event_id] - duration[event_id]
    slack = {event_id: max(0.0, latest_start[event_id] - earliest[event_id])
             for event_id in order}
    return CriticalPathReport(makespan, tuple(path), earliest, latest_start, slack)


@dataclasses.dataclass
class TensorCost:
    tensor_key: str
    read_s: float = 0.0
    decode_s: float = 0.0
    transfer_s: float = 0.0
    compute_s: float = 0.0
    critical_s: float = 0.0
    counterfactual_s: float = 0.0
    observations: int = 0

    @property
    def total_prepare_s(self) -> float:
        return self.read_s + self.decode_s + self.transfer_s


@dataclasses.dataclass
class CriticalPathProfile:
    tensors: dict[str, TensorCost]
    trace_count: int = 0
    schema_version: int = 2

    @classmethod
    def from_traces(cls, traces: list[list[TraceEvent]],
                    critical_tolerance_s: float = 1e-6) -> "CriticalPathProfile":
        modelled = sorted({event.tensor_key for events in traces for event in events
                           if event.tensor_key and event.metadata.get("modelled")})
        if modelled:
            raise ValueError(
                "trace contains modelled (not genuinely measured) prepare spans "
                "for %d tensor(s), e.g. %s -- a critical-path profile requires "
                "real per-tensor timing; re-record with storage_read_policy="
                "'per_blob', or a reader that times each tensor individually" %
                (len(modelled), ", ".join(modelled[:5])))
        totals: dict[str, TensorCost] = {}
        for events in traces:
            report = critical_path(events)
            touched: set[str] = set()
            prepare_ids: dict[str, list[str]] = defaultdict(list)
            for event in events:
                if not event.tensor_key:
                    continue
                touched.add(event.tensor_key)
                cost = totals.setdefault(event.tensor_key, TensorCost(event.tensor_key))
                value = event.duration_s
                if event.kind in ("read", "io"):
                    cost.read_s += value
                    prepare_ids[event.tensor_key].append(event.id)
                elif event.kind == "decode":
                    cost.decode_s += value
                    prepare_ids[event.tensor_key].append(event.id)
                elif event.kind in ("transfer", "copy"):
                    cost.transfer_s += value
                    prepare_ids[event.tensor_key].append(event.id)
                elif event.kind == "compute":
                    cost.compute_s += value
                if report.is_critical(event.id, critical_tolerance_s):
                    cost.critical_s += value
            for key, event_ids in prepare_ids.items():
                replay = critical_path(events, {event_id: 0.0 for event_id in event_ids})
                totals[key].counterfactual_s += max(
                    0.0, report.duration_s - replay.duration_s)
            for key in touched:
                totals[key].observations += 1
        for cost in totals.values():
            n = max(cost.observations, 1)
            cost.read_s /= n
            cost.decode_s /= n
            cost.transfer_s /= n
            cost.compute_s /= n
            cost.critical_s /= n
            cost.counterfactual_s /= n
        return cls(totals, trace_count=len(traces))

    def score(self, tensor_key: str, policy: str) -> float:
        cost = self.tensors.get(tensor_key)
        if cost is None:
            return 0.0
        if policy == "profiled_knapsack":
            return cost.total_prepare_s
        if policy == "critical_path":
            return cost.counterfactual_s
        raise ValueError("profile cannot score policy %r" % policy)

    def save(self, path) -> None:
        payload = dataclasses.asdict(self)
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> "CriticalPathProfile":
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 2:
            raise ValueError("unsupported critical-path profile schema")
        return cls({key: TensorCost(**value) for key, value in payload["tensors"].items()},
                   trace_count=payload.get("trace_count", 0),
                   schema_version=payload["schema_version"])


class TraceRecorder:
    """Thread-safe event recorder with automatic per-resource ordering."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._events: list[TraceEvent] = []
        self._resource_tail: dict[str, str] = {}
        self._counter = 0
        self._lock = threading.Lock()

    @property
    def events(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)

    @contextlib.contextmanager
    def span(self, kind: str, resource: str, *, tensor_key: str | None = None,
             nbytes: int = 0, dependencies=(), metadata: dict | None = None) -> Iterator[str | None]:
        if not self.enabled:
            yield None
            return
        with self._lock:
            self._counter += 1
            event_id = "%s-%d" % (kind, self._counter)
            deps = list(dependencies)
            tail = self._resource_tail.get(resource)
            if tail and tail not in deps:
                deps.append(tail)
        start = time.perf_counter()
        try:
            yield event_id
        finally:
            end = time.perf_counter()
            event = TraceEvent(event_id, kind, resource, start, end, tuple(deps),
                               tensor_key, nbytes, metadata or {})
            with self._lock:
                self._events.append(event)
                self._resource_tail[resource] = event_id

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._resource_tail.clear()
            self._counter = 0

    def record(self, kind: str, resource: str, start_s: float, end_s: float, *,
               tensor_key: str | None = None, nbytes: int = 0, dependencies=(),
               metadata: dict | None = None) -> str | None:
        if not self.enabled:
            return None
        with self._lock:
            self._counter += 1
            event_id = "%s-%d" % (kind, self._counter)
            deps = list(dependencies)
            tail = self._resource_tail.get(resource)
            if tail and tail not in deps:
                deps.append(tail)
            self._events.append(TraceEvent(event_id, kind, resource, start_s, end_s,
                                           tuple(deps), tensor_key, nbytes,
                                           metadata or {}))
            self._resource_tail[resource] = event_id
        return event_id

    def save(self, path) -> None:
        payload = {"schema_version": 1,
                   "events": [dataclasses.asdict(event) for event in self.events]}
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def load(path) -> list[TraceEvent]:
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported trace schema")
        return [TraceEvent(
            id=row["id"], kind=row["kind"], resource=row["resource"],
            start_s=float(row["start_s"]), end_s=float(row["end_s"]),
            dependencies=tuple(row.get("dependencies", ())),
            tensor_key=row.get("tensor_key"), nbytes=int(row.get("nbytes", 0)),
            metadata=row.get("metadata", {}))
            for row in payload["events"]]


def replay_speedup(events: list[TraceEvent], duration_overrides: dict[str, float]) -> float:
    baseline = critical_path(events).duration_s
    candidate = critical_path(events, duration_overrides).duration_s
    return baseline / candidate if candidate > 0 else float("inf")
