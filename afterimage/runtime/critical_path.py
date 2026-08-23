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

import numpy as np

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


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


@dataclasses.dataclass(frozen=True)
class CompiledTopology:
    """A trace's dependency graph, precomputed once so critical_path_fast can
    re-score it many times -- varying only which events' durations get
    overridden -- without repeating the topological sort or graph-building
    work on every call.

    This is the shape replay_planner.py's CEM/QUBO search needs: it scores
    hundreds to thousands of candidate residency masks against the SAME
    handful of traces, and profiling showed critical_path() itself (not just
    its numeric DP) was 55% of that search's wall time on a 720-event trace,
    3.68M of it plain max()/min() calls (docs/RESULTS_LOG.md speed audit).
    compile_topology() does the graph work in Python once per trace;
    critical_path_fast() re-runs only the numeric forward/backward passes,
    compiled, per call.
    """
    event_ids: tuple[str, ...]      # index -> event id, in original events order
    id_to_index: dict[str, int]
    order: np.ndarray               # int32, topological order, as indices
    parent_ptr: np.ndarray          # int32[n+1], CSR row pointers into parent_idx
    parent_idx: np.ndarray          # int32, each event's parents in its own dependency order
    child_ptr: np.ndarray           # int32[n+1], CSR row pointers into child_idx
    child_idx: np.ndarray           # int32, each event's children in child-original-index order
    base_duration: np.ndarray       # float64, max(0.0, event.duration_s) per index

    @property
    def n(self) -> int:
        return len(self.event_ids)


def compile_topology(events: list[TraceEvent]) -> CompiledTopology:
    """Builds the same dependency graph and topological order critical_path()
    computes internally, as integer-indexed CSR arrays instead of string-keyed
    dicts. Raises the same errors critical_path() does (duplicate ids, a
    dependency on a missing event, a cycle) -- just once here, at compile
    time, rather than on every subsequent score.
    """
    n = len(events)
    id_to_index: dict[str, int] = {}
    for i, event in enumerate(events):
        if event.id in id_to_index:
            raise ValueError("trace event ids must be unique")
        id_to_index[event.id] = i

    children_lists: list[list[int]] = [[] for _ in range(n)]
    parents_lists: list[list[int]] = [[] for _ in range(n)]
    indegree = [0] * n
    for i, event in enumerate(events):
        for parent_id in event.dependencies:
            p = id_to_index.get(parent_id)
            if p is None:
                raise ValueError("event %s depends on missing event %s" % (event.id, parent_id))
            children_lists[p].append(i)
            parents_lists[i].append(p)
            indegree[i] += 1

    # Kahn's algorithm, index-based -- same FIFO order as critical_path()'s
    # string-keyed version, since both seed from and append children in
    # ascending original-event-index order.
    queue = deque(i for i in range(n) if indegree[i] == 0)
    order: list[int] = []
    remaining = list(indegree)
    while queue:
        i = queue.popleft()
        order.append(i)
        for c in children_lists[i]:
            remaining[c] -= 1
            if remaining[c] == 0:
                queue.append(c)
    if len(order) != n:
        raise ValueError("trace dependency graph contains a cycle")

    parent_ptr = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        parent_ptr[i + 1] = parent_ptr[i] + len(parents_lists[i])
    parent_idx = np.empty(int(parent_ptr[-1]), dtype=np.int32)
    for i in range(n):
        parent_idx[parent_ptr[i]:parent_ptr[i + 1]] = parents_lists[i]

    child_ptr = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        child_ptr[i + 1] = child_ptr[i] + len(children_lists[i])
    child_idx = np.empty(int(child_ptr[-1]), dtype=np.int32)
    for i in range(n):
        child_idx[child_ptr[i]:child_ptr[i + 1]] = children_lists[i]

    base_duration = np.array([max(0.0, event.duration_s) for event in events], dtype=np.float64)

    return CompiledTopology(
        event_ids=tuple(event.id for event in events),
        id_to_index=id_to_index,
        order=np.array(order, dtype=np.int32),
        parent_ptr=parent_ptr, parent_idx=parent_idx,
        child_ptr=child_ptr, child_idx=child_idx,
        base_duration=base_duration,
    )


def _critical_path_numeric_python(order, parent_ptr, parent_idx, child_ptr, child_idx, duration):
    """Pure-Python numeric core -- the fallback when numba isn't installed,
    and the thing the compiled kernel is checked against. Same algorithm as
    critical_path()'s forward/backward passes, over index arrays instead of
    dicts; see _critical_path_numeric_numba's tie-break notes below."""
    n = order.shape[0]
    earliest = np.empty(n, dtype=np.float64)
    finish = np.empty(n, dtype=np.float64)
    predecessor = np.full(n, -1, dtype=np.int32)
    for oi in range(n):
        i = order[oi]
        p0, p1 = parent_ptr[i], parent_ptr[i + 1]
        if p1 > p0:
            best = parent_idx[p0]
            best_finish = finish[best]
            for k in range(p0 + 1, p1):
                cand = parent_idx[k]
                if finish[cand] > best_finish:
                    best = cand
                    best_finish = finish[cand]
            earliest[i] = best_finish
            predecessor[i] = best
        else:
            earliest[i] = 0.0
        finish[i] = earliest[i] + duration[i]

    makespan = finish[order[0]]
    last = order[0]
    for oi in range(1, n):
        i = order[oi]
        if finish[i] > makespan:
            makespan = finish[i]
            last = i

    latest_start = np.empty(n, dtype=np.float64)
    for oi in range(n - 1, -1, -1):
        i = order[oi]
        c0, c1 = child_ptr[i], child_ptr[i + 1]
        if c1 > c0:
            lf = latest_start[child_idx[c0]]
            for k in range(c0 + 1, c1):
                v = latest_start[child_idx[k]]
                if v < lf:
                    lf = v
        else:
            lf = makespan
        latest_start[i] = lf - duration[i]

    slack = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = latest_start[i] - earliest[i]
        slack[i] = s if s > 0.0 else 0.0

    return earliest, finish, predecessor, latest_start, slack, makespan, last


if _HAS_NUMBA:
    @numba.njit(cache=True)
    def _critical_path_numeric_numba(order, parent_ptr, parent_idx, child_ptr, child_idx, duration):
        """Same algorithm as _critical_path_numeric_python, compiled.

        Tie-breaks matter here, not just totals: critical_path()'s forward
        pass picks max(deps, key=finish.get), and Python's max() returns the
        FIRST maximal element on a tie -- replicated here with a strict '>'
        comparison over each event's parents in their original dependency
        order (parent_idx preserves that order; see compile_topology). The
        makespan/"last" selection has the same first-wins-tie property,
        walking `order` (not raw index order) with a strict '>'. The
        backward pass's min() over children only needs the numeric result,
        not which child achieves it, so no tie-break there matters.
        """
        n = order.shape[0]
        earliest = np.empty(n, dtype=np.float64)
        finish = np.empty(n, dtype=np.float64)
        predecessor = np.full(n, -1, dtype=np.int32)
        for oi in range(n):
            i = order[oi]
            p0, p1 = parent_ptr[i], parent_ptr[i + 1]
            if p1 > p0:
                best = parent_idx[p0]
                best_finish = finish[best]
                for k in range(p0 + 1, p1):
                    cand = parent_idx[k]
                    if finish[cand] > best_finish:
                        best = cand
                        best_finish = finish[cand]
                earliest[i] = best_finish
                predecessor[i] = best
            else:
                earliest[i] = 0.0
            finish[i] = earliest[i] + duration[i]

        makespan = finish[order[0]]
        last = order[0]
        for oi in range(1, n):
            i = order[oi]
            if finish[i] > makespan:
                makespan = finish[i]
                last = i

        latest_start = np.empty(n, dtype=np.float64)
        for oi in range(n - 1, -1, -1):
            i = order[oi]
            c0, c1 = child_ptr[i], child_ptr[i + 1]
            if c1 > c0:
                lf = latest_start[child_idx[c0]]
                for k in range(c0 + 1, c1):
                    v = latest_start[child_idx[k]]
                    if v < lf:
                        lf = v
            else:
                lf = makespan
            latest_start[i] = lf - duration[i]

        slack = np.empty(n, dtype=np.float64)
        for i in range(n):
            s = latest_start[i] - earliest[i]
            slack[i] = s if s > 0.0 else 0.0

        return earliest, finish, predecessor, latest_start, slack, makespan, last


def critical_path_fast(topology: CompiledTopology,
                       duration_overrides: dict[str, float] | None = None) -> CriticalPathReport:
    """critical_path(), scored against a precompiled CompiledTopology instead
    of a raw event list -- same return type, same values, for a graph whose
    topology was already validated and sorted by compile_topology(). Use this
    (with one compile_topology() call reused across many scores) anywhere a
    trace gets replayed repeatedly with different duration_overrides; use
    critical_path() directly for a one-off score.
    """
    n = topology.n
    if n == 0:
        return CriticalPathReport(0.0, (), {}, {}, {})

    duration = topology.base_duration.copy()
    if duration_overrides:
        for event_id, value in duration_overrides.items():
            idx = topology.id_to_index.get(event_id)
            if idx is not None:
                duration[idx] = max(0.0, float(value))

    numeric = _critical_path_numeric_numba if _HAS_NUMBA else _critical_path_numeric_python
    earliest, finish, predecessor, latest_start, slack, makespan, last = numeric(
        topology.order, topology.parent_ptr, topology.parent_idx,
        topology.child_ptr, topology.child_idx, duration)

    path_idx = []
    cur = int(last)
    while cur != -1:
        path_idx.append(cur)
        cur = int(predecessor[cur])
    path_idx.reverse()

    event_ids = topology.event_ids
    path = tuple(event_ids[i] for i in path_idx)
    earliest_d = {event_ids[i]: float(earliest[i]) for i in range(n)}
    latest_d = {event_ids[i]: float(latest_start[i]) for i in range(n)}
    slack_d = {event_ids[i]: float(slack[i]) for i in range(n)}
    return CriticalPathReport(float(makespan), path, earliest_d, latest_d, slack_d)


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
