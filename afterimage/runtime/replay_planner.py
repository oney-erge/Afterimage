"""Whole-plan residency search against an event-DAG digital twin.

The ordinary planners assign each tensor an independent value and then sort.
That approximation misses a basic scheduling fact: removing one preparation
span can expose a different critical path, so the value of tensor A depends on
which other tensors are resident.  This module searches complete feasible
sets with the cross-entropy method (CEM) and scores them by replaying measured
traces.  Exploration is offline; a live engine only loads a frozen plan.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
from collections import defaultdict

import numpy as np

from .critical_path import CriticalPathProfile, TraceEvent, critical_path
from .vram_planner import TierPlan, plan_from_manifest


PREPARE_KINDS = frozenset(("read", "io", "decode", "transfer", "copy"))


def manifest_fingerprint(manifest: dict) -> str:
    """Hash placement-relevant facts, not paths or mutable benchmark notes."""
    payload = {
        key: {
            "orig_bytes": int(meta["orig_bytes"]),
            "comp_bytes": int(meta["comp_bytes"]),
            "row_gather": bool(meta.get("row_gather")),
            "shape": meta.get("shape"),
            "dtype": meta.get("dtype"),
        }
        for key, meta in sorted(manifest["tensors"].items())
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class ReplaySearchReport:
    baseline_s: float
    control_s: float
    control_policy: str
    optimized_s: float
    predicted_speedup: float
    optimized_over_control: float
    evaluations: int
    iterations: int
    population: int
    elite_fraction: float
    observed_coverage: float


@dataclasses.dataclass(frozen=True)
class ReplayResidencyPlan:
    manifest_sha256: str
    vram_budget_bytes: int
    vram_headroom_bytes: int
    vram_keys: tuple[str, ...]
    vram_bytes: int
    report: ReplaySearchReport
    seed: int
    schema_version: int = 1

    def save(self, path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(self), indent=2,
                                  sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> "ReplayResidencyPlan":
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported replay-residency plan schema")
        payload["vram_keys"] = tuple(payload["vram_keys"])
        payload["report"] = ReplaySearchReport(**payload["report"])
        return cls(**payload)

    def to_tier_plan(self, manifest: dict, *, vram_budget_gb: float,
                     ram_budget_gb: float = 0.0,
                     decode_slice_elems: int | None = None,
                     stream_only: dict | None = None) -> TierPlan:
        """Validate the frozen plan against the runtime and materialize it."""
        if ram_budget_gb:
            raise ValueError("replay_cem v1 optimizes VRAM/disk only; RAM must be zero")
        if manifest_fingerprint(manifest) != self.manifest_sha256:
            raise ValueError("replay plan belongs to a different compressed manifest")
        budget = int(vram_budget_gb * 1e9)
        if budget != self.vram_budget_bytes:
            raise ValueError(
                "replay plan budget is %.6f GB, runtime requested %.6f GB" %
                (self.vram_budget_bytes / 1e9, budget / 1e9))
        stream_only = stream_only or {}
        baseline = plan_from_manifest(
            manifest, vram_budget_gb=vram_budget_gb,
            ram_budget_gb=0.0, decode_slice_elems=decode_slice_elems,
            stream_only=stream_only)
        if not baseline.feasible:
            return baseline
        if baseline.vram_headroom_bytes != self.vram_headroom_bytes:
            raise ValueError(
                "replay plan headroom differs from this runtime; rebuild it with "
                "the same decode_slice_elems and head policy")

        row_gather = [key for key, meta in manifest["tensors"].items()
                      if meta.get("row_gather")]
        candidates = {key for key, meta in manifest["tensors"].items()
                      if not meta.get("row_gather") and key not in stream_only}
        selected = set(self.vram_keys)
        invalid = selected - candidates
        if invalid:
            raise ValueError("replay plan selects ineligible tensors: %s" %
                             ", ".join(sorted(invalid)))
        actual_bytes = sum(int(manifest["tensors"][key]["orig_bytes"])
                           for key in selected)
        available = budget - self.vram_headroom_bytes
        if actual_bytes != self.vram_bytes or actual_bytes > available:
            raise ValueError("replay plan byte accounting is invalid for this runtime")

        disk = sorted((candidates - selected) | set(stream_only))
        return TierPlan(
            vram_budget_bytes=budget, ram_budget_bytes=0,
            vram_headroom_bytes=self.vram_headroom_bytes,
            vram_keys=sorted(selected), ram_keys=[], disk_keys=disk,
            row_gather_keys=sorted(row_gather), vram_bytes=actual_bytes,
            ram_bytes=0,
            disk_bytes_per_token=sum(
                int(manifest["tensors"][key]["comp_bytes"]) for key in disk),
            feasible=True, reason="")


def _mean_makespan(traces: list[list[TraceEvent]], selected: set[str],
                   event_ids: list[dict[str, tuple[str, ...]]]) -> float:
    durations = []
    for events, trace_ids in zip(traces, event_ids):
        overrides = {
            event_id: 0.0
            for key in selected
            for event_id in trace_ids.get(key, ())
        }
        durations.append(critical_path(events, overrides).duration_s)
    return float(np.mean(durations))


def optimize_replay_residency(
        manifest: dict, traces: list[list[TraceEvent]], *,
        vram_budget_gb: float, decode_slice_elems: int = 1 << 25,
        iterations: int = 12, population: int = 64,
        elite_fraction: float = 0.15, smoothing: float = 0.7,
        seed: int = 0, stream_only: dict | None = None,
        minimum_coverage: float = 0.90) -> ReplayResidencyPlan:
    """Search feasible resident sets and return an immutable frozen plan.

    CEM is used as model-based policy search: a Bernoulli policy samples
    complete placement actions, the event-DAG replay supplies reward, and
    elite actions update the policy. Deterministic traffic-density,
    measured-cost and critical-path plans are included, so search cannot
    return a worse simulated solution than the best existing planner.
    """
    if not traces or any(not trace for trace in traces):
        raise ValueError("at least one non-empty calibration trace is required")
    if iterations < 1 or population < 2:
        raise ValueError("iterations >= 1 and population >= 2 are required")
    if not (0.0 < elite_fraction <= 0.5):
        raise ValueError("elite_fraction must be in (0, 0.5]")
    if not (0.0 < smoothing <= 1.0):
        raise ValueError("smoothing must be in (0, 1]")

    stream_only = stream_only or {}
    baseline_plan = plan_from_manifest(
        manifest, vram_budget_gb=vram_budget_gb, ram_budget_gb=0.0,
        decode_slice_elems=decode_slice_elems, stream_only=stream_only)
    if not baseline_plan.feasible:
        raise ValueError("infeasible replay-search budget: " + baseline_plan.reason)
    available = baseline_plan.vram_budget_bytes - baseline_plan.vram_headroom_bytes
    keys = sorted(key for key, meta in manifest["tensors"].items()
                  if not meta.get("row_gather") and key not in stream_only)
    sizes = np.asarray([int(manifest["tensors"][key]["orig_bytes"])
                        for key in keys], dtype=np.int64)
    if not keys:
        raise ValueError("manifest has no replay-search placement candidates")

    event_ids: list[dict[str, tuple[str, ...]]] = []
    observed: set[str] = set()
    for events in traces:
        grouped: dict[str, list[str]] = defaultdict(list)
        for event in events:
            if (event.tensor_key in keys and event.kind in PREPARE_KINDS):
                grouped[event.tensor_key].append(event.id)
                observed.add(event.tensor_key)
        event_ids.append({key: tuple(ids) for key, ids in grouped.items()})
    coverage = len(observed) / len(keys)
    if coverage < minimum_coverage:
        missing = sorted(set(keys) - observed)
        raise ValueError(
            "calibration traces cover %.1f%% of candidates, below %.1f%%; "
            "missing examples: %s" %
            (100 * coverage, 100 * minimum_coverage, ", ".join(missing[:5])))

    rng = np.random.default_rng(seed)
    total_size = max(int(sizes.sum()), 1)
    initial = min(0.95, max(0.02, available / total_size))
    probability = np.full(len(keys), initial, dtype=np.float64)

    def repair(mask: np.ndarray) -> np.ndarray:
        mask = mask.copy()
        used = int(sizes[mask].sum())
        if used > available:
            selected = np.flatnonzero(mask)
            # Preserve high-probability items; size-normalization avoids one
            # giant tensor surviving merely because it was sampled first.
            order = sorted(selected,
                           key=lambda i: (probability[i] / max(int(sizes[i]), 1),
                                          probability[i]))
            for idx in order:
                if used <= available:
                    break
                mask[idx] = False
                used -= int(sizes[idx])
        noise = rng.gumbel(size=len(keys))
        add_order = np.argsort(-(np.log(probability + 1e-12) + noise))
        for idx in add_order:
            if not mask[idx] and used + int(sizes[idx]) <= available:
                mask[idx] = True
                used += int(sizes[idx])
        return mask

    baseline_s = _mean_makespan(traces, set(), event_ids)
    measured_profile = CriticalPathProfile.from_traces(traces)
    seed_plans = {"traffic_density": baseline_plan}
    for policy in ("profiled_knapsack", "critical_path"):
        seed_plans[policy] = plan_from_manifest(
            manifest, vram_budget_gb=vram_budget_gb, ram_budget_gb=0.0,
            decode_slice_elems=decode_slice_elems, stream_only=stream_only,
            critical_path_profile=measured_profile, placement_policy=policy)
    seeded = []
    for policy, plan in seed_plans.items():
        mask = np.asarray([key in set(plan.vram_keys) for key in keys])
        mask = repair(mask)
        selected = {key for key, keep in zip(keys, mask) if keep}
        seeded.append((_mean_makespan(traces, selected, event_ids), policy, mask))
    seeded.sort(key=lambda item: item[0])
    control_s, control_policy, best_mask = seeded[0]
    best_mask = best_mask.copy()
    best_s = control_s
    evaluations = len(seeded)
    elite_count = max(1, math.ceil(population * elite_fraction))

    for _ in range(iterations):
        candidates: list[tuple[float, np.ndarray]] = [(best_s, best_mask.copy())]
        for _sample in range(population - 1):
            mask = repair(rng.random(len(keys)) < probability)
            selected = {key for key, keep in zip(keys, mask) if keep}
            score = _mean_makespan(traces, selected, event_ids)
            candidates.append((score, mask))
        evaluations += population - 1
        candidates.sort(key=lambda pair: pair[0])
        if candidates[0][0] < best_s:
            best_s, best_mask = candidates[0][0], candidates[0][1].copy()
        elite = np.stack([mask for _, mask in candidates[:elite_count]])
        target = elite.mean(axis=0)
        probability = ((1.0 - smoothing) * probability + smoothing * target)
        probability = np.clip(probability, 0.02, 0.98)

    selected_keys = tuple(key for key, keep in zip(keys, best_mask) if keep)
    selected_bytes = sum(int(manifest["tensors"][key]["orig_bytes"])
                         for key in selected_keys)
    report = ReplaySearchReport(
        baseline_s=baseline_s, control_s=control_s,
        control_policy=control_policy, optimized_s=best_s,
        predicted_speedup=(baseline_s / best_s if best_s > 0 else float("inf")),
        optimized_over_control=(control_s / best_s - 1.0
                                if best_s > 0 else float("inf")),
        evaluations=evaluations, iterations=iterations, population=population,
        elite_fraction=elite_fraction, observed_coverage=coverage)
    return ReplayResidencyPlan(
        manifest_sha256=manifest_fingerprint(manifest),
        vram_budget_bytes=baseline_plan.vram_budget_bytes,
        vram_headroom_bytes=baseline_plan.vram_headroom_bytes,
        vram_keys=selected_keys, vram_bytes=selected_bytes,
        report=report, seed=seed)
