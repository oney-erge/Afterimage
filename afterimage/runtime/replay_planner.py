"""Whole-plan residency search against an event-DAG digital twin.

The ordinary planners assign each tensor an independent value and then sort.
That approximation misses a basic scheduling fact: removing one preparation
span can expose a different critical path, so the value of tensor A depends on
which other tensors are resident.  This module searches complete feasible
sets with either the cross-entropy method (CEM) or a pairwise QUBO surrogate
and scores them by replaying measured traces.  Exploration is offline; a live
engine only loads a frozen plan.
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
    search_method: str = "cem"
    treatment_diverged: bool = False
    control_overlap: float = 1.0
    candidate_group_count: int = 0
    selected_group_count: int = 0
    selected_storage_span_bytes: int = 0


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
            raise ValueError("replay residency v1 optimizes VRAM/disk only; RAM must be zero")
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


def _set_overlap(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@dataclasses.dataclass(frozen=True)
class StorageExtent:
    """One bounded, physically contiguous residency action."""

    keys: tuple[str, ...]
    start: int
    end: int
    resident_bytes: int

    @property
    def span_bytes(self) -> int:
        return self.end - self.start


def build_storage_extents(manifest: dict, keys: list[str], *,
                          max_extent_bytes: int = 1 << 28,
                          max_gap_bytes: int = 0,
                          max_tensors_per_extent: int = 8) -> tuple[StorageExtent, ...]:
    """Group tensor blobs by their real ``weights.bin`` geometry.

    Grouping is an offline action constraint only. It never rewrites the store
    and therefore cannot invalidate offsets or checksums. A tensor larger than
    the requested span remains a singleton because it cannot be split as a
    residency decision.
    """
    if max_extent_bytes < 1:
        raise ValueError("max_extent_bytes must be positive")
    if max_gap_bytes < 0:
        raise ValueError("max_gap_bytes must be non-negative")
    if max_tensors_per_extent < 1:
        raise ValueError("max_tensors_per_extent must be positive")

    located = []
    for key in keys:
        refs = list(manifest["tensors"][key].get("blobs", {}).values())
        if not refs:
            raise ValueError("tensor %s has no physical blob offsets" % key)
        start = min(int(ref["offset"]) for ref in refs)
        end = max(int(ref["offset"]) + int(ref["nbytes"]) for ref in refs)
        located.append((start, end, key,
                        int(manifest["tensors"][key]["orig_bytes"])))
    located.sort()

    extents: list[StorageExtent] = []
    current: list[tuple[int, int, str, int]] = []
    for item in located:
        if current:
            merged_end = max(current[-1][1], item[1])
            gap = item[0] - current[-1][1]
            if (gap < 0 or gap > max_gap_bytes
                    or merged_end - current[0][0] > max_extent_bytes
                    or len(current) >= max_tensors_per_extent):
                extents.append(StorageExtent(
                    tuple(entry[2] for entry in current), current[0][0],
                    max(entry[1] for entry in current),
                    sum(entry[3] for entry in current)))
                current = []
        current.append(item)
    if current:
        extents.append(StorageExtent(
            tuple(entry[2] for entry in current), current[0][0],
            max(entry[1] for entry in current),
            sum(entry[3] for entry in current)))
    return tuple(extents)


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
    control_keys = {key for key, keep in zip(keys, seeded[0][2]) if keep}
    selected_set = set(selected_keys)
    selected_bytes = sum(int(manifest["tensors"][key]["orig_bytes"])
                         for key in selected_keys)
    report = ReplaySearchReport(
        baseline_s=baseline_s, control_s=control_s,
        control_policy=control_policy, optimized_s=best_s,
        predicted_speedup=(baseline_s / best_s if best_s > 0 else float("inf")),
        optimized_over_control=(control_s / best_s - 1.0
                                if best_s > 0 else float("inf")),
        evaluations=evaluations, iterations=iterations, population=population,
        elite_fraction=elite_fraction, observed_coverage=coverage,
        treatment_diverged=selected_set != control_keys,
        control_overlap=_set_overlap(selected_set, control_keys))
    return ReplayResidencyPlan(
        manifest_sha256=manifest_fingerprint(manifest),
        vram_budget_bytes=baseline_plan.vram_budget_bytes,
        vram_headroom_bytes=baseline_plan.vram_headroom_bytes,
        vram_keys=selected_keys, vram_bytes=selected_bytes,
        report=report, seed=seed)


def optimize_qubo_residency(
        manifest: dict, traces: list[list[TraceEvent]], *,
        vram_budget_gb: float, decode_slice_elems: int = 1 << 25,
        pairwise_candidates: int = 24, restarts: int = 8,
        sweeps: int = 2000, seed: int = 0,
        stream_only: dict | None = None,
        minimum_coverage: float = 0.90) -> ReplayResidencyPlan:
    """Fit a pairwise event-interference QUBO and solve it by annealing.

    A resident tensor is a binary variable.  The linear coefficient is its
    single-tensor counterfactual makespan reduction.  For selected promising
    pairs, the quadratic coefficient is the residual pair benefit after both
    linear effects are removed.  Positive values represent synergy (two spans
    must disappear together to expose a shorter critical path); negative values
    represent redundant benefit.  A quadratic capacity penalty produces the
    QUBO energy, and a classical simulated annealer searches it.

    This is quantum-inspired optimization, not quantum execution.  Every final
    candidate is repaired to the exact byte budget and rescored on the original
    event DAG.  Existing traffic, measured-knapsack and critical-path plans are
    immutable seeds, so the returned simulated plan cannot be worse than the
    best of them.
    """
    if not traces or any(not trace for trace in traces):
        raise ValueError("at least one non-empty calibration trace is required")
    if pairwise_candidates < 2:
        raise ValueError("pairwise_candidates must be >= 2")
    if restarts < 1 or sweeps < 1:
        raise ValueError("restarts and sweeps must be >= 1")

    stream_only = stream_only or {}
    baseline_plan = plan_from_manifest(
        manifest, vram_budget_gb=vram_budget_gb, ram_budget_gb=0.0,
        decode_slice_elems=decode_slice_elems, stream_only=stream_only)
    if not baseline_plan.feasible:
        raise ValueError("infeasible QUBO-search budget: " + baseline_plan.reason)
    available = baseline_plan.vram_budget_bytes - baseline_plan.vram_headroom_bytes
    keys = sorted(key for key, meta in manifest["tensors"].items()
                  if not meta.get("row_gather") and key not in stream_only)
    sizes = np.asarray([int(manifest["tensors"][key]["orig_bytes"])
                        for key in keys], dtype=np.int64)
    if not keys:
        raise ValueError("manifest has no QUBO-search placement candidates")

    event_ids: list[dict[str, tuple[str, ...]]] = []
    observed: set[str] = set()
    for events in traces:
        grouped: dict[str, list[str]] = defaultdict(list)
        for event in events:
            if event.tensor_key in keys and event.kind in PREPARE_KINDS:
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

    def score(mask: np.ndarray) -> float:
        selected = {key for key, keep in zip(keys, mask) if keep}
        return _mean_makespan(traces, selected, event_ids)

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
        mask = np.asarray([key in set(plan.vram_keys) for key in keys], dtype=bool)
        seeded.append((score(mask), policy, mask))
    seeded.sort(key=lambda item: item[0])
    control_s, control_policy, best_mask = seeded[0]
    best_s = control_s
    best_mask = best_mask.copy()
    evaluations = len(seeded) + 1  # include the all-disk replay

    # Linear and pairwise event-DAG counterfactual coefficients.
    linear = np.zeros(len(keys), dtype=np.float64)
    for idx in range(len(keys)):
        if int(sizes[idx]) <= available:
            singleton = np.zeros(len(keys), dtype=bool)
            singleton[idx] = True
            linear[idx] = max(0.0, baseline_s - score(singleton))
            evaluations += 1
    density = linear / np.maximum(sizes.astype(np.float64), 1.0)
    top = list(np.argsort(-density)[:min(pairwise_candidates, len(keys))])
    interactions: dict[tuple[int, int], float] = {}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in keys]
    for pos, left in enumerate(top):
        for right in top[pos + 1:]:
            if int(sizes[left]) + int(sizes[right]) > available:
                continue
            pair = np.zeros(len(keys), dtype=bool)
            pair[left] = pair[right] = True
            pair_gain = max(0.0, baseline_s - score(pair))
            interaction = pair_gain - linear[left] - linear[right]
            evaluations += 1
            if abs(interaction) > 1e-12:
                interactions[(left, right)] = interaction
                adjacency[left].append((right, interaction))
                adjacency[right].append((left, interaction))

    def marginal(idx: int, mask: np.ndarray) -> float:
        return linear[idx] + sum(value for other, value in adjacency[idx]
                                 if mask[other])

    def repair(mask: np.ndarray) -> np.ndarray:
        """Evict back to budget only -- deliberately NOT a greedy top-up.

        An earlier version also backfilled freed capacity by
        marginal(idx)/size(idx), which is exactly critical_path's and
        profiled_knapsack's own ranking function. That made every annealed
        state converge back onto the deterministic control regardless of
        what the anneal actually explored, so the search could never
        diverge from its own seed (see docs/HYPOTHESIS_LINEAGE.md, the
        H13/H15 correction). Any headroom left unused here is real
        information about what the anneal found, not something to paper
        over with the control's own algorithm.
        """
        mask = mask.copy()
        used = int(sizes[mask].sum())
        while used > available:
            chosen = np.flatnonzero(mask)
            idx = min(chosen, key=lambda item: (
                marginal(int(item), mask) / max(int(sizes[item]), 1),
                marginal(int(item), mask)))
            mask[idx] = False
            used -= int(sizes[idx])
        return mask

    rng = np.random.default_rng(seed)
    normalized_sizes = sizes.astype(np.float64) / max(float(available), 1.0)
    coefficient_scale = max(
        baseline_s * 0.02,
        float(np.max(np.abs(linear))) if linear.size else 0.0,
        max((abs(value) for value in interactions.values()), default=0.0),
        1e-6,
    )
    penalty = max(baseline_s, coefficient_scale) * 20.0
    seed_masks = [item[2] for item in seeded]
    fill_probability = min(0.95, available / max(int(sizes.sum()), 1))
    for restart in range(restarts):
        if restart < len(seed_masks):
            mask = seed_masks[restart].copy()
        else:
            mask = repair(rng.random(len(keys)) < fill_probability)
        used_fraction = float(normalized_sizes[mask].sum())
        t_start = coefficient_scale * 2.0
        t_end = max(coefficient_scale * 0.01, 1e-9)
        for step in range(sweeps):
            idx = int(rng.integers(0, len(keys)))
            sign = -1.0 if mask[idx] else 1.0
            delta_benefit = sign * marginal(idx, mask)
            new_used = used_fraction + sign * normalized_sizes[idx]
            old_overflow = max(0.0, used_fraction - 1.0)
            new_overflow = max(0.0, new_used - 1.0)
            delta_energy = (-delta_benefit
                            + penalty * (new_overflow ** 2 - old_overflow ** 2))
            fraction = step / max(sweeps - 1, 1)
            temperature = t_start * ((t_end / t_start) ** fraction)
            if (delta_energy <= 0.0
                    or rng.random() < math.exp(-min(delta_energy / temperature, 700.0))):
                mask[idx] = not mask[idx]
                used_fraction = new_used
        mask = repair(mask)
        candidate_s = score(mask)
        evaluations += 1
        if candidate_s < best_s:
            best_s = candidate_s
            best_mask = mask.copy()

    selected_keys = tuple(key for key, keep in zip(keys, best_mask) if keep)
    control_keys = {key for key, keep in zip(keys, seeded[0][2]) if keep}
    selected_set = set(selected_keys)
    selected_bytes = sum(int(manifest["tensors"][key]["orig_bytes"])
                         for key in selected_keys)
    report = ReplaySearchReport(
        baseline_s=baseline_s, control_s=control_s,
        control_policy=control_policy, optimized_s=best_s,
        predicted_speedup=(baseline_s / best_s if best_s > 0 else float("inf")),
        optimized_over_control=(control_s / best_s - 1.0
                                if best_s > 0 else float("inf")),
        evaluations=evaluations, iterations=sweeps, population=restarts,
        elite_fraction=0.0, observed_coverage=coverage,
        search_method="qubo_anneal",
        treatment_diverged=selected_set != control_keys,
        control_overlap=_set_overlap(selected_set, control_keys))
    return ReplayResidencyPlan(
        manifest_sha256=manifest_fingerprint(manifest),
        vram_budget_bytes=baseline_plan.vram_budget_bytes,
        vram_headroom_bytes=baseline_plan.vram_headroom_bytes,
        vram_keys=selected_keys, vram_bytes=selected_bytes,
        report=report, seed=seed)


def optimize_extent_qubo_residency(
        manifest: dict, traces: list[list[TraceEvent]], *,
        vram_budget_gb: float, decode_slice_elems: int = 1 << 25,
        max_extent_bytes: int = 1 << 28, max_gap_bytes: int = 0,
        max_tensors_per_extent: int = 8, pairwise_candidates: int = 24,
        restarts: int = 8, sweeps: int = 2000, seed: int = 0,
        stream_only: dict | None = None,
        minimum_coverage: float = 0.90) -> ReplayResidencyPlan:
    """Search physical storage extents, then validate on the original DAG.

    Unlike :func:`optimize_qubo_residency`, one binary variable represents a
    bounded contiguous ``weights.bin`` span. Selecting it makes every tensor
    in that span resident. This couples physical request geometry to residency
    while retaining the exact runtime's immutable tensor-level plan artifact.
    The best existing tensor planner remains the control and is returned when
    no genuinely different extent plan beats it in replay.
    """
    if not traces or any(not trace for trace in traces):
        raise ValueError("at least one non-empty calibration trace is required")
    if pairwise_candidates < 2:
        raise ValueError("pairwise_candidates must be >= 2")
    if restarts < 1 or sweeps < 1:
        raise ValueError("restarts and sweeps must be >= 1")

    stream_only = stream_only or {}
    baseline_plan = plan_from_manifest(
        manifest, vram_budget_gb=vram_budget_gb, ram_budget_gb=0.0,
        decode_slice_elems=decode_slice_elems, stream_only=stream_only)
    if not baseline_plan.feasible:
        raise ValueError("infeasible extent-QUBO budget: " + baseline_plan.reason)
    available = baseline_plan.vram_budget_bytes - baseline_plan.vram_headroom_bytes
    keys = sorted(key for key, meta in manifest["tensors"].items()
                  if not meta.get("row_gather") and key not in stream_only)
    if not keys:
        raise ValueError("manifest has no extent-search placement candidates")

    event_ids: list[dict[str, tuple[str, ...]]] = []
    observed: set[str] = set()
    for events in traces:
        grouped: dict[str, list[str]] = defaultdict(list)
        for event in events:
            if event.tensor_key in keys and event.kind in PREPARE_KINDS:
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

    extents = build_storage_extents(
        manifest, keys, max_extent_bytes=max_extent_bytes,
        max_gap_bytes=max_gap_bytes,
        max_tensors_per_extent=max_tensors_per_extent)
    sizes = np.asarray([extent.resident_bytes for extent in extents], dtype=np.int64)

    def selected_from_mask(mask: np.ndarray) -> set[str]:
        return {key for extent, keep in zip(extents, mask) if keep
                for key in extent.keys}

    def score_mask(mask: np.ndarray) -> float:
        return _mean_makespan(traces, selected_from_mask(mask), event_ids)

    baseline_s = _mean_makespan(traces, set(), event_ids)
    measured_profile = CriticalPathProfile.from_traces(traces)
    seed_plans = {"traffic_density": baseline_plan}
    for policy in ("profiled_knapsack", "critical_path"):
        seed_plans[policy] = plan_from_manifest(
            manifest, vram_budget_gb=vram_budget_gb, ram_budget_gb=0.0,
            decode_slice_elems=decode_slice_elems, stream_only=stream_only,
            critical_path_profile=measured_profile, placement_policy=policy)
    seeded_controls = []
    for policy, plan in seed_plans.items():
        selected = set(plan.vram_keys)
        seeded_controls.append((
            _mean_makespan(traces, selected, event_ids), policy, selected))
    seeded_controls.sort(key=lambda item: item[0])
    control_s, control_policy, control_keys = seeded_controls[0]
    best_s = control_s
    best_keys = set(control_keys)
    best_extent_mask: np.ndarray | None = None
    evaluations = len(seeded_controls) + 1

    linear = np.zeros(len(extents), dtype=np.float64)
    for idx, size in enumerate(sizes):
        if int(size) <= available:
            singleton = np.zeros(len(extents), dtype=bool)
            singleton[idx] = True
            linear[idx] = max(0.0, baseline_s - score_mask(singleton))
            evaluations += 1
    density = linear / np.maximum(sizes.astype(np.float64), 1.0)
    top = list(np.argsort(-density)[:min(pairwise_candidates, len(extents))])
    adjacency: list[list[tuple[int, float]]] = [[] for _ in extents]
    interactions: list[float] = []
    for pos, left in enumerate(top):
        for right in top[pos + 1:]:
            if int(sizes[left]) + int(sizes[right]) > available:
                continue
            pair = np.zeros(len(extents), dtype=bool)
            pair[left] = pair[right] = True
            pair_gain = max(0.0, baseline_s - score_mask(pair))
            interaction = pair_gain - linear[left] - linear[right]
            evaluations += 1
            if abs(interaction) > 1e-12:
                adjacency[left].append((right, interaction))
                adjacency[right].append((left, interaction))
                interactions.append(interaction)

    def marginal(idx: int, mask: np.ndarray) -> float:
        return linear[idx] + sum(value for other, value in adjacency[idx]
                                 if mask[other])

    def repair(mask: np.ndarray) -> np.ndarray:
        """Evict back to budget only -- deliberately NOT a greedy top-up.

        See the identical fix and rationale in optimize_qubo_residency's
        repair(): a greedy value/size top-up here would reconstruct
        critical_path's and profiled_knapsack's own ranking and silently
        erase whatever the anneal explored.
        """
        mask = mask.copy()
        used = int(sizes[mask].sum())
        while used > available:
            chosen = np.flatnonzero(mask)
            idx = min(chosen, key=lambda item: (
                marginal(int(item), mask) / max(int(sizes[item]), 1),
                marginal(int(item), mask)))
            mask[idx] = False
            used -= int(sizes[idx])
        return mask

    rng = np.random.default_rng(seed)
    normalized_sizes = sizes.astype(np.float64) / max(float(available), 1.0)
    coefficient_scale = max(
        baseline_s * 0.02,
        float(np.max(np.abs(linear))) if linear.size else 0.0,
        max((abs(value) for value in interactions), default=0.0), 1e-6)
    penalty = max(baseline_s, coefficient_scale) * 20.0
    seed_masks = []
    for _, _, tensor_keys in seeded_controls:
        seed_masks.append(repair(np.asarray([
            set(extent.keys).issubset(tensor_keys) for extent in extents],
            dtype=bool)))
    fill_probability = min(0.95, available / max(int(sizes.sum()), 1))

    for restart in range(restarts):
        if restart < len(seed_masks):
            mask = seed_masks[restart].copy()
        else:
            mask = repair(rng.random(len(extents)) < fill_probability)
        used_fraction = float(normalized_sizes[mask].sum())
        t_start = coefficient_scale * 2.0
        t_end = max(coefficient_scale * 0.01, 1e-9)
        for step in range(sweeps):
            idx = int(rng.integers(0, len(extents)))
            sign = -1.0 if mask[idx] else 1.0
            delta_benefit = sign * marginal(idx, mask)
            new_used = used_fraction + sign * normalized_sizes[idx]
            old_overflow = max(0.0, used_fraction - 1.0)
            new_overflow = max(0.0, new_used - 1.0)
            delta_energy = (-delta_benefit
                            + penalty * (new_overflow ** 2 - old_overflow ** 2))
            fraction = step / max(sweeps - 1, 1)
            temperature = t_start * ((t_end / t_start) ** fraction)
            if (delta_energy <= 0.0
                    or rng.random() < math.exp(-min(delta_energy / temperature, 700.0))):
                mask[idx] = not mask[idx]
                used_fraction = new_used
        mask = repair(mask)
        candidate_s = score_mask(mask)
        evaluations += 1
        candidate_keys = selected_from_mask(mask)
        if candidate_s < best_s - 1e-12 and candidate_keys != control_keys:
            best_s = candidate_s
            best_keys = candidate_keys
            best_extent_mask = mask.copy()

    selected_keys = tuple(sorted(best_keys))
    selected_bytes = sum(int(manifest["tensors"][key]["orig_bytes"])
                         for key in selected_keys)
    selected_groups = (int(best_extent_mask.sum())
                       if best_extent_mask is not None else 0)
    selected_span = (sum(extent.span_bytes for extent, keep
                         in zip(extents, best_extent_mask) if keep)
                     if best_extent_mask is not None else 0)
    report = ReplaySearchReport(
        baseline_s=baseline_s, control_s=control_s,
        control_policy=control_policy, optimized_s=best_s,
        predicted_speedup=(baseline_s / best_s if best_s > 0 else float("inf")),
        optimized_over_control=(control_s / best_s - 1.0
                                if best_s > 0 else float("inf")),
        evaluations=evaluations, iterations=sweeps, population=restarts,
        elite_fraction=0.0, observed_coverage=coverage,
        search_method="extent_qubo_anneal",
        treatment_diverged=best_keys != control_keys,
        control_overlap=_set_overlap(best_keys, control_keys),
        candidate_group_count=len(extents),
        selected_group_count=selected_groups,
        selected_storage_span_bytes=selected_span)
    return ReplayResidencyPlan(
        manifest_sha256=manifest_fingerprint(manifest),
        vram_budget_bytes=baseline_plan.vram_budget_bytes,
        vram_headroom_bytes=baseline_plan.vram_headroom_bytes,
        vram_keys=selected_keys, vram_bytes=selected_bytes,
        report=report, seed=seed)
