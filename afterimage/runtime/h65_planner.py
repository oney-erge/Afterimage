"""H6.5: matched-budget, whole-trace representation planning.

Legacy H6 minimized a sum of independently measured tensor costs.  That can
select a plan whose byte total looks good while its many small disk reads
expose a longer prefetch critical path.  H6.5 keeps the same exact live
representations, but scores complete plans against retained event-DAG traces,
uses the ordinary tier planner's transient VRAM reserve, and falls back to the
traffic-density control unless a candidate clears explicit safety gates.

The search is intentionally offline.  The runtime consumes only the frozen
``RepresentationPlan`` returned here.
"""
from __future__ import annotations

import dataclasses
import random
import statistics
from collections import Counter, defaultdict

from .critical_path import TraceEvent, compile_topology, critical_path_fast
from .representations import RepresentationOption, RepresentationPlan
from .vram_planner import (
    DEFAULT_ACTIVATION_SLACK_BYTES, TierPlan, plan_from_manifest,
)


PREPARE_KINDS = frozenset(("read", "io", "decode", "transfer", "copy"))
DISK_NAMES = frozenset(("compressed_disk", "raw_disk"))
RAM_NAMES = frozenset(("compressed_ram", "decoded_ram"))


@dataclasses.dataclass(frozen=True)
class PlanGeometry:
    disk_bytes_per_sweep: int
    disk_read_calls_per_sweep: int
    disk_tensor_count: int
    fully_resident_layers: int
    all_disk_layers: int
    partially_resident_layers: int
    maximum_layer_read_calls: int


@dataclasses.dataclass(frozen=True)
class H65SearchReport:
    control_replay_s: float
    candidate_replay_s: float
    control_objective_s: float
    candidate_objective_s: float
    predicted_improvement: float
    minimum_predicted_improvement: float
    fallback_to_control: bool
    fallback_reason: str
    treatment_diverged: bool
    trace_count: int
    observed_coverage: float
    h2d_gbps: float
    vram_budget_bytes: int
    ram_budget_bytes: int
    vram_headroom_bytes: int
    vram_safety_margin_bytes: int
    persistent_vram_limit_bytes: int
    candidates_scored: int
    fragmentation_penalty_s_per_call: float
    control_geometry: PlanGeometry
    candidate_geometry: PlanGeometry
    control_choices: dict[str, int]
    candidate_choices: dict[str, int]
    guard_failures: tuple[str, ...]
    schema_version: int = 1


@dataclasses.dataclass(frozen=True)
class H65PlanningResult:
    """The guarded deployment plan plus the best diagnostic candidate."""

    plan: RepresentationPlan
    candidate_plan: RepresentationPlan
    report: H65SearchReport


def _disk_name(meta: dict) -> str:
    return "compressed_disk" if meta.get("compressed") else "raw_disk"


def _option_map(manifest: dict, traces: list[list[TraceEvent]],
                h2d_gbps: float) -> dict[str, dict[str, RepresentationOption]]:
    if h2d_gbps <= 0:
        raise ValueError("h2d_gbps must be positive")
    durations: dict[str, Counter] = defaultdict(Counter)
    observations: Counter = Counter()
    for events in traces:
        touched = set()
        for event in events:
            if event.tensor_key and event.kind in PREPARE_KINDS:
                durations[event.tensor_key][event.kind] += event.duration_s
                touched.add(event.tensor_key)
        observations.update(touched)

    result = {}
    for key, meta in manifest["tensors"].items():
        storage = int(meta["comp_bytes"])
        if meta.get("row_gather"):
            result[key] = {
                "row_gather": RepresentationOption(
                    key, "row_gather", storage_bytes=storage)
            }
            continue
        original = int(meta["orig_bytes"])
        count = max(int(observations[key]), 1)
        measured = durations[key]
        read_s = (measured["read"] + measured["io"]) / count
        decode_s = measured["decode"] / count
        transfer_s = max(
            (measured["transfer"] + measured["copy"]) / count,
            original / (h2d_gbps * 1e9),
        )
        disk = _disk_name(meta)
        options = {
            disk: RepresentationOption(
                key, disk, storage_bytes=storage,
                prepare_s=read_s + decode_s + transfer_s),
            "decoded_ram": RepresentationOption(
                key, "decoded_ram", ram_bytes=original,
                storage_bytes=storage, prepare_s=transfer_s),
            "decoded_vram": RepresentationOption(
                key, "decoded_vram", vram_bytes=original,
                storage_bytes=storage, prepare_s=0.0),
        }
        if meta.get("compressed"):
            options["compressed_ram"] = RepresentationOption(
                key, "compressed_ram", ram_bytes=storage,
                storage_bytes=storage, prepare_s=decode_s + transfer_s)
        result[key] = options
    return result


def _choices_from_tier_plan(
        manifest: dict, tier_plan: TierPlan,
        options: dict[str, dict[str, RepresentationOption]]) -> dict[str, RepresentationOption]:
    vram = set(tier_plan.vram_keys)
    ram = set(tier_plan.ram_keys)
    result = {}
    for key, meta in manifest["tensors"].items():
        if meta.get("row_gather"):
            name = "row_gather"
        elif key in vram:
            name = "decoded_vram"
        elif key in ram:
            name = "decoded_ram"
        else:
            name = _disk_name(meta)
        result[key] = options[key][name]
    return result


def _plan_from_choices(
        choices: dict[str, RepresentationOption], *, predicted_s: float,
        vram_budget_bytes: int, ram_budget_bytes: int,
        vram_headroom_bytes: int) -> RepresentationPlan:
    return RepresentationPlan(
        choices=dict(choices),
        vram_bytes=sum(option.vram_bytes for option in choices.values()),
        ram_bytes=sum(option.ram_bytes for option in choices.values()),
        storage_bytes=sum(option.storage_bytes for option in choices.values()),
        predicted_prepare_s=float(predicted_s),
        vram_budget_bytes=vram_budget_bytes,
        ram_budget_bytes=ram_budget_bytes,
        vram_headroom_bytes=vram_headroom_bytes,
    )


def build_uniform_disk_plan(
        manifest: dict, *, vram_budget_gb: float, ram_budget_gb: float,
        decode_slice_elems: int = 1 << 22,
        vram_safety_margin_gb: float = 0.0) -> RepresentationPlan:
    """Build a trace-calibration control with correct transient headroom."""
    tier = plan_from_manifest(
        manifest, vram_budget_gb=vram_budget_gb,
        ram_budget_gb=ram_budget_gb,
        decode_slice_elems=decode_slice_elems,
        activation_slack_bytes=(
            DEFAULT_ACTIVATION_SLACK_BYTES
            + int(vram_safety_margin_gb * 1e9)))
    if not tier.feasible:
        raise ValueError("infeasible H6.5 disk control: " + tier.reason)
    choices = {}
    for key, meta in manifest["tensors"].items():
        name = "row_gather" if meta.get("row_gather") else _disk_name(meta)
        choices[key] = RepresentationOption(
            key, name, storage_bytes=int(meta["comp_bytes"]))
    return _plan_from_choices(
        choices, predicted_s=0.0,
        vram_budget_bytes=tier.vram_budget_bytes,
        ram_budget_bytes=tier.ram_budget_bytes,
        vram_headroom_bytes=tier.vram_headroom_bytes)


def _layer_index(key: str) -> int | None:
    if not key.startswith("model.layers."):
        return None
    try:
        return int(key.split(".")[2])
    except (IndexError, ValueError):
        return None


def _mutation_group(key: str) -> str:
    layer = _layer_index(key)
    if layer is None:
        return key
    if ".self_attn." in key:
        family = "attention"
    elif ".mlp." in key:
        family = "mlp"
    else:
        family = "aux"
    return "layer-%d/%s" % (layer, family)


def plan_geometry(manifest: dict,
                  choices: dict[str, RepresentationOption]) -> PlanGeometry:
    disk = {key for key, option in choices.items() if option.name in DISK_NAMES}
    read_calls = {
        key: len(manifest["tensors"][key].get("blobs", {})) for key in disk
    }
    layers: dict[int, set[str]] = defaultdict(set)
    for key in manifest["tensors"]:
        layer = _layer_index(key)
        if layer is not None:
            layers[layer].add(key)
    full = all_disk = partial = 0
    maximum = 0
    for keys in layers.values():
        disk_keys = keys & disk
        calls = sum(read_calls.get(key, 0) for key in disk_keys)
        maximum = max(maximum, calls)
        if not disk_keys:
            full += 1
        elif disk_keys == keys:
            all_disk += 1
        else:
            partial += 1
    return PlanGeometry(
        disk_bytes_per_sweep=sum(
            int(manifest["tensors"][key]["comp_bytes"]) for key in disk),
        disk_read_calls_per_sweep=sum(read_calls.values()),
        disk_tensor_count=len(disk),
        fully_resident_layers=full,
        all_disk_layers=all_disk,
        partially_resident_layers=partial,
        maximum_layer_read_calls=maximum,
    )


def _guard_failures(control: PlanGeometry, candidate: PlanGeometry) -> tuple[str, ...]:
    failures = []
    if candidate.disk_bytes_per_sweep > control.disk_bytes_per_sweep:
        failures.append("disk_bytes_increased")
    if candidate.disk_read_calls_per_sweep > control.disk_read_calls_per_sweep:
        failures.append("disk_read_calls_increased")
    if candidate.maximum_layer_read_calls > control.maximum_layer_read_calls:
        failures.append("maximum_layer_read_calls_increased")
    if candidate.fully_resident_layers < control.fully_resident_layers:
        failures.append("fewer_fully_resident_layers")
    if candidate.all_disk_layers > control.all_disk_layers:
        failures.append("more_all_disk_layers")
    return tuple(failures)


class _ReplayScorer:
    def __init__(self, manifest: dict, traces: list[list[TraceEvent]], *,
                 h2d_gbps: float, fragmentation_penalty_weight: float):
        self.manifest = manifest
        self.traces = traces
        self.h2d_gbps = h2d_gbps
        self.topologies = [compile_topology(events) for events in traces]
        self.by_tensor = []
        read_call_samples = []
        observed = set()
        for events in traces:
            grouped: dict[str, dict[str, list[TraceEvent]]] = defaultdict(
                lambda: defaultdict(list))
            for event in events:
                if event.tensor_key and event.kind in PREPARE_KINDS:
                    grouped[event.tensor_key][event.kind].append(event)
                    observed.add(event.tensor_key)
                    if event.kind in ("read", "io"):
                        calls = max(1, len(manifest["tensors"][event.tensor_key].get(
                            "blobs", {})))
                        read_call_samples.append(event.duration_s / calls)
            self.by_tensor.append(grouped)
        eligible = {
            key for key, meta in manifest["tensors"].items()
            if not meta.get("row_gather")
        }
        self.observed = observed & eligible
        self.coverage = len(self.observed) / max(len(eligible), 1)
        measured_call_s = (
            statistics.median(read_call_samples) if read_call_samples else 0.0)
        self.call_penalty_s = measured_call_s * fragmentation_penalty_weight
        self.evaluations = 0
        self._cache = {}

    def _fingerprint(self, choices: dict[str, RepresentationOption]) -> tuple:
        return tuple((key, choices[key].name) for key in sorted(choices))

    def replay(self, choices: dict[str, RepresentationOption]) -> float:
        fingerprint = self._fingerprint(choices)
        cached = self._cache.get(fingerprint)
        if cached is not None:
            return cached
        durations = []
        for topology, grouped in zip(self.topologies, self.by_tensor):
            overrides = {}
            for key, option in choices.items():
                events = grouped.get(key)
                if not events or option.name in DISK_NAMES or option.name == "row_gather":
                    continue
                if option.name == "compressed_ram":
                    for kind in ("read", "io"):
                        overrides.update((event.id, 0.0) for event in events.get(kind, ()))
                    continue
                if option.name == "decoded_vram":
                    for kind in PREPARE_KINDS:
                        overrides.update((event.id, 0.0) for event in events.get(kind, ()))
                    continue
                if option.name != "decoded_ram":
                    raise ValueError("unsupported H6.5 choice %r" % option.name)
                for kind in PREPARE_KINDS:
                    overrides.update((event.id, 0.0) for event in events.get(kind, ()))
                floor = int(self.manifest["tensors"][key]["orig_bytes"]) / (
                    self.h2d_gbps * 1e9)
                anchors = events.get("transfer", ()) or events.get("copy", ())
                if not anchors:
                    anchors = events.get("decode", ())
                if not anchors:
                    anchors = events.get("read", ()) or events.get("io", ())
                for event in anchors:
                    overrides[event.id] = max(event.duration_s, floor)
            durations.append(critical_path_fast(topology, overrides).duration_s)
        value = statistics.mean(durations)
        self._cache[fingerprint] = value
        self.evaluations += 1
        return value

    def objective(self, choices: dict[str, RepresentationOption]) -> float:
        geometry = plan_geometry(self.manifest, choices)
        return self.replay(choices) + (
            self.call_penalty_s * geometry.disk_read_calls_per_sweep)


def _usage(choices: dict[str, RepresentationOption]) -> tuple[int, int]:
    return (
        sum(option.vram_bytes for option in choices.values()),
        sum(option.ram_bytes for option in choices.values()),
    )


def _choice_counts(choices: dict[str, RepresentationOption]) -> dict[str, int]:
    return dict(sorted(Counter(option.name for option in choices.values()).items()))


def optimize_h65_plan(
        manifest: dict, traces: list[list[TraceEvent]], *,
        vram_budget_gb: float, ram_budget_gb: float, h2d_gbps: float,
        decode_slice_elems: int = 1 << 22, search_iterations: int = 512,
        seed: int = 0, minimum_coverage: float = 0.90,
        minimum_predicted_improvement: float = 0.08,
        fragmentation_penalty_weight: float = 0.0,
        vram_safety_margin_gb: float = 0.0) -> H65PlanningResult:
    """Return a guarded H6.5 plan and the best divergent diagnostic plan.

    The search starts from the exact traffic-density plan, adds exact-byte
    greedy multi-representation seeds, then mutates attention/MLP/auxiliary
    bundles and scores every feasible complete plan by event-DAG replay.
    Read duration and overlap are already represented in that replay, so the
    additional per-call fragmentation term defaults to zero.  A nonzero value
    is valid only when it was calibrated on the same storage/runtime path.
    Deployment falls back to the control unless the best candidate both
    passes locality guards and clears ``minimum_predicted_improvement``.
    """
    if not traces or any(not trace for trace in traces):
        raise ValueError("at least one non-empty raw calibration trace is required")
    if search_iterations < 0:
        raise ValueError("search_iterations must be non-negative")
    if not 0.0 <= minimum_predicted_improvement < 1.0:
        raise ValueError("minimum_predicted_improvement must be in [0, 1)")
    if not 0.0 <= fragmentation_penalty_weight <= 1.0:
        raise ValueError("fragmentation_penalty_weight must be in [0, 1]")
    if vram_safety_margin_gb < 0:
        raise ValueError("vram_safety_margin_gb must be non-negative")

    safety_margin_bytes = int(vram_safety_margin_gb * 1e9)

    control_tier = plan_from_manifest(
        manifest, vram_budget_gb=vram_budget_gb,
        ram_budget_gb=ram_budget_gb,
        decode_slice_elems=decode_slice_elems,
        activation_slack_bytes=(
            DEFAULT_ACTIVATION_SLACK_BYTES + safety_margin_bytes),
        placement_policy="traffic_density")
    if not control_tier.feasible:
        raise ValueError("infeasible H6.5 control: " + control_tier.reason)
    options = _option_map(manifest, traces, h2d_gbps)
    control = _choices_from_tier_plan(manifest, control_tier, options)
    disk = {
        key: option_set["row_gather" if manifest["tensors"][key].get("row_gather")
                        else _disk_name(manifest["tensors"][key])]
        for key, option_set in options.items()
    }
    scorer = _ReplayScorer(
        manifest, traces, h2d_gbps=h2d_gbps,
        fragmentation_penalty_weight=fragmentation_penalty_weight)
    if scorer.coverage < minimum_coverage:
        missing = sorted(
            key for key, meta in manifest["tensors"].items()
            if not meta.get("row_gather") and key not in scorer.observed)
        raise ValueError(
            "raw H6.5 traces cover %.1f%% of candidates, below %.1f%%; "
            "missing examples: %s" %
            (100 * scorer.coverage, 100 * minimum_coverage, ", ".join(missing[:5])))

    vram_limit = control_tier.vram_budget_bytes - control_tier.vram_headroom_bytes
    ram_limit = control_tier.ram_budget_bytes
    control_geometry = plan_geometry(manifest, control)
    control_replay = scorer.replay(control)
    control_objective = scorer.objective(control)
    all_disk_replay = scorer.replay(disk)

    # Single-choice counterfactuals are used only to seed and repair plans;
    # final ranking always replays the complete DAG.
    independent_benefit = {}
    for key in sorted(scorer.observed):
        for name, option in options[key].items():
            if name in DISK_NAMES or name == "row_gather":
                independent_benefit[(key, name)] = 0.0
                continue
            trial = dict(disk)
            trial[key] = option
            independent_benefit[(key, name)] = max(
                0.0, all_disk_replay - scorer.replay(trial))

    def density(key: str, name: str) -> float:
        option = options[key][name]
        memory = option.vram_bytes + option.ram_bytes
        return independent_benefit.get((key, name), 0.0) / max(memory, 1)

    def feasible(choices) -> bool:
        used_vram, used_ram = _usage(choices)
        return used_vram <= vram_limit and used_ram <= ram_limit

    def guarded(choices) -> bool:
        return not _guard_failures(control_geometry, plan_geometry(manifest, choices))

    def fill(seed_choices, order: str):
        choices = dict(disk)

        def add_vram():
            used_vram, _ = _usage(choices)
            ranked = sorted(
                (key for key in scorer.observed if "decoded_vram" in options[key]),
                key=lambda key: (density(key, "decoded_vram"),
                                 independent_benefit[(key, "decoded_vram")]),
                reverse=True)
            for key in ranked:
                option = options[key]["decoded_vram"]
                current = choices[key]
                if used_vram + option.vram_bytes <= vram_limit:
                    choices[key] = option
                    used_vram += option.vram_bytes
                    # Moving RAM -> VRAM frees RAM for the second fill pass.
                    if current.name in RAM_NAMES:
                        pass

        def add_ram():
            _, used_ram = _usage(choices)
            candidates = []
            for key in scorer.observed:
                if choices[key].name == "decoded_vram":
                    continue
                for name in RAM_NAMES & set(options[key]):
                    candidates.append((density(key, name),
                                       independent_benefit[(key, name)], key, name))
            for _ratio, _benefit, key, name in sorted(candidates, reverse=True):
                option = options[key][name]
                current = choices[key]
                delta = option.ram_bytes - current.ram_bytes
                if delta >= 0 and used_ram + delta <= ram_limit:
                    choices[key] = option
                    used_ram += delta

        if order == "vram-first":
            add_vram()
            add_ram()
        else:
            add_ram()
            add_vram()
            add_ram()
        return choices

    seeds = [dict(control), fill(disk, "vram-first"), fill(disk, "ram-first")]
    valid = []
    for choices in seeds:
        if feasible(choices) and guarded(choices):
            valid.append((scorer.objective(choices), dict(choices)))
    valid.sort(key=lambda item: item[0])
    best = valid[0][1] if valid else dict(control)
    best_objective = scorer.objective(best)

    def consider(proposal):
        nonlocal best, best_objective
        if not feasible(proposal) or not guarded(proposal):
            return
        objective = scorer.objective(proposal)
        if objective < best_objective:
            best, best_objective = dict(proposal), objective

    # A full budget often prevents a useful tensor from being promoted unless
    # another tensor leaves the same tier in the same proposal.  Check the
    # strongest exact-byte one-for-one swaps deterministically before the
    # stochastic bundle search.  This closes the most important local-search
    # blind spot without attempting an intractable exhaustive enumeration.
    for _pass in range(2):
        base = dict(best)
        for target_name in ("decoded_vram", "decoded_ram", "compressed_ram"):
            if target_name == "decoded_vram":
                residents = [
                    key for key, option in base.items()
                    if option.name == target_name
                ]
            else:
                residents = [
                    key for key, option in base.items()
                    if option.name in RAM_NAMES
                ]
            promotions = [
                key for key, option in base.items()
                if option.name in DISK_NAMES and target_name in options[key]
            ]
            residents = sorted(
                residents,
                key=lambda key: independent_benefit.get(
                    (key, base[key].name), 0.0))[:32]
            promotions = sorted(
                promotions,
                key=lambda key: independent_benefit.get(
                    (key, target_name), 0.0),
                reverse=True)[:32]
            for promoted in promotions:
                proposal = dict(base)
                proposal[promoted] = options[promoted][target_name]
                consider(proposal)
                for demoted in residents:
                    proposal = dict(base)
                    proposal[demoted] = disk[demoted]
                    proposal[promoted] = options[promoted][target_name]
                    consider(proposal)

    groups: dict[str, list[str]] = defaultdict(list)
    for key in scorer.observed:
        groups[_mutation_group(key)].append(key)
    group_names = sorted(groups)
    rng = random.Random(seed)

    def repair(choices):
        choices = dict(choices)
        while _usage(choices)[0] > vram_limit:
            resident = [key for key, option in choices.items()
                        if option.name == "decoded_vram"]
            if not resident:
                break
            key = min(resident, key=lambda item: density(item, "decoded_vram"))
            choices[key] = disk[key]
        while _usage(choices)[1] > ram_limit:
            resident = [key for key, option in choices.items()
                        if option.name in RAM_NAMES]
            if not resident:
                break
            key = min(resident, key=lambda item: density(item, choices[item].name))
            choices[key] = disk[key]
        return choices

    # Bundle mutations let attention/MLP/auxiliary tensors move together;
    # two groups per proposal permit exact-byte swaps under full budgets.
    for _ in range(search_iterations):
        proposal = dict(best if rng.random() < 0.8 else rng.choice(seeds))
        for _move in range(1 if rng.random() < 0.65 else 2):
            group = groups[rng.choice(group_names)]
            target_family = rng.choice(
                ("disk", "compressed_ram", "decoded_ram", "decoded_vram"))
            for key in group:
                if target_family == "disk":
                    proposal[key] = disk[key]
                elif target_family in options[key]:
                    proposal[key] = options[key][target_family]
        proposal = repair(proposal)
        consider(proposal)

    candidate_replay = scorer.replay(best)
    candidate_geometry = plan_geometry(manifest, best)
    failures = _guard_failures(control_geometry, candidate_geometry)
    improvement = (
        (control_replay - candidate_replay) / control_replay
        if control_replay > 0 else 0.0)
    diverged = any(best[key].name != control[key].name for key in control)
    fallback_reason = ""
    if failures:
        fallback_reason = "candidate failed locality guards: " + ", ".join(failures)
    elif not diverged:
        fallback_reason = "search found no guarded plan better than traffic placement"
    elif improvement < minimum_predicted_improvement:
        fallback_reason = (
            "predicted improvement %.2f%% is below the %.2f%% deployment gate"
            % (100 * improvement, 100 * minimum_predicted_improvement))
    fallback = bool(fallback_reason)
    deployment = control if fallback else best
    deployment_replay = control_replay if fallback else candidate_replay

    candidate_plan = _plan_from_choices(
        best, predicted_s=candidate_replay,
        vram_budget_bytes=control_tier.vram_budget_bytes,
        ram_budget_bytes=control_tier.ram_budget_bytes,
        vram_headroom_bytes=control_tier.vram_headroom_bytes)
    plan = _plan_from_choices(
        deployment, predicted_s=deployment_replay,
        vram_budget_bytes=control_tier.vram_budget_bytes,
        ram_budget_bytes=control_tier.ram_budget_bytes,
        vram_headroom_bytes=control_tier.vram_headroom_bytes)
    report = H65SearchReport(
        control_replay_s=control_replay,
        candidate_replay_s=candidate_replay,
        control_objective_s=control_objective,
        candidate_objective_s=best_objective,
        predicted_improvement=improvement,
        minimum_predicted_improvement=minimum_predicted_improvement,
        fallback_to_control=fallback,
        fallback_reason=fallback_reason,
        treatment_diverged=diverged,
        trace_count=len(traces), observed_coverage=scorer.coverage,
        h2d_gbps=h2d_gbps,
        vram_budget_bytes=control_tier.vram_budget_bytes,
        ram_budget_bytes=control_tier.ram_budget_bytes,
        vram_headroom_bytes=control_tier.vram_headroom_bytes,
        vram_safety_margin_bytes=safety_margin_bytes,
        persistent_vram_limit_bytes=vram_limit,
        candidates_scored=scorer.evaluations,
        fragmentation_penalty_s_per_call=scorer.call_penalty_s,
        control_geometry=control_geometry,
        candidate_geometry=candidate_geometry,
        control_choices=_choice_counts(control),
        candidate_choices=_choice_counts(best),
        guard_failures=failures,
    )
    return H65PlanningResult(plan, candidate_plan, report)
