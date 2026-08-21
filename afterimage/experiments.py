"""Versioned hypothesis registry and immutable paired experiment results."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import platform
import random
import statistics
import subprocess
import sys
import time
import uuid
from typing import Callable

import numpy as np

from .runtime.config import EngineConfig


@dataclasses.dataclass(frozen=True)
class MethodProfile:
    id: str
    title: str
    overrides: dict
    status: str = "experimental"
    exactness_contract: str = "reference_execution_equivalent"

    def resolve(self, extra_overrides: dict | None = None) -> EngineConfig:
        # The profile is the immutable experimental treatment. Callers may
        # supply shared budgets and paths, but may not accidentally turn the
        # treatment off with a later dictionary merge.
        values = dict(extra_overrides or {})
        values.update(self.overrides)
        return EngineConfig.from_dict(values)


@dataclasses.dataclass(frozen=True)
class Hypothesis:
    id: str
    title: str
    statement: str
    candidate_profile: str
    control_profile: str
    runner: str
    primary_metric: str
    minimum_effect: float
    exactness_contract: str
    required_inputs: tuple[str, ...] = ()
    kill_criterion: str = ""
    minimum_repeats: int = 1
    minimum_new_tokens: int = 1


PROFILES = {
    "exact-streaming-v1": MethodProfile(
        "exact-streaming-v1", "Current exact streaming control", {} , status="stable"),
    "profiled-knapsack-v1": MethodProfile(
        "profiled-knapsack-v1", "Measured-cost residency",
        {"placement_policy": "profiled_knapsack"}),
    "critical-path-v1": MethodProfile(
        "critical-path-v1", "Critical-path residency",
        {"placement_policy": "critical_path"}),
    "hazard-cost-v1": MethodProfile(
        "hazard-cost-v1", "Cost-aware rejection hazard",
        {"draft_mode": "model", "spec_k_policy": "hazard_cost"},
        exactness_contract="distribution_exact"),
    "tuned-fixed-spec-v1": MethodProfile(
        "tuned-fixed-spec-v1", "Tuned fixed speculative chain",
        {"draft_mode": "model", "spec_k_policy": "fixed"}, status="stable",
        exactness_contract="distribution_exact"),
    "pi-prefetch-v1": MethodProfile(
        "pi-prefetch-v1", "PI prefetch controller", {"prefetch_policy": "pi"}),
    "mpc-prefetch-v1": MethodProfile(
        "mpc-prefetch-v1", "One-step MPC prefetch controller",
        {"prefetch_policy": "mpc"}),
    "fixed-prefetch-v1": MethodProfile(
        "fixed-prefetch-v1", "Fixed prefetch control", {}, status="stable"),
    "contextual-linucb-v1": MethodProfile(
        "contextual-linucb-v1", "Baseline-guarded contextual profile bandit",
        {"execution_policy": "linucb"}),
    "certified-mips-v1": MethodProfile(
        "certified-mips-v1", "Certified greedy output-head MIPS",
        {"lm_head_policy": "certified_mips"},
        exactness_contract="greedy_token_exact"),
    "per-tensor-representation-v1": MethodProfile(
        "per-tensor-representation-v1", "Per-tensor exact physical design",
        {"representation_policy": "per_tensor"}),
    "xor-reference-v1": MethodProfile(
        "xor-reference-v1", "Expert-local BitX/XOR reference codec",
        {"expert_codec": "xor_reference"}),
    "model-based-rl-v1": MethodProfile(
        "model-based-rl-v1", "Shadow model-based profile controller",
        {"execution_policy": "model_based_rl"}),
    "ram-overlay-head-v1": MethodProfile(
        "ram-overlay-head-v1", "Liveness-guided RAM overlay for lm_head",
        {"lm_head_policy": "ram_overlay"}),
    "replay-cem-v1": MethodProfile(
        "replay-cem-v1", "Digital-twin CEM residency plan",
        {"placement_policy": "replay_cem"}),
    "neural-utility-spec-v1": MethodProfile(
        "neural-utility-spec-v1", "Tiny neural survival-utility stopping",
        {"draft_mode": "model", "spec_k_policy": "neural_utility"},
        exactness_contract="distribution_exact"),
}


HYPOTHESES = {
    "h0-joint-oracle-gap": Hypothesis(
        "h0-joint-oracle-gap", "Joint semantic/system oracle gap",
        "Joint context has at least 12% value over the best global profile.",
        "contextual-linucb-v1", "exact-streaming-v1", "oracle_gap",
        "joint_uplift", 0.12, "reference_execution_equivalent",
        ("result_dataset",), "Kill adaptive control when the joint oracle gap is <12%."),
    "h1-critical-path": Hypothesis(
        "h1-critical-path", "Critical-path residency planner",
        "Event-DAG criticality ranks residency better than traffic density.",
        "critical-path-v1", "exact-streaming-v1", "generation",
        "committed_tokens_per_second", 0.08, "reference_execution_equivalent",
        ("critical_path_profile",), "Kill if held-out rank correlation <0.8 or gain <8%.",
        minimum_repeats=5, minimum_new_tokens=16),
    "h2-hazard-cost": Hypothesis(
        "h2-hazard-cost", "Cost-aware rejection hazard",
        "Survival-calibrated stopping beats the best tuned fixed chain.",
        "hazard-cost-v1", "tuned-fixed-spec-v1", "generation",
        "committed_tokens_per_second", 0.08, "distribution_exact",
        ("draft_model_id", "spec_policy_state"),
        "Kill if the paired lower confidence bound is not positive.",
        minimum_repeats=5, minimum_new_tokens=128),
    "h3-contextual-bandit": Hypothesis(
        "h3-contextual-bandit", "Baseline-guarded contextual profile bandit",
        "A safe bandit reaches 95% of the joint oracle across requests.",
        "contextual-linucb-v1", "exact-streaming-v1", "profile_bandit",
        "oracle_fraction", 0.95, "reference_execution_equivalent",
        ("calibration_dataset", "result_dataset"),
        "Kill if the oracle itself has <12% headroom."),
    "h4-feedback-prefetch": Hypothesis(
        "h4-feedback-prefetch", "Feedback-controlled prefetch",
        "PI control reduces exposed stalls under changing storage conditions.",
        "pi-prefetch-v1", "fixed-prefetch-v1", "generation",
        "committed_tokens_per_second", 0.05, "reference_execution_equivalent",
        (), "Kill if throughput gain <5% and stall reduction <10%.",
        minimum_repeats=5, minimum_new_tokens=32),
    "h5-certified-mips": Hypothesis(
        "h5-certified-mips", "Certified greedy LM-head search",
        "Roundoff-aware MIPS bounds avoid most output rows with no token changes.",
        "certified-mips-v1", "exact-streaming-v1", "generation",
        "committed_tokens_per_second", 0.08, "greedy_token_exact",
        (), "Kill if certificates cover <70% of rows or any certificate is wrong.",
        minimum_repeats=5, minimum_new_tokens=32),
    "h6-representations": Hypothesis(
        "h6-representations", "Per-tensor exact physical representations",
        "A multi-choice physical design beats every uniform representation.",
        "per-tensor-representation-v1", "exact-streaming-v1", "representation_plan",
        "gain_over_uniform", 0.10, "reference_execution_equivalent",
        ("representation_options", "uniform_prepare_s"),
        "Kill before kernels if predicted gain <10%."),
    "h7-xor-reference": Hypothesis(
        "h7-xor-reference", "Expert-local lossless reference coding",
        "Known BitX-style XOR deltas transfer profitably from model-family "
        "storage to related experts inside one MoE checkpoint.",
        "xor-reference-v1", "exact-streaming-v1", "xor_audit",
        "total_storage_reduction", 0.10, "weight_exact",
        ("expert_tensors", "independent_compressed_bytes", "reference_bases"),
        "Kill if residuals are not at least 10% smaller."),
    "h8-model-based-rl": Hypothesis(
        "h8-model-based-rl", "Shadow model-based joint controller",
        "A calibrated trace simulator closes residual contextual-controller regret.",
        "model-based-rl-v1", "contextual-linucb-v1", "trace_simulator",
        "improvement_over_baseline", 0.10, "reference_execution_equivalent",
        ("trace_dataset",), "Kill if simulator MAPE >10% or rank correlation <0.9."),
    "h9-ram-overlay-head": Hypothesis(
        "h9-ram-overlay-head", "Liveness-guided RAM output-head overlay",
        "A decoded pinned-RAM lm_head beats disk/decode streaming at matched "
        "peak VRAM by exploiting its late, non-overlapping live range.",
        "ram-overlay-head-v1", "exact-streaming-v1", "generation",
        "committed_tokens_per_second", 0.10, "reference_execution_equivalent",
        (), "Kill if gain <10%, peak VRAM rises >5%, or any token differs.",
        minimum_repeats=5, minimum_new_tokens=16),
    "h10-replay-cem": Hypothesis(
        "h10-replay-cem", "Digital-twin whole-set residency search",
        "Offline CEM search over complete resident sets beats independent "
        "profiled-knapsack scores by learning bottleneck-switch interactions.",
        "replay-cem-v1", "profiled-knapsack-v1", "generation",
        "committed_tokens_per_second", 0.08, "reference_execution_equivalent",
        ("replay_plan_state", "critical_path_profile"),
        "Kill if held-out replay error >10% or paired throughput gain <8%.",
        minimum_repeats=5, minimum_new_tokens=16),
    "h11-neural-utility-spec": Hypothesis(
        "h11-neural-utility-spec", "Tiny censored-survival utility controller",
        "A pooled nonlinear survival model trained on cascade feedback chooses "
        "draft stopping points better than a tuned fixed chain.",
        "neural-utility-spec-v1", "tuned-fixed-spec-v1", "generation",
        "committed_tokens_per_second", 0.08, "distribution_exact",
        ("draft_model_id", "spec_policy_state"),
        "Kill if held-out calibration error is poor or the paired lower "
        "confidence bound is not positive.",
        minimum_repeats=5, minimum_new_tokens=128),
}


def registry_payload() -> dict:
    return {
        "schema_version": 1,
        "profiles": [dataclasses.asdict(p) for p in PROFILES.values()],
        "hypotheses": [dataclasses.asdict(h) for h in HYPOTHESES.values()],
    }


@dataclasses.dataclass
class ExperimentRun:
    id: str
    hypothesis_id: str
    status: str
    started_at: float
    completed_at: float | None = None
    candidate_trials: list[dict] = dataclasses.field(default_factory=list)
    control_trials: list[dict] = dataclasses.field(default_factory=list)
    summary: dict = dataclasses.field(default_factory=dict)
    verdict: str = "running"
    metadata: dict = dataclasses.field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class ResultStore:
    def __init__(self, root):
        self.root = pathlib.Path(root)

    def write_once(self, run: ExperimentRun) -> pathlib.Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (run.id + ".json")
        if path.exists():
            raise FileExistsError("immutable experiment result already exists: %s" % path)
        payload = json.dumps(run.to_dict(), indent=2, sort_keys=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
        return path

    def get(self, run_id: str) -> dict | None:
        if len(run_id) != 16 or any(ch not in "0123456789abcdef" for ch in run_id):
            return None
        path = self.root / (run_id + ".json")
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def environment_manifest(repo_root=None) -> dict:
    """Capture reproducibility facts without claiming unavailable hardware."""
    import torch

    commit = None
    if repo_root is not None:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
                text=True, timeout=5, check=False)
            if completed.returncode == 0:
                commit = completed.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "git_commit": commit,
    }


def _bootstrap_paired(candidate: list[float], control: list[float], seed: int = 0,
                      n_bootstrap: int = 2000) -> tuple[float, float]:
    if len(candidate) != len(control) or not candidate:
        raise ValueError("paired bootstrap needs equal non-empty samples")
    ratios = np.asarray(candidate, dtype=np.float64) / np.maximum(control, 1e-12)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        draw = rng.choice(ratios, size=len(ratios), replace=True)
        samples.append(float(np.exp(np.log(draw).mean()) - 1.0))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def run_paired(hypothesis_id: str, executor: Callable[[MethodProfile, int], dict], *,
               repeats: int = 5, seed: int = 0, metadata: dict | None = None,
               progress: Callable[[dict], None] | None = None) -> ExperimentRun:
    hypothesis = HYPOTHESES[hypothesis_id]
    candidate = PROFILES[hypothesis.candidate_profile]
    control = PROFILES[hypothesis.control_profile]
    run = ExperimentRun(uuid.uuid4().hex[:16], hypothesis_id, "running", time.time(),
                        metadata=metadata or {})
    schedule = [("candidate", i) for i in range(repeats)] + [
        ("control", i) for i in range(repeats)]
    random.Random(seed).shuffle(schedule)
    for position, (arm, repeat) in enumerate(schedule):
        profile = candidate if arm == "candidate" else control
        result = executor(profile, repeat)
        result = {**result, "profile_id": profile.id, "repeat": repeat,
                  "order": position}
        (run.candidate_trials if arm == "candidate" else run.control_trials).append(result)
        if progress:
            progress({"phase": "experiment", "completed": position + 1,
                      "total": len(schedule), "arm": arm})

    metric = hypothesis.primary_metric
    cand = [float(row[metric]) for row in sorted(run.candidate_trials,
                                                  key=lambda row: row["repeat"])]
    ctrl = [float(row[metric]) for row in sorted(run.control_trials,
                                                  key=lambda row: row["repeat"])]
    exact = all(bool(row.get("exact", True)) for row in
                run.candidate_trials + run.control_trials)
    candidate_by_repeat = {row["repeat"]: row for row in run.candidate_trials}
    control_by_repeat = {row["repeat"]: row for row in run.control_trials}
    if hypothesis.exactness_contract in (
            "reference_execution_equivalent", "greedy_token_exact", "weight_exact"):
        for repeat in set(candidate_by_repeat) & set(control_by_repeat):
            left = candidate_by_repeat[repeat].get("output_token_ids")
            right = control_by_repeat[repeat].get("output_token_ids")
            if left is not None and right is not None and left != right:
                exact = False
    effect = statistics.median(cand) / max(statistics.median(ctrl), 1e-12) - 1.0
    lo, hi = _bootstrap_paired(cand, ctrl, seed=seed)
    run.summary = {"metric": metric, "candidate_median": statistics.median(cand),
                   "control_median": statistics.median(ctrl), "effect": effect,
                   "ci95": [lo, hi], "exact": exact}
    if not exact:
        run.verdict = "invalid"
    elif lo > 0 and effect >= hypothesis.minimum_effect:
        run.verdict = "favored"
    elif hi < hypothesis.minimum_effect:
        run.verdict = "falsified"
    else:
        run.verdict = "inconclusive"
    run.status = "done"
    run.completed_at = time.time()
    return run


def oracle_gap(rows: list[dict], metric: str = "committed_tokens_per_second") -> dict:
    """Compute global, one-sided and joint profile oracles from logged rows."""
    required = {"profile", "semantic_bucket", "system_bucket", metric}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError("oracle row missing fields: %s" % sorted(missing))

    profiles = sorted({row["profile"] for row in rows})
    global_mean = {profile: statistics.mean(
        float(row[metric]) for row in rows if row["profile"] == profile)
        for profile in profiles}
    global_best = max(global_mean.values())

    def oracle(group_fields):
        groups = {}
        for row in rows:
            group = tuple(row[field] for field in group_fields)
            groups.setdefault(group, {}).setdefault(row["profile"], []).append(float(row[metric]))
        best = [max(statistics.mean(values) for values in profiles_map.values())
                for profiles_map in groups.values()]
        return statistics.mean(best)

    semantic = oracle(("semantic_bucket",))
    system = oracle(("system_bucket",))
    joint = oracle(("semantic_bucket", "system_bucket"))
    return {"global": global_best, "semantic_oracle": semantic,
            "system_oracle": system, "joint_oracle": joint,
            "joint_uplift": joint / max(global_best, 1e-12) - 1.0,
            "joint_over_semantic": joint / max(semantic, 1e-12) - 1.0,
            "joint_over_system": joint / max(system, 1e-12) - 1.0}


def run_fingerprint(hypothesis_id: str, metadata: dict) -> str:
    payload = json.dumps({"hypothesis": hypothesis_id, "metadata": metadata},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
