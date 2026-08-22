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
class MeasuredOutcome:
    """What was actually measured, in the words a non-researcher needs, not
    just the numeric gate. Single-sourced from docs/RESULTS_LOG.md and
    docs/FINAL_TEST_RESULTS_2026-08-21.md so the UI's Lab cards and the
    written record can never silently drift apart.

    verdict is one of:
      "gate"            -- a diagnostic gate, not a speed treatment; it did its job
      "below_threshold" -- measured a real but sub-gate positive direction
      "positive_screen" -- passed its registered L1 mechanism/effect gate,
                            but is not yet a scale-matched L3 result
      "contradicted"    -- measured slower/worse than its named control
      "mechanism_only"  -- the mechanism worked exactly as designed; the
                            end-to-end objective regressed anyway
      "no_comparison"   -- an upstream mechanism/action/environment gate
                            stopped it before a valid speed comparison existed
      "not_applicable"  -- this checkpoint/host cannot exercise the hypothesis
    """
    verdict: str
    plain_language: str
    detail: str = ""
    effect_pct: float | None = None
    n_pairs: int | None = None


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
    measured: MeasuredOutcome | None = None


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
    "bayes-probit-prefetch-v1": MethodProfile(
        "bayes-probit-prefetch-v1", "Bayesian probit prefetch depth",
        {"prefetch_policy": "bayes_probit"}),
    "replay-qubo-v1": MethodProfile(
        "replay-qubo-v1", "Pairwise event-interference QUBO residency",
        {"placement_policy": "replay_qubo"}),
    "coalesced-storage-v1": MethodProfile(
        "coalesced-storage-v1", "Bounded contiguous storage reads",
        {"storage_read_policy": "coalesced_extents"}),
    "extent-qubo-v1": MethodProfile(
        "extent-qubo-v1", "Physical-extent QUBO residency",
        {"placement_policy": "replay_extent_qubo"}),
    "spec-critical-path-v1": MethodProfile(
        "spec-critical-path-v1", "Critical-path residency with fixed speculation",
        {"placement_policy": "critical_path", "draft_mode": "model",
         "spec_k": 8, "spec_k_policy": "fixed"},
        exactness_contract="distribution_exact"),
    "tensor-extents-v1": MethodProfile(
        "tensor-extents-v1", "Tensor-scoped micro-extents",
        {"storage_read_policy": "tensor_extents",
         "storage_extent_max_bytes": 1 << 23,
         "storage_extent_max_gap_bytes": 0}),
    "rollback-cached-spec-v1": MethodProfile(
        "rollback-cached-spec-v1", "Rollback-cached target speculation",
        {"draft_mode": "model", "spec_k": 8, "spec_k_policy": "fixed",
         "spec_target_cache": True},
        exactness_contract="distribution_exact"),
}


HYPOTHESES = {
    "h0-joint-oracle-gap": Hypothesis(
        "h0-joint-oracle-gap", "Joint semantic/system oracle gap",
        "Joint context has at least 12% value over the best global profile.",
        "contextual-linucb-v1", "exact-streaming-v1", "oracle_gap",
        "joint_uplift", 0.12, "reference_execution_equivalent",
        ("result_dataset",), "Kill adaptive control when the joint oracle gap is <12%.",
        measured=MeasuredOutcome(
            "gate",
            "Not a speedup itself -- this measurement is what told us the "
            "other adaptive-control ideas (H3, H8) weren't worth building.",
            "Measured joint oracle uplift: 2.56%, against a 12% gate.",
            effect_pct=2.56)),
    "h1-critical-path": Hypothesis(
        "h1-critical-path", "Critical-path residency planner",
        "Event-DAG criticality ranks residency better than traffic density.",
        "critical-path-v1", "exact-streaming-v1", "generation",
        "committed_tokens_per_second", 0.08, "reference_execution_equivalent",
        ("critical_path_profile",), "Kill if held-out rank correlation <0.8 or gain <8%.",
        minimum_repeats=5, minimum_new_tokens=16,
        measured=MeasuredOutcome(
            "below_threshold",
            "No -- it's the strongest exact non-speculative candidate measured "
            "so far, but the gain is too small to promise you over the "
            "simpler traffic-density default.",
            "18.99 s/token, 1.58x AirLLM at 4 GB -- the best exact "
            "non-speculative row seen, but not confirmed at L3 yet.",
            effect_pct=1.61)),
    "h2-hazard-cost": Hypothesis(
        "h2-hazard-cost", "Cost-aware rejection hazard",
        "Survival-calibrated stopping beats the best tuned fixed chain.",
        "hazard-cost-v1", "tuned-fixed-spec-v1", "generation",
        "committed_tokens_per_second", 0.08, "distribution_exact",
        ("draft_model_id", "spec_policy_state"),
        "Kill if the paired lower confidence bound is not positive.",
        minimum_repeats=5, minimum_new_tokens=128,
        measured=MeasuredOutcome(
            "contradicted",
            "No -- measured slower than just fixing the draft chain length. "
            "Use fixed-k speculation instead.",
            "9.773 vs fixed-k's 9.150 s/token: 6.4% lower throughput.",
            effect_pct=-6.4)),
    "h3-contextual-bandit": Hypothesis(
        "h3-contextual-bandit", "Baseline-guarded contextual profile bandit",
        "A safe bandit reaches 95% of the joint oracle across requests.",
        "contextual-linucb-v1", "exact-streaming-v1", "profile_bandit",
        "oracle_fraction", 0.95, "reference_execution_equivalent",
        ("calibration_dataset", "result_dataset"),
        "Kill if the oracle itself has <12% headroom.",
        measured=MeasuredOutcome(
            "below_threshold",
            "It ran, but learned to use the same fixed policy as the control; "
            "there was no throughput improvement to capture.",
            "Four-fold held-out replay reached 97.50% of the oracle, while "
            "chosen reward equaled baseline reward exactly. H0 limits the "
            "available oracle headroom to 2.56%.")),
    "h4-feedback-prefetch": Hypothesis(
        "h4-feedback-prefetch", "Feedback-controlled prefetch",
        "PI control reduces exposed stalls under changing storage conditions.",
        "pi-prefetch-v1", "fixed-prefetch-v1", "generation",
        "committed_tokens_per_second", 0.05, "reference_execution_equivalent",
        (), "Kill if throughput gain <5% and stall reduction <10%.",
        minimum_repeats=5, minimum_new_tokens=32,
        measured=MeasuredOutcome(
            "contradicted",
            "No -- feedback-controlled prefetch made things worse, one "
            "variant much worse. Leave prefetch depth fixed.",
            "PI control was 35.7% slower; MPC regressed further by "
            "sometimes choosing a prefetch depth of zero.",
            effect_pct=-35.7)),
    "h5-certified-mips": Hypothesis(
        "h5-certified-mips", "Certified greedy LM-head search",
        "Roundoff-aware MIPS bounds avoid most output rows with no token changes.",
        "certified-mips-v1", "exact-streaming-v1", "generation",
        "committed_tokens_per_second", 0.08, "greedy_token_exact",
        (), "Kill if certificates cover <70% of rows or any certificate is wrong.",
        minimum_repeats=5, minimum_new_tokens=32,
        measured=MeasuredOutcome(
            "contradicted",
            "No -- almost nothing got pruned, and the bookkeeping to try "
            "cost more than it saved.",
            "Only 0.084% of output rows were certifiably prunable; 30.5% "
            "lower throughput than the plain full head.",
            effect_pct=-30.5)),
    "h6-representations": Hypothesis(
        "h6-representations", "Per-tensor exact physical representations",
        "A multi-choice physical design beats every uniform representation.",
        "per-tensor-representation-v1", "exact-streaming-v1", "representation_plan",
        "gain_over_uniform", 0.10, "reference_execution_equivalent",
        ("representation_options", "uniform_prepare_s"),
        "Kill before kernels if predicted gain <10%.",
        measured=MeasuredOutcome(
            "positive_screen",
            "The offline mixed-representation plan passed its prediction "
            "gate, but it has not passed a held-out live execution.",
            "Across 441 real tensors, the exact DP predicted 15.01 s of "
            "preparation versus 24.42 s for uniform disk (38.56% lower), "
            "using 3.96 GB VRAM and 7.99 GB RAM. Pinned H2D was independently "
            "measured at 6.669 GB/s.", effect_pct=38.56)),
    "h7-xor-reference": Hypothesis(
        "h7-xor-reference", "Expert-local lossless reference coding",
        "Known BitX-style XOR deltas transfer profitably from model-family "
        "storage to related experts inside one MoE checkpoint.",
        "xor-reference-v1", "exact-streaming-v1", "xor_audit",
        "total_storage_reduction", 0.10, "weight_exact",
        ("expert_tensors", "independent_compressed_bytes", "reference_bases"),
        "Kill if residuals are not at least 10% smaller.",
        measured=MeasuredOutcome(
            "contradicted",
            "The codec is exact, but real experts were less compressible as "
            "XOR deltas than independently.",
            "Eight Qwen1.5-MoE layer-0 experts round-tripped bit-for-bit. "
            "Forced XOR-reference storage was 2.24% larger, so the safe "
            "chooser fell back to independent storage and saved 0%.",
            effect_pct=0.0)),
    "h8-model-based-rl": Hypothesis(
        "h8-model-based-rl", "Shadow model-based joint controller",
        "A calibrated trace simulator closes residual contextual-controller regret.",
        "model-based-rl-v1", "contextual-linucb-v1", "trace_simulator",
        "improvement_over_baseline", 0.10, "reference_execution_equivalent",
        ("trace_dataset",), "Kill if simulator MAPE >10% or rank correlation <0.9.",
        measured=MeasuredOutcome(
            "contradicted",
            "The shadow simulator ran and failed its calibration gate; its "
            "chosen policy improved only 0.33%.",
            "Two real held-out CEM/control timing pairs produced 11.87% MAPE "
            "and -0.80 rank correlation, versus required <=10% and >=0.90.",
            effect_pct=0.33, n_pairs=2)),
    "h9-ram-overlay-head": Hypothesis(
        "h9-ram-overlay-head", "Liveness-guided RAM output-head overlay",
        "A decoded pinned-RAM lm_head beats disk/decode streaming at matched "
        "peak VRAM by exploiting its late, non-overlapping live range.",
        "ram-overlay-head-v1", "exact-streaming-v1", "generation",
        "committed_tokens_per_second", 0.10, "reference_execution_equivalent",
        (), "Kill if gain <10%, peak VRAM rises >5%, or any token differs.",
        minimum_repeats=5, minimum_new_tokens=16,
        measured=MeasuredOutcome(
            "positive_screen",
            "The pinned output-head mechanism passed at a scale this WSL2 "
            "host can genuinely pin; the 14B head still exceeds that ceiling.",
            "On Qwen3-0.6B, 1.279 versus 1.809 s/token is 41.4% higher "
            "throughput at matched 0.419 GB peak VRAM, with identical tokens "
            "and no pageable fallback. It reached 3.33x AirLLM. The 1.556 GB "
            "14B head cannot be pinned under this host's ~1 GB WSL2 ceiling.",
            effect_pct=41.4)),
    "h10-replay-cem": Hypothesis(
        "h10-replay-cem", "Digital-twin whole-set residency search",
        "Offline CEM search over complete resident sets beats independent "
        "profiled-knapsack scores by learning bottleneck-switch interactions.",
        "replay-cem-v1", "profiled-knapsack-v1", "generation",
        "committed_tokens_per_second", 0.08, "reference_execution_equivalent",
        ("replay_plan_state", "critical_path_profile"),
        "Kill if held-out replay error >10% or paired throughput gain <8%.",
        minimum_repeats=5, minimum_new_tokens=16,
        measured=MeasuredOutcome(
            "below_threshold",
            "No -- the search landed at parity with the simpler measured-"
            "cost planner, not ahead of it.",
            "19.480 versus 19.554 s/token in the common screen (+0.38%); "
            "a one-token pilot showed +2.1%. Both are below the 8% gate.",
            effect_pct=0.38)),
    "h11-neural-utility-spec": Hypothesis(
        "h11-neural-utility-spec", "Tiny censored-survival utility controller",
        "A pooled nonlinear survival model trained on cascade feedback chooses "
        "draft stopping points better than a tuned fixed chain.",
        "neural-utility-spec-v1", "tuned-fixed-spec-v1", "generation",
        "committed_tokens_per_second", 0.08, "distribution_exact",
        ("draft_model_id", "spec_policy_state"),
        "Kill if held-out calibration error is poor or the paired lower "
        "confidence bound is not positive.",
        minimum_repeats=5, minimum_new_tokens=128,
        measured=MeasuredOutcome(
            "no_comparison",
            "No -- it never actually decided anything differently than "
            "just fixing the draft length, so its timing can't be trusted.",
            "Well-calibrated (88% positive labels, Brier 0.115) but chose "
            "to keep drafting in all 47 held-out opportunities -- zero stop "
            "decisions, so it behaves exactly like fixed-k. Any timing "
            "difference is noise, not the network.")),
    "h12-bayesian-prefetch": Hypothesis(
        "h12-bayesian-prefetch", "Bayesian chance-constrained prefetch",
        "A probit constraint over posterior read and lead-time distributions "
        "reduces exposed stalls without the PI controller's over-prefetching.",
        "bayes-probit-prefetch-v1", "fixed-prefetch-v1", "generation",
        "committed_tokens_per_second", 0.05, "reference_execution_equivalent",
        (), "Kill if throughput gain <5% or exposed wait does not fall by 10%.",
        minimum_repeats=5, minimum_new_tokens=32,
        measured=MeasuredOutcome(
            "contradicted",
            "No -- the regulated rerun was slower, and it waited on I/O "
            "more, not less.",
            "5.89% slower paired (8 blocks); won only 3 of 8 pairs; exposed "
            "wait rose 28.2% instead of falling.",
            effect_pct=-5.89, n_pairs=8)),
    "h13-qubo-residency": Hypothesis(
        "h13-qubo-residency", "Event-interference QUBO residency",
        "A pairwise Hamiltonian of event-DAG counterfactual interactions finds "
        "a faster resident tensor set than independent measured knapsack.",
        "replay-qubo-v1", "profiled-knapsack-v1", "generation",
        "committed_tokens_per_second", 0.05, "reference_execution_equivalent",
        ("replay_plan_state", "critical_path_profile"),
        "Kill if held-out replay error >10% or paired throughput gain <5%.",
        minimum_repeats=5, minimum_new_tokens=16,
        measured=MeasuredOutcome(
            "no_comparison",
            "Can't tell yet -- the search returned the exact same plan its "
            "control already uses, so there's nothing new to time.",
            "730 candidates evaluated; returned the profiled-knapsack "
            "control exactly (0% predicted gain, 100% plan overlap), even "
            "after the greedy-refill bug behind this was fixed.",
            effect_pct=0.0)),
    "h14-coalesced-storage": Hypothesis(
        "h14-coalesced-storage", "Bounded contiguous storage reads",
        "Coalescing adjacent compressed arrays lowers fixed storage-request "
        "overhead without unacceptable byte amplification.",
        "coalesced-storage-v1", "exact-streaming-v1", "generation",
        "committed_tokens_per_second", 0.05,
        "reference_execution_equivalent", (),
        "Kill if read calls do not fall 50%, byte amplification exceeds 5%, "
        "or throughput gain is below 5%.",
        minimum_repeats=3, minimum_new_tokens=16,
        measured=MeasuredOutcome(
            "mechanism_only",
            "No -- it does exactly what it says (89% fewer storage reads, "
            "zero extra bytes) and still made every run slower.",
            "89.07% fewer read calls, 0% byte amplification, identical "
            "tokens in all 4 pairs -- but 27.73% lower throughput, every "
            "pair unfavorable. One big read blocks the decode it was "
            "supposed to overlap with.",
            effect_pct=-27.73, n_pairs=4)),
    "h15-extent-qubo-residency": Hypothesis(
        "h15-extent-qubo-residency", "Physical-extent QUBO residency",
        "Searching bounded contiguous storage groups exposes a useful plan "
        "that tensor-independent measured knapsack cannot select.",
        "extent-qubo-v1", "profiled-knapsack-v1", "generation",
        "committed_tokens_per_second", 0.05,
        "reference_execution_equivalent",
        ("replay_plan_state", "critical_path_profile"),
        "Kill before GPU timing unless the frozen plan differs from control "
        "and predicts at least 2% replay gain.",
        minimum_repeats=3, minimum_new_tokens=16,
        measured=MeasuredOutcome(
            "no_comparison",
            "Can't tell yet -- same as H13: the search over storage extents "
            "also returned its control's plan exactly.",
            "369 candidates evaluated over 81 real storage extents; "
            "returned the profiled-knapsack control exactly (0% predicted "
            "gain, 100% overlap), blocked at its own pre-GPU mechanism gate.",
            effect_pct=0.0)),
    "h16-spec-critical-path": Hypothesis(
        "h16-spec-critical-path", "Speculation-conditioned critical-path residency",
        "Jointly using the measured critical-path resident set and the proven "
        "fixed speculative chain compounds their gains at the same total VRAM.",
        "spec-critical-path-v1", "tuned-fixed-spec-v1", "generation",
        "committed_tokens_per_second", 0.05, "distribution_exact",
        ("draft_model_id", "critical_path_profile"),
        "Kill if the resident action does not differ, peak VRAM rises more "
        "than 5%, or paired throughput gain is below 5%.",
        minimum_repeats=3, minimum_new_tokens=8),
    "h17-tensor-extents": Hypothesis(
        "h17-tensor-extents", "Tensor-scoped overlap-preserving micro-extents",
        "Coalescing only inside each tensor keeps H14's fixed-request savings "
        "without turning a prefetched layer into one blocking extent.",
        "tensor-extents-v1", "exact-streaming-v1", "generation",
        "committed_tokens_per_second", 0.05, "reference_execution_equivalent",
        (), "Kill if calls fall less than 20%, byte amplification exceeds 5%, "
        "or paired throughput gain is below 5%.",
        minimum_repeats=3, minimum_new_tokens=8),
    "h18-rollback-cached-spec": Hypothesis(
        "h18-rollback-cached-spec", "Rollback-cached target verification",
        "Cropping the exact target KV cache to the accepted prefix avoids "
        "recomputing immutable context on every speculative target sweep.",
        "rollback-cached-spec-v1", "tuned-fixed-spec-v1", "generation",
        "committed_tokens_per_second", 0.05, "distribution_exact",
        ("draft_model_id",),
        "Kill on any greedy-token mismatch at temperature zero, unavailable "
        "cache crop support, over 5% peak-VRAM growth, or under 5% gain.",
        minimum_repeats=3, minimum_new_tokens=8),
}


def registry_payload() -> dict:
    from .protocols import protocol_payload, validate_protocol_registry

    validate_protocol_registry(HYPOTHESES)
    payload = {
        "schema_version": 1,
        "profiles": [dataclasses.asdict(p) for p in PROFILES.values()],
        "hypotheses": [dataclasses.asdict(h) for h in HYPOTHESES.values()],
    }
    payload.update(protocol_payload())
    return payload


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
