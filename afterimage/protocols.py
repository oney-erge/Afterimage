"""Hypothesis-aware evidence protocols and paired screening decisions.

A two-token generation is useful for catching crashes and contract violations,
but it is not a universal scientific test.  Placement, online prefetch control,
learned speculation and offline artifact hypotheses have different units of
evidence.  This module makes those differences machine-readable and keeps an
exploratory screen from being mislabeled as confirmation.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass(frozen=True)
class EvidenceStage:
    id: str
    level: str
    purpose: str
    minimum_cases: int = 0
    tokens_per_case: int = 0
    paired_repeats: int = 0
    calibration_observations: int = 0
    budgets: int = 1
    max_minutes: int = 60
    confirmatory: bool = False


@dataclasses.dataclass(frozen=True)
class TestProtocol:
    id: str
    family: str
    estimand: str
    required_diagnostics: tuple[str, ...]
    stages: tuple[EvidenceStage, ...]
    advance_rule: str
    confirmation_rule: str


COMMON_INVARIANT = EvidenceStage(
    "l0-invariant", "L0", "Validate exactness, budget and artifact contracts",
    max_minutes=5)


PROTOCOLS = {
    "offline-controller": TestProtocol(
        "offline-controller", "request-level policy / RL",
        "held-out oracle fraction and baseline regret",
        ("disjoint_split", "oracle_gap", "regret", "simulator_error"),
        (
            COMMON_INVARIANT,
            EvidenceStage("l1-replay-smoke", "L1", "Schema and replay smoke test",
                          minimum_cases=30, max_minutes=5),
            EvidenceStage("l2-held-out-screen", "L2", "Blocked held-out replay",
                          minimum_cases=100, paired_repeats=3, max_minutes=20),
            EvidenceStage("l3-confirm", "L3", "Fixed chronological test set",
                          minimum_cases=300, paired_repeats=5, max_minutes=60,
                          confirmatory=True),
        ),
        "Advance only when the held-out oracle gap exceeds the hypothesis gate.",
        "Lower confidence bound clears the gate, simulator MAPE <=10%, and no "
        "baseline-safety violation."),
    "placement-latency": TestProtocol(
        "placement-latency", "offline tensor placement",
        "paired median log throughput ratio at equal measured peak VRAM",
        ("token_ids", "peak_vram", "bytes_read", "replay_mape",
         "plan_overlap", "search_time"),
        (
            COMMON_INVARIANT,
            EvidenceStage("l1-mechanism", "L1", "One-plan execution smoke test",
                          minimum_cases=1, tokens_per_case=2, paired_repeats=1,
                          max_minutes=15),
            EvidenceStage("l2-screen", "L2", "Diverse paired cold-cache screen",
                          minimum_cases=4, tokens_per_case=2, paired_repeats=2,
                          budgets=1, max_minutes=35),
            EvidenceStage("l3-confirm", "L3", "Frozen plan on disjoint traces",
                          minimum_cases=4, tokens_per_case=4, paired_repeats=3,
                          budgets=3, max_minutes=60, confirmatory=True),
        ),
        "Stop for futility when the regulated-screen upper interval is below "
        "the practical-effect gate; otherwise advance without claiming a win.",
        "At every budget, tokens match, peak VRAM is within 5%, replay MAPE is "
        "<=10%, and the paired 95% lower bound is positive with point effect "
        "at or above the hypothesis gate."),
    "adaptive-prefetch": TestProtocol(
        "adaptive-prefetch", "online I/O scheduling",
        "steady-state paired log throughput ratio after disjoint burn-in",
        ("posterior_count", "chosen_depths", "predicted_ready_probability",
         "prefetch_hits", "prefetch_misses", "prefetch_wait_seconds",
         "inflight_bytes"),
        (
            COMMON_INVARIANT,
            EvidenceStage("l1-controller-smoke", "L1", "Controller burn-in and bounds",
                          minimum_cases=1, tokens_per_case=4, paired_repeats=1,
                          calibration_observations=80, max_minutes=12),
            EvidenceStage("l2-screen", "L2", "Randomized storage/prompt blocks",
                          minimum_cases=4, tokens_per_case=4, paired_repeats=2,
                          calibration_observations=160, max_minutes=40),
            EvidenceStage("l3-confirm", "L3", "Steady-state fixed-stage test",
                          minimum_cases=3, tokens_per_case=6, paired_repeats=3,
                          calibration_observations=240, max_minutes=60,
                          confirmatory=True),
        ),
        "Require posterior burn-in, bounded depth and lower exposed wait before "
        "advancing; a transient first-token gain is insufficient.",
        "Point throughput effect clears the gate, paired 95% lower bound is "
        "positive, exposed wait falls >=10%, and posterior ready probabilities "
        "are calibrated on held-out layer demands."),
    "learned-speculation": TestProtocol(
        "learned-speculation", "learned speculative stopping",
        "held-out committed tokens/second versus calibrated best fixed k",
        ("calibration_positions", "brier_score", "chosen_chain_lengths",
         "decision_stops", "decision_continues", "accepted_tokens_per_sweep",
         "target_sweeps", "distribution_test"),
        (
            COMMON_INVARIANT,
            EvidenceStage("l1-action-smoke", "L1", "Prove policy action divergence",
                          minimum_cases=1, tokens_per_case=16, paired_repeats=1,
                          calibration_observations=200, max_minutes=20),
            EvidenceStage("l2-screen", "L2", "Frozen state on held-out regimes",
                          minimum_cases=3, tokens_per_case=32, paired_repeats=2,
                          calibration_observations=400, max_minutes=50),
            EvidenceStage("l3-confirm", "L3", "Fixed-stage latency and distribution test",
                          minimum_cases=2, tokens_per_case=48, paired_repeats=3,
                          calibration_observations=600, max_minutes=60,
                          confirmatory=True),
        ),
        "Do not compare speed until calibration is held out and candidate actions "
        "differ from fixed k in at least 10% of opportunities.",
        "Paired 95% lower bound is positive, point effect clears the gate, no "
        "major prompt family regresses, and the target-distribution test passes."),
    "certified-search": TestProtocol(
        "certified-search", "exact branch-and-bound search",
        "certified rows avoided and end-to-end greedy throughput",
        ("adversarial_argmax", "certificate_rate", "rows_pruned",
         "fallback_rate", "index_bytes", "index_build_seconds"),
        (
            COMMON_INVARIANT,
            EvidenceStage("l1-certificate-smoke", "L1", "Adversarial bound audit",
                          minimum_cases=100, max_minutes=10),
            EvidenceStage("l2-screen", "L2", "Real-head pruning screen",
                          minimum_cases=4, tokens_per_case=8, paired_repeats=2,
                          max_minutes=40),
            EvidenceStage("l3-confirm", "L3", "Fixed greedy head-to-head",
                          minimum_cases=4, tokens_per_case=16, paired_repeats=3,
                          max_minutes=60, confirmatory=True),
        ),
        "Kill before latency testing if fewer than 70% of rows are certified away.",
        "No incorrect certificate, point effect clears the gate, and index RAM "
        "plus build amortization are reported."),
    "artifact-design": TestProtocol(
        "artifact-design", "exact representation / codec",
        "storage and preparation reduction over tensor families",
        ("bitwise_round_trip", "tensor_family_coverage", "artifact_bytes",
         "prepare_seconds", "dependency_failures"),
        (
            COMMON_INVARIANT,
            EvidenceStage("l1-artifact-smoke", "L1", "Round-trip representative tensors",
                          minimum_cases=20, max_minutes=10),
            EvidenceStage("l2-family-screen", "L2", "Cross-layer/family audit",
                          minimum_cases=100, max_minutes=30),
            EvidenceStage("l3-confirm", "L3", "Multiple checkpoints and live preparation",
                          minimum_cases=300, paired_repeats=3, max_minutes=60,
                          confirmatory=True),
        ),
        "Advance to GPU work only if the offline physical-design oracle clears "
        "the storage/preparation gate.",
        "Every artifact round-trips bitwise and a held-out checkpoint clears the "
        "practical-effect gate."),
    "storage-extent": TestProtocol(
        "storage-extent", "physical storage request geometry",
        "paired exact throughput at fixed residency and cold-cache policy",
        ("token_ids", "storage_read_calls", "storage_extent_bytes",
         "logical_bytes_read", "read_call_reduction", "byte_amplification",
         "peak_host_buffer_bytes"),
        (
            COMMON_INVARIANT,
            EvidenceStage("l1-request-smoke", "L1", "Prove requests coalesce",
                          minimum_cases=1, tokens_per_case=2,
                          paired_repeats=1, max_minutes=12),
            EvidenceStage("l2-screen", "L2", "Randomized paired extent screen",
                          minimum_cases=4, tokens_per_case=4,
                          paired_repeats=2, max_minutes=40),
            EvidenceStage("l3-confirm", "L3", "Frozen extent geometry test",
                          minimum_cases=4, tokens_per_case=8,
                          paired_repeats=3, max_minutes=60,
                          confirmatory=True),
        ),
        "Do not time the treatment unless read calls fall by at least 50% and "
        "actual bytes read stay within 5% of the control.",
        "Tokens match, point effect clears the gate, paired 95% lower bound is "
        "positive, read calls fall >=50%, and byte amplification is <=5%."),
    "ram-overlay": TestProtocol(
        "ram-overlay", "liveness-guided host-memory overlay",
        "paired throughput at matched peak VRAM with pinned host memory",
        ("memlock_limit", "pinned_keys", "pageable_fallback_keys", "h2d_bytes",
         "token_ids", "peak_vram"),
        (
            COMMON_INVARIANT,
            EvidenceStage("l1-environment-gate", "L1", "Verify >=1.6 GB pin capability",
                          minimum_cases=1, max_minutes=5),
            EvidenceStage("l2-screen", "L2", "Pinned overlay paired screen",
                          minimum_cases=4, tokens_per_case=2, paired_repeats=2,
                          max_minutes=35),
            EvidenceStage("l3-confirm", "L3", "Pinned fixed-stage comparison",
                          minimum_cases=4, tokens_per_case=4, paired_repeats=3,
                          max_minutes=60, confirmatory=True),
        ),
        "Do not interpret a pageable fallback as a test of the pinned hypothesis.",
        "All allocations are pinned, tokens match, peak VRAM is within 5%, and "
        "the paired 95% lower bound is positive with point effect above gate."),
}


HYPOTHESIS_PROTOCOLS = {
    "h0-joint-oracle-gap": "offline-controller",
    "h1-critical-path": "placement-latency",
    "h2-hazard-cost": "learned-speculation",
    "h3-contextual-bandit": "offline-controller",
    "h4-feedback-prefetch": "adaptive-prefetch",
    "h5-certified-mips": "certified-search",
    "h6-representations": "artifact-design",
    "h7-xor-reference": "artifact-design",
    "h8-model-based-rl": "offline-controller",
    "h9-ram-overlay-head": "ram-overlay",
    "h10-replay-cem": "placement-latency",
    "h11-neural-utility-spec": "learned-speculation",
    "h12-bayesian-prefetch": "adaptive-prefetch",
    "h13-qubo-residency": "placement-latency",
    "h14-coalesced-storage": "storage-extent",
    "h15-extent-qubo-residency": "placement-latency",
}


def protocol_for(hypothesis_id: str) -> TestProtocol:
    try:
        return PROTOCOLS[HYPOTHESIS_PROTOCOLS[hypothesis_id]]
    except KeyError as exc:
        raise KeyError("no regulated test protocol for %s" % hypothesis_id) from exc


def protocol_payload() -> dict:
    return {
        "schema_version": 1,
        "evidence_levels": {
            "L0": "contract/invariant only",
            "L1": "mechanism smoke; cannot support or falsify performance",
            "L2": "regulated exploratory screen; can stop for futility",
            "L3": "fixed-stage confirmation eligible for a performance claim",
        },
        "protocols": [dataclasses.asdict(protocol) for protocol in PROTOCOLS.values()],
        "hypothesis_protocols": dict(HYPOTHESIS_PROTOCOLS),
    }


def validate_protocol_registry(hypothesis_ids) -> None:
    expected = set(hypothesis_ids)
    mapped = set(HYPOTHESIS_PROTOCOLS)
    if expected != mapped:
        missing = sorted(expected - mapped)
        extra = sorted(mapped - expected)
        raise ValueError("protocol mapping mismatch; missing=%s extra=%s" %
                         (missing, extra))
    unknown = sorted(set(HYPOTHESIS_PROTOCOLS.values()) - set(PROTOCOLS))
    if unknown:
        raise ValueError("unknown protocol ids: %s" % unknown)


def assess_paired_effect(control_seconds, candidate_seconds, *,
                         minimum_effect: float, level: str,
                         bootstrap_samples: int = 5000,
                         seed: int = 0) -> dict:
    """Robust paired effect summary with evidence-level-aware language.

    The effect is ``control/candidate - 1`` on a paired log scale.  A median
    resists one cold-cache outlier.  Bootstrap intervals are descriptive for
    L1/L2; only a predeclared fixed L3 sample is confirmation-eligible.
    """
    control = np.asarray(control_seconds, dtype=np.float64)
    candidate = np.asarray(candidate_seconds, dtype=np.float64)
    if control.shape != candidate.shape or control.ndim != 1 or not len(control):
        raise ValueError("control and candidate need equal non-empty 1D samples")
    if np.any(control <= 0) or np.any(candidate <= 0):
        raise ValueError("paired seconds must be positive")
    if level not in ("L1", "L2", "L3"):
        raise ValueError("level must be L1, L2 or L3")
    log_ratio = np.log(control) - np.log(candidate)
    center_log = float(np.median(log_ratio))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(log_ratio), size=(bootstrap_samples, len(log_ratio)))
    boot = np.median(log_ratio[indices], axis=1)
    alpha = 0.05 if level == "L3" else 0.10
    lower_log, upper_log = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    effect = math.exp(center_log) - 1.0
    lower = math.exp(float(lower_log)) - 1.0
    upper = math.exp(float(upper_log)) - 1.0
    sign_consistency = float(np.mean(log_ratio > 0))
    if level == "L1":
        decision = "mechanism_only"
    elif level == "L2":
        if upper < minimum_effect:
            decision = "stop_futility"
        elif effect >= minimum_effect and sign_consistency >= 0.67:
            decision = "advance_to_confirmation"
        else:
            decision = "extend_or_redesign"
    else:
        decision = ("supported" if effect >= minimum_effect and lower > 0
                    else "not_supported")
    return {
        "pairs": len(log_ratio), "level": level,
        "median_speedup_effect": effect,
        "interval": [lower, upper],
        "interval_level": 1.0 - alpha,
        "sign_consistency": sign_consistency,
        "minimum_effect": minimum_effect,
        "decision": decision,
        "confirmation_eligible": level == "L3",
    }
