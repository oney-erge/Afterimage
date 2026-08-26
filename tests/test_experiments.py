import pytest

from afterimage.experiments import (
    HYPOTHESES, PROFILES, ResultStore, _paired_log_ratio_effect, oracle_gap, run_paired,
)


def test_all_hypotheses_and_profiles_are_registered():
    assert set(HYPOTHESES) == {f"h{i}-" + suffix for i, suffix in [
        (0, "joint-oracle-gap"), (1, "critical-path"), (2, "hazard-cost"),
        (3, "contextual-bandit"), (4, "feedback-prefetch"),
        (5, "certified-mips"), (6, "representations"),
        (7, "xor-reference"), (8, "model-based-rl"),
        (9, "ram-overlay-head"), (10, "replay-cem"),
        (11, "neural-utility-spec"),
        (12, "bayesian-prefetch"), (13, "qubo-residency"),
        (14, "coalesced-storage"), (15, "extent-qubo-residency"),
        (16, "spec-critical-path"), (17, "tensor-extents"),
        (18, "rollback-cached-spec"),
    ]}
    assert all(h.candidate_profile in PROFILES and h.control_profile in PROFILES
               for h in HYPOTHESES.values())


def test_paired_runner_uses_named_control_and_marks_clear_win():
    updates = []

    def execute(profile, repeat):
        value = 2.0 if profile.id == "pi-prefetch-v1" else 1.0
        return {"committed_tokens_per_second": value,
                "output_token_ids": [repeat, 1], "exact": True}

    run = run_paired("h4-feedback-prefetch", execute, repeats=4,
                     progress=updates.append)
    assert run.verdict == "favored"
    assert run.summary["effect"] == 1.0
    assert updates[-1]["completed"] == 8


def test_paired_runner_invalidates_token_mismatch():
    def execute(profile, repeat):
        token = 1 if profile.id == "pi-prefetch-v1" else 2
        return {"committed_tokens_per_second": 2.0,
                "output_token_ids": [token], "exact": True}

    assert run_paired("h4-feedback-prefetch", execute, repeats=2).verdict == "invalid"


def test_distribution_exact_runner_does_not_require_identical_random_draws():
    def execute(profile, repeat):
        token = 1 if profile.id == "hazard-cost-v1" else 2
        return {"committed_tokens_per_second": 2.0,
                "output_token_ids": [token], "exact": True}

    assert run_paired("h2-hazard-cost", execute, repeats=2).verdict != "invalid"


def test_method_profile_selector_cannot_be_overridden_by_common_config():
    cfg = PROFILES["pi-prefetch-v1"].resolve({"prefetch_policy": "fixed"})
    assert cfg.prefetch_policy == "pi"


def test_h16_profile_composes_critical_path_placement_with_fixed_speculation():
    """H16's hypothesis is that critical-path residency (H1's mechanism,
    already shown elsewhere to select a different resident set than the
    default traffic_density policy) and fixed k=8 speculation compound when
    used together. Nothing previously asserted that the h16 profile actually
    carries *both* ingredients rather than silently dropping one -- a
    plausible-looking edit to either MethodProfile could otherwise turn H16
    back into a rename of H1 or of plain fixed speculation without any test
    noticing."""
    # critical_path is a real measured-cost policy, not a free-standing
    # switch: EngineConfig requires a critical_path_profile artifact before
    # it will resolve, so both profiles below carry a placeholder path.
    extra = {"critical_path_profile": "unused-profile.json", "vram_budget_gb": 4.0}
    cfg = PROFILES["spec-critical-path-v1"].resolve(extra)
    assert cfg.placement_policy == "critical_path"
    assert cfg.spec_k_policy == "fixed"
    assert cfg.spec_k == 8
    assert cfg.draft_mode == "model"
    control_cfg = PROFILES[HYPOTHESES["h16-spec-critical-path"].control_profile].resolve(extra)
    assert control_cfg.placement_policy != cfg.placement_policy


def test_h9_profile_requires_pinned_ram_matching_the_validated_hardware_run():
    """The published H9 result came from scripts/run_bounded_suite.py's
    METHODS["ram-overlay-head"], which sets require_pinned_ram=True so a
    pin_memory failure raises instead of silently degrading to pageable RAM
    (see streaming_engine.py's documented fallback and its RuntimeWarning).
    The registry's ram-overlay-head-v1 profile -- what the live web UI
    Experiment Lab actually runs via run_paired -- must carry the same
    fail-closed guarantee, or a live H9 run through the UI could violate
    H9's own kill criterion ("Kill on ... a pinned-memory failure") without
    it being visible as anything worse than a warning."""
    cfg = PROFILES["ram-overlay-head-v1"].resolve(
        {"vram_budget_gb": 4.0, "ram_budget_gb": 4.0})
    assert cfg.lm_head_policy == "ram_overlay"
    assert cfg.require_pinned_ram is True


def test_h3_profile_selects_linucb_execution_policy_and_engine_rejects_it_directly():
    """H3's actions are complete profiles selected *between* requests by
    LinearProfileBandit (see test_controllers.py), not a per-request engine
    setting. Two things must both hold for that design to actually be safe:
    the profile really does set execution_policy="linucb" (so the bandit has
    something to select), and StreamingLosslessModel really does refuse to
    run with it directly, fail-closed, rather than silently ignoring it."""
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg = PROFILES["contextual-linucb-v1"].resolve()
    assert cfg.execution_policy == "linucb"
    with pytest.raises(RuntimeError, match="request-boundary controller"):
        StreamingLosslessModel("unused", "unused", config=cfg)


def test_result_store_is_immutable(tmp_path):
    run = run_paired("h4-feedback-prefetch", lambda profile, repeat: {
        "committed_tokens_per_second": 1.0, "output_token_ids": [1]}, repeats=1)
    store = ResultStore(tmp_path)
    store.write_once(run)
    try:
        store.write_once(run)
        assert False, "second write should fail"
    except FileExistsError:
        pass
    assert store.get("../outside-file") is None


def test_paired_effect_point_estimate_falls_within_its_own_interval():
    """Regression test: run_paired previously reported effect as a ratio of
    medians (candidate_median / control_median - 1) while its bootstrap CI
    was built from a different statistic (the geometric mean of per-pair
    ratios). On skewed paired data those two disagree, so the reported point
    estimate could sit outside its own interval. _paired_log_ratio_effect
    computes the CI from the same statistic as the point effect."""
    candidate = [2.0, 2.0, 8.0, 8.0]
    control = [1.0, 4.0, 4.0, 4.0]
    # ratio-of-medians would give 5/4 - 1 = 0.25; the paired per-index
    # ratios are [2.0, 0.5, 2.0, 2.0], whose median is 2.0 (effect 1.0).
    effect, lo, hi = _paired_log_ratio_effect(candidate, control, seed=0)
    assert effect == pytest.approx(1.0, abs=1e-9)
    assert lo <= effect <= hi


def test_paired_runner_effect_uses_the_same_statistic_as_its_interval():
    values_candidate = {0: 2.0, 1: 2.0, 2: 8.0, 3: 8.0}
    values_control = {0: 1.0, 1: 4.0, 2: 4.0, 3: 4.0}

    def execute(profile, repeat):
        table = values_candidate if profile.id == "pi-prefetch-v1" else values_control
        return {"committed_tokens_per_second": table[repeat],
                "output_token_ids": [1], "exact": True}

    run = run_paired("h4-feedback-prefetch", execute, repeats=4)
    assert run.summary["effect"] == pytest.approx(1.0, abs=1e-9)
    assert run.summary["ci95"][0] <= run.summary["effect"] <= run.summary["ci95"][1]


def test_joint_oracle_gap_uses_both_context_dimensions():
    rows = []
    for semantic in ("easy", "hard"):
        for system in ("fast", "slow"):
            winner = "a" if semantic == system.replace("fast", "easy").replace("slow", "hard") else "b"
            for profile in ("a", "b"):
                rows.append({"profile": profile, "semantic_bucket": semantic,
                             "system_bucket": system,
                             "committed_tokens_per_second": 2.0 if profile == winner else 1.0})
    result = oracle_gap(rows)
    assert result["joint_oracle"] >= result["global"]
