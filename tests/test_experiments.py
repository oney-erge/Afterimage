from afterimage.experiments import HYPOTHESES, PROFILES, ResultStore, oracle_gap, run_paired


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
