from afterimage.bench.harness import aggregate, run_matrix, unstable_configs


def test_aggregate_computes_median_and_flags_stable_run():
    trials = [{"gb_per_token": v} for v in [1.0, 1.02, 0.98, 1.01, 0.99]]
    result = aggregate(trials)
    assert abs(result["gb_per_token"].median - 1.0) < 0.05
    assert result["gb_per_token"].stable is True


def test_aggregate_flags_unstable_run():
    trials = [{"gb_per_token": v} for v in [1.0, 1.0, 1.0, 1.0, 5.0]]
    result = aggregate(trials)
    assert result["gb_per_token"].stable is False


def test_run_matrix_collects_n_repeats_per_config():
    calls = {"a": 0, "b": 0}

    def make_fn(name):
        def fn():
            calls[name] += 1
            return {"metric": float(calls[name])}
        return fn

    configs = {"a": make_fn("a"), "b": make_fn("b")}
    results = run_matrix(configs, n_repeats=5, seed=1)

    assert calls["a"] == 5
    assert calls["b"] == 5
    assert set(results.keys()) == {"a", "b"}
    assert len(results["a"]["metric"].values) == 5


def test_unstable_configs_reports_failing_pairs():
    results = {
        "cfgA": {"gb": aggregate([{"gb": v} for v in [1, 1, 1, 1]])["gb"]},
        "cfgB": {"gb": aggregate([{"gb": v} for v in [1, 1, 1, 100]])["gb"]},
    }
    bad = unstable_configs(results)
    assert ("cfgB", "gb") in bad
    assert ("cfgA", "gb") not in bad
