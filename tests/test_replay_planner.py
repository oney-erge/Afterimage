import pytest

from afterimage.runtime.critical_path import TraceEvent
from afterimage.runtime.replay_planner import (
    ReplayResidencyPlan, build_storage_extents, manifest_fingerprint,
    optimize_extent_qubo_residency, optimize_qubo_residency,
    optimize_replay_residency,
)


def _manifest():
    return {"tensors": {
        "a": {"orig_bytes": 1_000_000, "comp_bytes": 500_000},
        "b": {"orig_bytes": 1_000_000, "comp_bytes": 400_000},
        # Traffic density wrongly favors c over both critical spans.
        "c": {"orig_bytes": 1_000_000, "comp_bytes": 990_000},
        "embed": {"orig_bytes": 1_000_000, "comp_bytes": 1_000_000,
                  "row_gather": True},
    }}


def _trace():
    return [
        TraceEvent("a-read", "read", "disk-a", 0, 10, tensor_key="a"),
        TraceEvent("b-read", "read", "disk-b", 0, 9, tensor_key="b"),
        TraceEvent("c-read", "read", "disk-c", 0, 1, tensor_key="c"),
    ]


def _extent_manifest():
    manifest = _manifest()
    offsets = {"a": 0, "b": 500_000, "c": 10_000_000}
    for key, offset in offsets.items():
        meta = manifest["tensors"][key]
        meta["blobs"] = {"raw": {
            "offset": offset, "nbytes": meta["comp_bytes"],
            "dtype": "uint8", "shape": [meta["comp_bytes"]],
        }}
    manifest["tensors"]["embed"]["blobs"] = {"raw": {
        "offset": 20_000_000, "nbytes": 1_000_000,
        "dtype": "uint8", "shape": [1_000_000],
    }}
    return manifest


def test_replay_cem_optimizes_complete_sets_and_round_trips(tmp_path):
    plan = optimize_replay_residency(
        _manifest(), [_trace()], vram_budget_gb=0.1373,
        decode_slice_elems=1, iterations=5, population=20, seed=3)
    assert set(plan.vram_keys) == {"a", "b"}
    assert plan.report.optimized_s == pytest.approx(1.0)
    assert plan.report.predicted_speedup >= 9.0
    assert plan.report.observed_coverage == 1.0
    assert plan.report.search_method == "cem"

    path = tmp_path / "plan.json"
    plan.save(path)
    loaded = ReplayResidencyPlan.load(path)
    tiers = loaded.to_tier_plan(
        _manifest(), vram_budget_gb=0.1373, decode_slice_elems=1)
    assert set(tiers.vram_keys) == {"a", "b"}
    assert tiers.row_gather_keys == ["embed"]


def test_replay_plan_rejects_manifest_or_budget_drift(tmp_path):
    plan = optimize_replay_residency(
        _manifest(), [_trace()], vram_budget_gb=0.1373,
        decode_slice_elems=1, iterations=2, population=8, seed=1)
    changed = _manifest()
    changed["tensors"]["a"]["orig_bytes"] += 1
    with pytest.raises(ValueError, match="different compressed manifest"):
        plan.to_tier_plan(changed, vram_budget_gb=0.1373, decode_slice_elems=1)
    with pytest.raises(ValueError, match="runtime requested"):
        plan.to_tier_plan(_manifest(), vram_budget_gb=0.14,
                          decode_slice_elems=1)


def test_replay_search_requires_coverage():
    trace = [_trace()[0]]
    with pytest.raises(ValueError, match="cover"):
        optimize_replay_residency(
            _manifest(), [trace], vram_budget_gb=0.1373,
            decode_slice_elems=1, iterations=1, population=2)


def test_manifest_fingerprint_ignores_nonplacement_metadata():
    left = _manifest()
    right = _manifest()
    right["notes"] = "does not affect execution"
    assert manifest_fingerprint(left) == manifest_fingerprint(right)


def test_qubo_annealer_models_pairwise_critical_path_interference(tmp_path):
    plan = optimize_qubo_residency(
        _manifest(), [_trace()], vram_budget_gb=0.1373,
        decode_slice_elems=1, pairwise_candidates=3,
        restarts=4, sweeps=300, seed=7)
    assert set(plan.vram_keys) == {"a", "b"}
    assert plan.report.optimized_s == pytest.approx(1.0)
    assert plan.report.search_method == "qubo_anneal"
    assert plan.report.optimized_s <= plan.report.control_s

    path = tmp_path / "qubo-plan.json"
    plan.save(path)
    loaded = ReplayResidencyPlan.load(path)
    tiers = loaded.to_tier_plan(
        _manifest(), vram_budget_gb=0.1373, decode_slice_elems=1)
    assert set(tiers.vram_keys) == {"a", "b"}


def test_qubo_search_validates_hyperparameters():
    with pytest.raises(ValueError, match="pairwise_candidates"):
        optimize_qubo_residency(
            _manifest(), [_trace()], vram_budget_gb=0.1373,
            decode_slice_elems=1, pairwise_candidates=1)


def test_storage_extents_follow_physical_offsets_not_key_order():
    extents = build_storage_extents(
        _extent_manifest(), ["c", "b", "a"],
        max_extent_bytes=2_000_000, max_gap_bytes=0)
    assert extents[0].keys == ("a", "b")
    assert extents[0].span_bytes == 900_000
    assert extents[1].keys == ("c",)


def test_extent_qubo_is_control_safe_and_reports_action_gate():
    plan = optimize_extent_qubo_residency(
        _extent_manifest(), [_trace()], vram_budget_gb=0.1373,
        decode_slice_elems=1, max_extent_bytes=2_000_000,
        pairwise_candidates=2, restarts=3, sweeps=100, seed=5)
    assert plan.report.search_method == "extent_qubo_anneal"
    assert plan.report.candidate_group_count == 2
    assert plan.report.optimized_s <= plan.report.control_s
    assert plan.report.treatment_diverged == (
        plan.report.control_overlap < 1.0)
