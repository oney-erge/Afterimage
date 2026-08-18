import pytest

from afterimage.runtime.vram_planner import TensorInfo, plan_from_manifest, plan_residency


def _t(key, orig_mb, comp_mb, layer=True):
    return TensorInfo(key, int(orig_mb * 1e6), int(comp_mb * 1e6), layer)


def test_everything_resident_when_budget_is_ample():
    ts = [_t("a", 100, 70), _t("b", 100, 70)]
    plan = plan_residency(ts, budget_gb=10.0)
    assert plan.feasible
    assert set(plan.resident_keys) == {"a", "b"}
    assert plan.streamed_keys == []
    assert plan.streamed_bytes_per_token == 0


def test_infeasible_when_budget_below_largest_tensor_headroom():
    """A budget that cannot even materialize the biggest tensor plus its
    decode scratch must be rejected UP FRONT, not discovered as an OOM on
    the first layer -- which is how the earlier fixed-residency design
    failed at both 4 GB and 6 GB."""
    ts = [_t("big", 2000, 1400)]
    plan = plan_residency(ts, budget_gb=1.0)  # 2 GB tensor + scratch >> 1 GB
    assert not plan.feasible
    assert "largest tensor" in plan.reason
    assert plan.resident_keys == []


def test_least_compressible_tensors_are_kept_resident():
    """The counterintuitive core of the planner: a tensor that compresses
    POORLY costs the most bus traffic per byte of VRAM, so it is the best
    one to pin. Highly compressible tensors are cheap to re-stream."""
    incompressible = _t("dense", 100, 98)   # density 0.98
    compressible = _t("sparse", 100, 40)    # density 0.40
    # headroom = 100MB largest + scratch; budget leaves room for one tensor
    plan = plan_residency([compressible, incompressible], budget_gb=0.7,
                          scratch_bytes=int(500e6))
    assert plan.feasible
    assert plan.resident_keys == ["dense"]
    assert plan.streamed_keys == ["sparse"]


def test_streamed_bytes_reflect_only_evicted_tensors():
    ts = [_t("a", 100, 90), _t("b", 100, 50), _t("c", 100, 30)]
    plan = plan_residency(ts, budget_gb=0.7, scratch_bytes=int(500e6))
    assert plan.resident_keys == ["a"]
    assert plan.streamed_bytes_per_token == int(50e6) + int(30e6)


def test_larger_budget_never_streams_more():
    """Monotonicity: more VRAM must never increase per-token bus traffic."""
    ts = [_t("a", 100, 90), _t("b", 100, 60), _t("c", 100, 30)]
    prev = None
    for gb in [0.7, 0.8, 0.9, 1.5]:
        plan = plan_residency(ts, budget_gb=gb, scratch_bytes=int(500e6))
        if prev is not None:
            assert plan.streamed_bytes_per_token <= prev
        prev = plan.streamed_bytes_per_token


def test_plan_from_manifest_matches_real_shape():
    manifest = {"tensors": {
        "model.embed_tokens.weight": {"orig_bytes": int(1.56e9), "comp_bytes": int(1.1e9)},
        "model.layers.0.mlp.down_proj.weight": {"orig_bytes": int(27e6), "comp_bytes": int(18e6)},
        "lm_head.weight": {"orig_bytes": int(1.56e9), "comp_bytes": int(1.1e9)},
    }}
    # 1.56 GB embedding + 0.5 GB scratch => ~2.1 GB floor; 3 GB clears it
    plan = plan_from_manifest(manifest, budget_gb=3.0)
    assert plan.feasible
    assert plan.resident_bytes <= plan.budget_bytes - plan.headroom_bytes
    assert len(plan.resident_keys) + len(plan.streamed_keys) == 3


def test_describe_is_readable_and_flags_infeasible():
    plan = plan_residency([_t("big", 4000, 3000)], budget_gb=1.0)
    text = plan.describe()
    assert "VRAM plan" in text
    assert "feasible            : False" in text
