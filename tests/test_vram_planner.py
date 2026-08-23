import types

import pytest

from afterimage.runtime.critical_path import CriticalPathProfile, TensorCost
from afterimage.runtime.vram_planner import (
    TensorInfo, kv_cache_bytes_per_token, plan_from_manifest, plan_tiers,
)


def _t(key, orig_mb, comp_mb, layer=True, row_gather=False, uses=1):
    return TensorInfo(key, int(orig_mb * 1e6), int(comp_mb * 1e6), layer, row_gather, uses)


def test_everything_fits_in_vram_when_budget_is_ample():
    ts = [_t("a", 100, 70), _t("b", 100, 70)]
    plan = plan_tiers(ts, vram_budget_gb=10.0)
    assert plan.feasible
    assert set(plan.vram_keys) == {"a", "b"}
    assert plan.ram_keys == []
    assert plan.disk_keys == []
    assert plan.disk_bytes_per_token == 0


def test_infeasible_when_vram_budget_below_largest_tensor_headroom():
    """A budget that cannot even materialize the biggest tensor plus its
    decode scratch must be rejected UP FRONT, not discovered as an OOM on
    the first layer -- which is how the earlier fixed-residency design
    failed at both 4 GB and 6 GB."""
    ts = [_t("big", 2000, 1400)]
    plan = plan_tiers(ts, vram_budget_gb=1.0)  # 2 GB tensor + scratch >> 1 GB
    assert not plan.feasible
    assert "largest eligible tensor" in plan.reason
    assert plan.vram_keys == []


def test_least_compressible_tensors_fill_vram_first():
    """The counterintuitive core of the planner: a tensor that compresses
    POORLY costs the most bus traffic per byte of residency, so it is the
    best one to pin. Highly compressible tensors are cheap to re-stream."""
    incompressible = _t("dense", 100, 98)   # density 0.98
    compressible = _t("sparse", 100, 40)    # density 0.40
    plan = plan_tiers([compressible, incompressible], vram_budget_gb=0.7,
                      scratch_bytes=int(500e6))
    assert plan.feasible
    assert plan.vram_keys == ["dense"]
    assert plan.disk_keys == ["sparse"]


def test_ram_tier_catches_what_vram_could_not_hold():
    """Three tensors, VRAM only large enough for one: the second-highest
    density tensor should land in RAM (fast memcpy) rather than falling
    straight to disk (read + decode every token)."""
    a = _t("a", 100, 90)   # density .90 -> vram
    b = _t("b", 100, 60)   # density .60 -> ram
    c = _t("c", 100, 30)   # density .30 -> disk
    plan = plan_tiers([a, b, c], vram_budget_gb=0.7, ram_budget_gb=0.15,
                      scratch_bytes=int(500e6))
    assert plan.vram_keys == ["a"]
    assert plan.ram_keys == ["b"]
    assert plan.disk_keys == ["c"]
    assert plan.disk_bytes_per_token == int(30e6)


def test_forced_ram_overlay_reserves_host_budget_before_greedy_fill():
    head = _t("lm_head.weight", 100, 40, layer=False)
    layer = _t("layer", 100, 90)
    plan = plan_tiers(
        [head, layer], vram_budget_gb=0.7, ram_budget_gb=0.1,
        scratch_bytes=int(500e6), forced_ram_keys={"lm_head.weight"})
    assert plan.feasible
    assert plan.ram_keys == ["lm_head.weight"]
    assert plan.vram_keys == ["layer"]


def test_forced_ram_overlay_fails_before_runtime_when_host_budget_is_too_small():
    head = _t("lm_head.weight", 100, 40, layer=False)
    plan = plan_tiers(
        [head], vram_budget_gb=0.7, ram_budget_gb=0.05,
        scratch_bytes=int(500e6), forced_ram_keys={"lm_head.weight"})
    assert not plan.feasible
    assert "forced RAM tensors need" in plan.reason


def test_ram_budget_zero_is_legacy_two_tier_behaviour():
    a = _t("a", 100, 90)
    b = _t("b", 100, 60)
    plan = plan_tiers([a, b], vram_budget_gb=0.7, scratch_bytes=int(500e6))
    assert plan.ram_keys == []
    assert plan.disk_keys == ["b"]


def test_row_gather_tensors_are_excluded_from_every_tier():
    """A row-gathered embedding is never fully materialized (it reads a
    handful of rows per token, not the whole table), so charging it a
    VRAM/RAM/disk cost at all would either overstate it (as a stale earlier
    version did, treating it like any other tensor) or require guessing a
    per-call size this planner cannot know. It must not appear in any of
    the three tier lists."""
    normal = _t("normal", 100, 90)
    embed = _t("embed", 1560, 1560, layer=False, row_gather=True)
    plan = plan_tiers([normal, embed], vram_budget_gb=10.0, ram_budget_gb=10.0)
    assert embed.key not in plan.vram_keys
    assert embed.key not in plan.ram_keys
    assert embed.key not in plan.disk_keys
    assert plan.row_gather_keys == ["embed"]
    assert plan.tier_of("embed") == "row_gather"


def test_larger_vram_budget_never_increases_disk_traffic():
    """Monotonicity: more VRAM must never increase per-token bus traffic."""
    ts = [_t("a", 100, 90), _t("b", 100, 60), _t("c", 100, 30)]
    prev = None
    for gb in [0.7, 0.8, 0.9, 1.5]:
        plan = plan_tiers(ts, vram_budget_gb=gb, scratch_bytes=int(500e6))
        if prev is not None:
            assert plan.disk_bytes_per_token <= prev
        prev = plan.disk_bytes_per_token


def test_larger_ram_budget_never_increases_disk_traffic():
    ts = [_t("a", 100, 90), _t("b", 100, 60), _t("c", 100, 30)]
    prev = None
    for ram_gb in [0.0, 0.1, 0.2, 1.0]:
        plan = plan_tiers(ts, vram_budget_gb=0.7, ram_budget_gb=ram_gb,
                          scratch_bytes=int(500e6))
        if prev is not None:
            assert plan.disk_bytes_per_token <= prev
        prev = plan.disk_bytes_per_token


def test_tier_of_reports_disk_for_anything_unassigned():
    ts = [_t("a", 100, 90)]
    plan = plan_tiers(ts, vram_budget_gb=0.05, scratch_bytes=int(1e6))
    assert plan.tier_of("a") == "disk"
    assert plan.tier_of("nonexistent") == "disk"


def test_plan_from_manifest_matches_real_shape_and_excludes_row_gather():
    manifest = {"tensors": {
        "model.embed_tokens.weight": {
            "orig_bytes": int(1.56e9), "comp_bytes": int(1.56e9), "row_gather": True},
        "model.layers.0.mlp.down_proj.weight": {
            "orig_bytes": int(27e6), "comp_bytes": int(18e6)},
        "lm_head.weight": {"orig_bytes": int(1.56e9), "comp_bytes": int(1.1e9)},
    }}
    # 1.56 GB lm_head (largest eligible, embed is excluded) + 0.5 GB scratch
    # => ~2.1 GB floor; 3 GB clears it.
    plan = plan_from_manifest(manifest, vram_budget_gb=3.0)
    assert plan.feasible
    assert plan.vram_bytes <= plan.vram_budget_bytes - plan.vram_headroom_bytes
    assert len(plan.vram_keys) + len(plan.ram_keys) + len(plan.disk_keys) == 2
    assert plan.row_gather_keys == ["model.embed_tokens.weight"]


def test_describe_is_readable_and_flags_infeasible():
    plan = plan_tiers([_t("big", 4000, 3000)], vram_budget_gb=1.0)
    text = plan.describe()
    assert "Tier plan" in text
    assert "feasible             : False" in text


def test_decode_slice_elems_lowers_the_feasibility_floor():
    """The fix that makes low VRAM budgets actually settable. A flat 512 MB
    scratch reserve put the floor at ~2.06 GB on a 14B-shaped model even
    though the measured peak with small decode slices was 1.62 GB -- so
    budgets the engine could genuinely honour were refused. Deriving the
    reserve from decode_slice_elems ties it to what actually determines it."""
    ts = [_t("lm_head", 1556, 1100, layer=False), _t("layer", 27, 18)]

    coarse = plan_tiers(ts, vram_budget_gb=1.9, decode_slice_elems=1 << 25)
    assert not coarse.feasible, "1.9 GB should not fit a 1.56 GB tensor + 335 MB scratch"

    fine = plan_tiers(ts, vram_budget_gb=1.9, decode_slice_elems=1 << 22)
    assert fine.feasible, (
        "1.9 GB should fit a 1.56 GB tensor + 42 MB scratch + 128 MB slack; "
        "got refusal: " + fine.reason)


def test_explicit_scratch_bytes_still_overrides_the_derived_estimate():
    """Backward compatibility: callers (and this file's other tests) that
    state a flat reserve directly must keep getting exactly that."""
    ts = [_t("a", 100, 90)]
    plan = plan_tiers(ts, vram_budget_gb=0.7, scratch_bytes=int(500e6))
    assert plan.vram_headroom_bytes == int(100e6) + int(500e6)


def test_infeasible_reason_names_the_scratch_lever():
    ts = [_t("big", 2000, 1400)]
    plan = plan_tiers(ts, vram_budget_gb=1.0, decode_slice_elems=1 << 25)
    assert not plan.feasible
    assert "decode_slice_elems" in plan.reason


# -- draft-aware ranking (uses field / self-speculation, mechanism C) ------

def test_uses_field_defaults_to_one_and_is_backward_compatible():
    assert TensorInfo("a", 100, 50, True).uses == 1
    assert TensorInfo("a", 100, 50, True).value_density == pytest.approx(0.5)


def test_higher_uses_scales_value_density_proportionally():
    once = _t("a", 100, 50, uses=1)
    nine_times = _t("b", 100, 50, uses=9)
    assert nine_times.value_density == pytest.approx(9 * once.value_density)


def test_uses_can_flip_the_ranking_between_two_equal_tensors():
    """The whole point of mechanism C: two tensors with identical
    compression cost only tie under plain streaming. A draft layer touched
    9x/sweep (spec_k=8, +1 for verification) must be ranked strictly ahead
    of an equally-compressible non-draft layer once VRAM can't hold both."""
    non_draft = _t("layer.30", 100, 50, uses=1)
    draft = _t("layer.0", 100, 50, uses=9)
    # both would tie without `uses`; VRAM holds exactly one of them (100 MB)
    plan = plan_tiers([non_draft, draft], vram_budget_gb=0.7,
                      scratch_bytes=int(500e6))
    assert plan.vram_keys == ["layer.0"]
    assert plan.disk_keys == ["layer.30"]


def test_plan_from_manifest_marks_draft_layers_hot():
    manifest = {"tensors": {
        "model.layers.0.mlp.down_proj.weight": {"orig_bytes": int(50e6), "comp_bytes": int(25e6)},
        "model.layers.1.mlp.down_proj.weight": {"orig_bytes": int(50e6), "comp_bytes": int(25e6)},
        "model.layers.30.mlp.down_proj.weight": {"orig_bytes": int(50e6), "comp_bytes": int(25e6)},
    }}
    plan_plain = plan_from_manifest(manifest, vram_budget_gb=0.15, scratch_bytes=int(50e6))
    plan_draft = plan_from_manifest(manifest, vram_budget_gb=0.15, scratch_bytes=int(50e6),
                                    draft_layer_indices=[0, 1], draft_uses=9)
    # Plain: all three tie, so ranking (and thus who gets the one VRAM slot)
    # is stable/arbitrary. Draft-aware: layer 0 or 1 MUST win over layer 30.
    assert plan_draft.vram_keys[0].startswith(("model.layers.0.", "model.layers.1."))
    assert "model.layers.30.mlp.down_proj.weight" in plan_draft.disk_keys
    # Sanity: the plain plan is unaffected by an argument nobody passed to it.
    assert len(plan_plain.vram_keys) == 1


def test_plan_from_manifest_no_draft_args_is_unaffected():
    """Default draft_layer_indices=None, draft_uses=1 must reproduce the
    exact same plan as calling plan_from_manifest without those kwargs at
    all -- every existing caller is unaffected."""
    manifest = {"tensors": {
        "model.layers.0.mlp.down_proj.weight": {"orig_bytes": int(50e6), "comp_bytes": int(25e6)},
        "lm_head.weight": {"orig_bytes": int(50e6), "comp_bytes": int(45e6)},
    }}
    a = plan_from_manifest(manifest, vram_budget_gb=0.2, scratch_bytes=int(50e6))
    b = plan_from_manifest(manifest, vram_budget_gb=0.2, scratch_bytes=int(50e6),
                           draft_layer_indices=None, draft_uses=1)
    assert a.vram_keys == b.vram_keys
    assert a.ram_keys == b.ram_keys
    assert a.disk_keys == b.disk_keys


def test_critical_path_policy_preserves_observed_zero_value():
    """An observed off-critical-path tensor is genuinely zero-value; it must
    not be silently promoted by the old traffic proxy."""
    manifest = {"tensors": {
        "off_path": {"orig_bytes": int(50e6), "comp_bytes": int(49e6)},
        "critical": {"orig_bytes": int(50e6), "comp_bytes": int(10e6)},
    }}
    profile = CriticalPathProfile({
        "off_path": TensorCost("off_path", counterfactual_s=0.0, observations=1),
        "critical": TensorCost("critical", counterfactual_s=2.0, observations=1),
    })
    plan = plan_from_manifest(
        manifest, vram_budget_gb=0.15, scratch_bytes=int(50e6),
        critical_path_profile=profile, placement_policy="critical_path")
    assert plan.vram_keys == ["critical"]


# -- stream_only / chunked projection (the lm_head VRAM floor) -------------

def test_stream_only_tensor_is_never_resident_and_lands_on_disk():
    head = TensorInfo("lm_head.weight", int(1556e6), int(1100e6), False,
                      stream_only=True, materialize_override=int(84e6))
    layer = _t("layer", 100, 90)
    plan = plan_tiers([head, layer], vram_budget_gb=5.0, scratch_bytes=int(50e6))
    assert plan.feasible
    assert "lm_head.weight" in plan.disk_keys
    assert "lm_head.weight" not in plan.vram_keys
    assert "layer" in plan.vram_keys
    assert plan.disk_bytes_per_token == int(1100e6)


def test_materialize_override_lowers_the_headroom_reserve():
    head = TensorInfo("lm_head.weight", int(1556e6), int(1100e6), False,
                      stream_only=True, materialize_override=int(84e6))
    small = _t("layer", 27, 18)

    plan = plan_tiers([head, small], vram_budget_gb=0.4, scratch_bytes=int(50e6))
    assert plan.feasible, plan.reason
    assert plan.vram_headroom_bytes == int(84e6) + int(50e6)

    whole = plan_tiers([TensorInfo("lm_head.weight", int(1556e6), int(1100e6), False),
                        small], vram_budget_gb=0.4, scratch_bytes=int(50e6))
    assert not whole.feasible


def test_materialize_bytes_defaults_to_orig_bytes():
    assert TensorInfo("a", 100, 50, True).materialize_bytes == 100
    assert TensorInfo("a", 100, 50, True, materialize_override=10).materialize_bytes == 10


def _hf_cfg(**kw):
    return types.SimpleNamespace(**kw)


def test_kv_cache_bytes_per_token_matches_qwen3_14b_by_hand():
    """Qwen3-14B: 40 layers, 8 KV heads (GQA), head_dim 128, bf16 -- the
    number the KV-cache reserve is meant to catch (docs/RESULTS_LOG.md
    speed audit): 2 * 40 * 8 * 128 * 2 = 163,840 bytes/token, ~160 KB."""
    cfg = _hf_cfg(num_hidden_layers=40, num_attention_heads=40,
                 num_key_value_heads=8, head_dim=128)
    assert kv_cache_bytes_per_token(cfg) == 2 * 40 * 8 * 128 * 2
    assert kv_cache_bytes_per_token(cfg) == 163840


def test_kv_cache_bytes_per_token_falls_back_to_attention_heads_without_gqa():
    """No num_key_value_heads at all -- plain multi-head attention, every
    head has its own K/V (the pre-GQA case)."""
    cfg = _hf_cfg(num_hidden_layers=12, num_attention_heads=12, head_dim=64)
    assert kv_cache_bytes_per_token(cfg) == 2 * 12 * 12 * 64 * 2


def test_kv_cache_bytes_per_token_derives_head_dim_from_hidden_size():
    cfg = _hf_cfg(num_hidden_layers=12, num_attention_heads=12,
                 num_key_value_heads=4, hidden_size=768)
    # head_dim = hidden_size // num_attention_heads = 64
    assert kv_cache_bytes_per_token(cfg) == 2 * 12 * 4 * 64 * 2


def test_kv_cache_bytes_per_token_raises_when_shape_is_unknowable():
    with pytest.raises(ValueError, match="num_hidden_layers"):
        kv_cache_bytes_per_token(_hf_cfg(num_attention_heads=12))
    with pytest.raises(ValueError, match="head_dim"):
        kv_cache_bytes_per_token(_hf_cfg(num_hidden_layers=12, num_attention_heads=12))


def test_kv_reserve_can_make_an_otherwise_feasible_budget_infeasible():
    """The actual bug this closes: a budget that fits without a KV reserve
    must be free to become infeasible once a real context length is
    accounted for, rather than silently ignoring it. scratch_bytes is left
    unset (decode_slice_elems given instead) so activation_slack_bytes
    actually reaches the headroom calculation -- plan_tiers only folds it
    in when deriving scratch_bytes itself; an explicit scratch_bytes
    bypasses that derivation entirely, which is why streaming_engine.py's
    real call site never passes both."""
    tensors = [_t("layer.0", 100, 60), _t("layer.1", 100, 60)]
    # largest materialize_bytes = 100e6; decode_slice_elems=100 keeps decode
    # scratch negligible (1000 bytes) so headroom is ~largest + slack.
    without_kv_reserve = plan_tiers(tensors, vram_budget_gb=0.15, decode_slice_elems=100,
                                    activation_slack_bytes=int(20e6))
    assert without_kv_reserve.feasible, without_kv_reserve.reason

    with_kv_reserve = plan_tiers(tensors, vram_budget_gb=0.15, decode_slice_elems=100,
                                 activation_slack_bytes=int(20e6) + int(40e6))
    assert not with_kv_reserve.feasible
    assert "scratch and activations" in with_kv_reserve.reason
