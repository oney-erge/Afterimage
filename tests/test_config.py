import pytest

from afterimage.runtime.config import EngineConfig


def test_default_is_lossless():
    """The default must be lossless -- the AirLLM comparison depends on it,
    and a silent lossy default would invalidate every accuracy claim."""
    cfg = EngineConfig()
    assert cfg.quantize is None
    assert cfg.is_lossless
    assert "LOSSLESS" in cfg.describe()


def test_q8_is_opt_in_and_declares_itself_lossy():
    cfg = EngineConfig(quantize="q8")
    assert not cfg.is_lossless
    assert "NOT bit-exact" in cfg.describe()


def test_rejects_unknown_quantize_mode():
    with pytest.raises(ValueError, match="quantize must be"):
        EngineConfig(quantize="q4")


def test_rejects_non_power_of_two_block_chunks():
    with pytest.raises(ValueError, match="power of 2"):
        EngineConfig(block_chunks=37)
    EngineConfig(block_chunks=32)  # valid


def test_rejects_out_of_range_max_bits():
    with pytest.raises(ValueError):
        EngineConfig(max_bits=17)
    with pytest.raises(ValueError):
        EngineConfig(max_bits=0)


def test_rejects_bad_chunk_size():
    with pytest.raises(ValueError):
        EngineConfig(chunk_size=0)


def test_default_uses_legacy_fixed_residency():
    cfg = EngineConfig()
    assert not cfg.uses_tiered_residency
    assert "legacy fixed policy" in cfg.describe()


def test_vram_budget_enables_tiered_residency():
    cfg = EngineConfig(vram_budget_gb=4.0)
    assert cfg.uses_tiered_residency
    assert "VRAM 4.00 GB" in cfg.describe()


def test_ram_budget_without_vram_budget_is_rejected():
    """A RAM budget only means something relative to a VRAM budget -- the
    planner ranks VRAM residency first, RAM second. Accepting ram_budget_gb
    alone would silently do nothing, which is worse than refusing it."""
    with pytest.raises(ValueError, match="requires vram_budget_gb"):
        EngineConfig(ram_budget_gb=8.0)


def test_max_context_without_vram_budget_is_rejected():
    """Same reasoning as ram_budget_gb: with no VRAM budget the legacy
    fixed residency policy applies and never refuses an infeasible plan,
    so a KV-cache reserve would have nothing to attach a refusal to."""
    with pytest.raises(ValueError, match="requires vram_budget_gb"):
        EngineConfig(max_context=8192)


def test_max_context_must_be_positive():
    with pytest.raises(ValueError, match="max_context"):
        EngineConfig(vram_budget_gb=4.0, max_context=0)


def test_max_context_with_vram_budget_is_accepted():
    cfg = EngineConfig(vram_budget_gb=4.0, max_context=8192)
    assert cfg.max_context == 8192


def test_rejects_negative_prefetch_depth():
    with pytest.raises(ValueError, match="io_prefetch_depth"):
        EngineConfig(io_prefetch_depth=-1)


def test_zero_prefetch_depth_is_valid_and_disables_prefetch():
    cfg = EngineConfig(io_prefetch_depth=0)
    assert cfg.io_prefetch_depth == 0


def test_existing_fixed_prefetch_depth_is_not_capped_by_adaptive_default():
    assert EngineConfig(io_prefetch_depth=16).io_prefetch_depth == 16
    with pytest.raises(ValueError, match="max_depth"):
        EngineConfig(io_prefetch_depth=16, prefetch_policy="pi")


def test_bayesian_prefetch_and_qubo_replay_are_explicit_opt_in():
    cfg = EngineConfig(prefetch_policy="bayes_probit", io_prefetch_depth=2,
                       io_prefetch_max_depth=8)
    assert cfg.prefetch_policy == "bayes_probit"
    qubo = EngineConfig(placement_policy="replay_qubo",
                        replay_plan_state="plan.json", vram_budget_gb=4.0)
    assert qubo.placement_policy == "replay_qubo"
    extent = EngineConfig(placement_policy="replay_extent_qubo",
                          replay_plan_state="plan.json", vram_budget_gb=4.0)
    assert extent.placement_policy == "replay_extent_qubo"


def test_coalesced_storage_reads_are_explicit_and_bounded():
    cfg = EngineConfig(storage_read_policy="coalesced_extents",
                       storage_extent_max_bytes=4096,
                       storage_extent_max_gap_bytes=16)
    assert cfg.storage_extent_max_bytes == 4096
    tensor_scoped = EngineConfig(storage_read_policy="tensor_extents",
                                 storage_extent_max_bytes=1024)
    assert tensor_scoped.storage_read_policy == "tensor_extents"
    with pytest.raises(ValueError, match="storage_read_policy"):
        EngineConfig(storage_read_policy="magic")
    with pytest.raises(ValueError, match="max_bytes"):
        EngineConfig(storage_extent_max_bytes=0)


def test_spec_target_cache_is_explicit_and_fingerprinted():
    control = EngineConfig(draft_mode="model")
    candidate = EngineConfig(draft_mode="model", spec_target_cache=True)
    assert not control.spec_target_cache
    assert candidate.spec_target_cache
    assert candidate.fingerprint() != control.fingerprint()


def test_require_pinned_ram_is_a_strict_ram_tier_contract():
    with pytest.raises(ValueError, match="ram_budget_gb"):
        EngineConfig(require_pinned_ram=True)
    with pytest.raises(ValueError, match="decoded"):
        EngineConfig(vram_budget_gb=2.0, ram_budget_gb=1.0,
                     ram_tier_format="compressed", require_pinned_ram=True)
    cfg = EngineConfig(vram_budget_gb=2.0, ram_budget_gb=1.0,
                       require_pinned_ram=True)
    assert cfg.require_pinned_ram


def test_ram_tier_format_default_is_decoded():
    cfg = EngineConfig()
    assert cfg.ram_tier_format == "decoded"


def test_rejects_bad_ram_tier_format():
    with pytest.raises(ValueError, match="ram_tier_format"):
        EngineConfig(ram_tier_format="raw")


# -- adaptive speculation (PROPOSAL_ADAPTIVE.md) ----------------------------

def test_draft_mode_default_is_none():
    cfg = EngineConfig()
    assert cfg.draft_mode == "none"


def test_rejects_bad_draft_mode():
    with pytest.raises(ValueError, match="draft_mode"):
        EngineConfig(draft_mode="cpu")


def test_self_draft_requires_exit_layer():
    with pytest.raises(ValueError, match="draft_exit_layer"):
        EngineConfig(draft_mode="self")
    with pytest.raises(ValueError, match="draft_exit_layer"):
        EngineConfig(draft_mode="self", draft_exit_layer=0)
    EngineConfig(draft_mode="self", draft_exit_layer=8)  # valid


def test_model_draft_mode_does_not_require_exit_layer():
    EngineConfig(draft_mode="model")  # must not raise


def test_rejects_bad_spec_k():
    with pytest.raises(ValueError, match="spec_k"):
        EngineConfig(spec_k=0)


def test_rejects_bad_spec_k_policy():
    with pytest.raises(ValueError, match="spec_k_policy"):
        EngineConfig(spec_k_policy="nn")


def test_pin_draft_layers_requires_self_draft_mode():
    with pytest.raises(ValueError, match="draft_mode='self'"):
        EngineConfig(draft_mode="model", pin_draft_layers=True, vram_budget_gb=4.0)


def test_pin_draft_layers_requires_vram_budget():
    with pytest.raises(ValueError, match="vram_budget_gb"):
        EngineConfig(draft_mode="self", draft_exit_layer=8, pin_draft_layers=True)


def test_pin_draft_layers_valid_combination():
    cfg = EngineConfig(draft_mode="self", draft_exit_layer=8,
                       pin_draft_layers=True, vram_budget_gb=4.0)
    assert cfg.pin_draft_layers


def test_measured_placement_requires_profile_and_real_budget():
    with pytest.raises(ValueError, match="critical_path_profile"):
        EngineConfig(placement_policy="critical_path", vram_budget_gb=4.0)
    with pytest.raises(ValueError, match="vram_budget_gb"):
        EngineConfig(placement_policy="critical_path",
                     critical_path_profile="trace-profile.json")
    cfg = EngineConfig(placement_policy="critical_path",
                       critical_path_profile="trace-profile.json",
                       vram_budget_gb=4.0)
    assert cfg.placement_policy == "critical_path"


def test_replay_cem_requires_frozen_plan_and_budget():
    with pytest.raises(ValueError, match="replay_plan_state"):
        EngineConfig(placement_policy="replay_cem", vram_budget_gb=4.0)
    with pytest.raises(ValueError, match="vram_budget_gb"):
        EngineConfig(placement_policy="replay_cem", replay_plan_state="plan.json")
    cfg = EngineConfig(placement_policy="replay_cem",
                       replay_plan_state="plan.json", vram_budget_gb=4.0)
    assert cfg.replay_plan_state == "plan.json"


def test_ram_overlay_head_requires_decoded_ram_and_tiered_budget():
    with pytest.raises(ValueError, match="ram_budget_gb"):
        EngineConfig(lm_head_policy="ram_overlay")
    with pytest.raises(ValueError, match="decoded RAM"):
        EngineConfig(lm_head_policy="ram_overlay", vram_budget_gb=2.0,
                     ram_budget_gb=2.0, ram_tier_format="compressed")
    cfg = EngineConfig(lm_head_policy="ram_overlay", vram_budget_gb=2.0,
                       ram_budget_gb=2.0)
    assert cfg.exactness_contract == "reference_execution_equivalent"


def test_config_round_trip_and_fingerprint_are_strict_and_stable():
    cfg = EngineConfig(prefetch_policy="pi", trace_events=True)
    assert EngineConfig.from_dict(cfg.to_dict()) == cfg
    assert EngineConfig.from_dict(cfg.to_dict()).fingerprint() == cfg.fingerprint()
    with pytest.raises(ValueError, match="unknown EngineConfig fields"):
        EngineConfig.from_dict({"not_a_setting": True})


def test_spec_policy_can_be_frozen_for_held_out_evaluation():
    cfg = EngineConfig(spec_policy_state="calibrated.json", spec_policy_learn=False)
    assert not cfg.spec_policy_learn


def test_trace_output_cannot_silently_write_an_empty_trace():
    with pytest.raises(ValueError, match="trace_events"):
        EngineConfig(trace_output="trace.json")
    EngineConfig(trace_events=True, trace_output="trace.json")


def test_vram_safety_margin_requires_a_tiered_budget():
    with pytest.raises(ValueError, match="vram_safety_margin_gb"):
        EngineConfig(vram_safety_margin_gb=0.5)
    EngineConfig(vram_budget_gb=8.0, vram_safety_margin_gb=0.5)


def test_exactness_contract_distinguishes_greedy_and_sampling():
    assert EngineConfig().exactness_contract == "reference_execution_equivalent"
    assert EngineConfig(draft_mode="model").exactness_contract == "distribution_exact"
    assert EngineConfig(lm_head_policy="certified_mips").exactness_contract == "greedy_token_exact"
    with pytest.raises(ValueError, match="mips_index_ram_limit_gb"):
        EngineConfig(mips_index_ram_limit_gb=0)
