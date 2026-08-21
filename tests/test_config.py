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


def test_rejects_negative_prefetch_depth():
    with pytest.raises(ValueError, match="io_prefetch_depth"):
        EngineConfig(io_prefetch_depth=-1)


def test_zero_prefetch_depth_is_valid_and_disables_prefetch():
    cfg = EngineConfig(io_prefetch_depth=0)
    assert cfg.io_prefetch_depth == 0


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
