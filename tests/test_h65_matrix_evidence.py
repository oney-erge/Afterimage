from types import SimpleNamespace

import pytest

from scripts.run_h65_paper_matrix import (
    allocator_cap_gb,
    physical_vram_ceiling_gb,
    representation_ablation_status,
    whole_cell_budget_ok,
)


@pytest.mark.parametrize("peak,source,expected", [
    (24., "whole_cell_nvidia_smi_delta", True),
    (24.15, "whole_cell_nvidia_smi_delta", False),
    (23., "whole_cell_torch_allocator", False),
    (None, "whole_cell_nvidia_smi_delta", False),
    (float("nan"), "whole_cell_nvidia_smi_delta", False),
    (-1., "whole_cell_nvidia_smi_delta", False),
])
def test_physical_memory_gate(peak, source, expected):
    assert whole_cell_budget_ok([{"peak_vram_gb": peak, "peak_vram_source": source}], 24.) is expected


def test_empty_rows_are_not_evidence_of_fitting():
    assert not whole_cell_budget_ok([], 24.)


def test_identical_choices_do_not_identify_full_representation_gain():
    placement = SimpleNamespace(choices={"a": SimpleNamespace(name="decoded_ram", prepare_s=1.)})
    full = SimpleNamespace(choices={"a": SimpleNamespace(name="decoded_ram", prepare_s=2.)})
    result = representation_ablation_status(placement, full)
    assert result["plans_identical"]
    assert not result["representation_effect_identifiable"]
    full.choices["a"].name = "compressed_ram"
    result = representation_ablation_status(placement, full)
    assert result["representation_effect_identifiable"]
    assert result["compressed_ram_tensor_count"] == 1


def test_worker_and_calibration_share_explicit_execution_settings():
    from scripts.run_h65_paper_matrix import common_overrides, worker_config

    args = SimpleNamespace(vram_gb=26, vram_cap_gb=28,
                           physical_vram_ceiling_gb=30, ram_gb=16,
                           vram_safety_margin_gb=4,
                           decode_slice_elems=1024, reuse_decode_tables=True, warmup_tokens=1,
                           model="model", store="store", cooldown_seconds=10,
                           cooldown_max_temp_c=75, cell_timeout_minutes=6)
    overrides = common_overrides(args)
    assert overrides["reuse_decode_tables"] is True
    assert overrides["vram_budget_gb"] == 26
    assert overrides["vram_cap_gb"] == 28
    for split in ("calibration", "evaluation"):
        config = worker_config(args=args, method_id="test", overrides=overrides,
                               block=0, split=split, case_ids=("case",), max_new_tokens=1)
        assert config["warmup_tokens"] == 1
        assert config["overrides"]["reuse_decode_tables"] is True
        assert config["budget"] == {
            "vram_gb": 26,
            "vram_cap_gb": 28,
            "physical_vram_ceiling_gb": 30,
            "ram_gb": 16,
        }


def test_memory_limit_defaults_preserve_historical_behavior():
    args = SimpleNamespace(vram_gb=8, vram_cap_gb=None,
                           physical_vram_ceiling_gb=None)
    assert allocator_cap_gb(args) == 8
    assert physical_vram_ceiling_gb(args) == 8


def test_physical_ceiling_defaults_to_explicit_allocator_cap():
    args = SimpleNamespace(vram_gb=8, vram_cap_gb=10,
                           physical_vram_ceiling_gb=None)
    assert allocator_cap_gb(args) == 10
    assert physical_vram_ceiling_gb(args) == 10
