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


def test_calibration_cases_default_to_the_historical_one_token_set():
    from scripts.run_h65_paper_matrix import (
        DEFAULT_CALIBRATION_CASES, resolve_calibration_cases)

    assert resolve_calibration_cases("calibration", None) == DEFAULT_CALIBRATION_CASES
    assert resolve_calibration_cases("calibration", "a, b") == ("a", "b")
    assert resolve_calibration_cases("calibration_long", None) == (
        "calibration-long-explain", "calibration-long-code",
        "calibration-long-compare")


def test_trace_sweep_count_counts_forward_passes():
    from scripts.run_h65_paper_matrix import trace_sweep_count

    events = [SimpleNamespace(kind=kind) for kind in (
        "forward_start", "read", "compute", "forward_end",
        "forward_start", "read", "compute", "forward_end")]
    assert trace_sweep_count(events) == 2


def test_multi_token_calibration_must_actually_generate_every_token():
    from scripts.run_h65_paper_matrix import calibration_rows_error

    full = {"rows": [{"case_id": "a", "output_tokens": 32}]}
    early = {"rows": [{"case_id": "a", "output_tokens": 32},
                      {"case_id": "b", "output_tokens": 9}]}
    assert calibration_rows_error(full, 32) is None
    assert "stopped early" in calibration_rows_error(early, 32)
    assert "recorded no rows" in calibration_rows_error({"rows": []}, 32)
    # The one-token default is unchanged: it never inspected rows.
    assert calibration_rows_error(early, 1) is None


@pytest.mark.parametrize("extra,message", [
    (["--reuse-calibration-traces", "somewhere"], "requires --plan-only"),
    (["--plan-only", "--calibration-tokens", "0"], "at least 1"),
    (["--calibration-split", "calibration_long", "--calibration-cases",
      "logic-ravens,json-status,copy-nonce"], "calibration_long split"),
])
def test_new_calibration_flags_reject_invalid_combinations(
        monkeypatch, capsys, extra, message):
    from scripts import run_h65_paper_matrix as matrix

    monkeypatch.setattr("sys.argv", [
        "run_h65_paper_matrix.py", "--store", "s", "--h2d", "h",
        "--out", "o", *extra])
    with pytest.raises(SystemExit) as raised:
        matrix.main()
    assert raised.value.code == 2
    assert message in capsys.readouterr().err
