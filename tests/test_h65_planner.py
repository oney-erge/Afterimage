from afterimage.runtime.critical_path import TraceEvent
import pytest

from afterimage.runtime.h65_planner import (
    PlanGeometry, _ReplayScorer, _guard_failures, optimize_h65_plan,
    plan_geometry,
)
from afterimage.runtime.representations import RepresentationOption


MB = 1_000_000


def _manifest():
    tensors = {}
    offset = 0
    for key in ("a", "b", "c", "d"):
        tensors[key] = {
            "orig_bytes": 30 * MB,
            "comp_bytes": 15 * MB,
            "compressed": True,
            "blobs": {
                "codes": {"offset": offset, "nbytes": 10 * MB},
                "tables": {"offset": offset + 10 * MB, "nbytes": 5 * MB},
            },
        }
        offset += 15 * MB
    return {"tensors": tensors}


def _trace(read_seconds):
    events = []
    tail = None
    clock = 0.0
    for index, key in enumerate(("a", "b", "c", "d")):
        read = "read-%d" % index
        decode = "decode-%d" % index
        duration = read_seconds[key]
        events.append(TraceEvent(
            read, "read", "storage", clock, clock + duration,
            (() if tail is None else (tail,)), key, 15 * MB))
        clock += duration
        events.append(TraceEvent(
            decode, "decode", "cuda", clock, clock + 0.1,
            (read,), key, 30 * MB))
        clock += 0.1
        tail = decode
    return events


def test_h65_swaps_toward_whole_trace_benefit_under_matched_budget():
    manifest = _manifest()
    result = optimize_h65_plan(
        manifest, [_trace({"a": 0.1, "b": 0.2, "c": 4.0, "d": 3.0})],
        vram_budget_gb=0.20, ram_budget_gb=0.03, h2d_gbps=1.0,
        decode_slice_elems=1, search_iterations=0,
        minimum_predicted_improvement=0.0,
        minimum_trace_count=1, validation_trace_count=0,
        risk_penalty_weight=0.0, require_causal_trace=False,
        require_live_validation=False)

    report = result.report
    assert report.treatment_diverged
    assert report.predicted_improvement > 0
    assert not report.fallback_to_control
    assert result.plan.vram_bytes + result.plan.vram_headroom_bytes <= int(0.20e9)
    assert result.plan.ram_bytes <= int(0.03e9)
    assert report.candidate_geometry.disk_read_calls_per_sweep <= (
        report.control_geometry.disk_read_calls_per_sweep)
    assert report.candidate_geometry.disk_bytes_per_sweep <= (
        report.control_geometry.disk_bytes_per_sweep)


def test_h65_falls_back_when_candidate_does_not_clear_gate():
    manifest = _manifest()
    result = optimize_h65_plan(
        manifest, [_trace({key: 1.0 for key in ("a", "b", "c", "d")})],
        vram_budget_gb=0.20, ram_budget_gb=0.03, h2d_gbps=1.0,
        decode_slice_elems=1, search_iterations=0,
        minimum_predicted_improvement=0.50,
        minimum_trace_count=1, validation_trace_count=0,
        require_causal_trace=False)

    assert result.report.fallback_to_control
    assert result.report.fallback_reason
    assert result.report.control_choices == {
        name: sum(option.name == name for option in result.plan.choices.values())
        for name in {option.name for option in result.plan.choices.values()}
    }
    assert plan_geometry(manifest, result.plan.choices) == result.report.control_geometry


def test_h65_requires_multiple_traces_by_default():
    result = optimize_h65_plan(
        _manifest(), [_trace({"a": 0.1, "b": 0.2, "c": 4.0, "d": 3.0})],
        vram_budget_gb=0.20, ram_budget_gb=0.03, h2d_gbps=1.0,
        decode_slice_elems=1, search_iterations=0,
        minimum_predicted_improvement=0.0,
        require_causal_trace=False, require_live_validation=False)

    assert result.report.fallback_to_control
    assert "at least 3" in result.report.fallback_reason
    assert result.report.minimum_trace_count == 3


def test_h65_rejects_legacy_trace_without_scheduler_edges():
    with pytest.raises(ValueError, match="scheduler-aware"):
        optimize_h65_plan(
            _manifest(), [_trace({key: 1.0 for key in ("a", "b", "c", "d")})],
            vram_budget_gb=0.20, ram_budget_gb=0.03, h2d_gbps=1.0,
            minimum_trace_count=1, validation_trace_count=0)


def test_h65_holds_out_last_trace_and_gates_on_every_trace():
    traces = [
        _trace({"a": 0.1, "b": 0.2, "c": 4.0, "d": 3.0}),
        _trace({"a": 0.2, "b": 0.1, "c": 3.5, "d": 3.2}),
        _trace({"a": 0.1, "b": 0.2, "c": 3.8, "d": 3.1}),
    ]
    result = optimize_h65_plan(
        _manifest(), traces,
        vram_budget_gb=0.20, ram_budget_gb=0.03, h2d_gbps=1.0,
        decode_slice_elems=1, search_iterations=0,
        minimum_predicted_improvement=0.0,
        require_causal_trace=False, require_live_validation=False)

    report = result.report
    assert report.training_trace_count == 2
    assert report.validation_trace_count == 1
    assert len(report.training_improvements) == 2
    assert len(report.validation_improvements) == 1
    assert report.conservative_predicted_improvement == pytest.approx(
        min(report.training_improvements + report.validation_improvements))
    assert not report.fallback_to_control


def test_decoded_ram_replay_replaces_decode_with_measured_h2d_cost():
    manifest = {"tensors": {"w": {
        "orig_bytes": 100 * MB,
        "comp_bytes": 50 * MB,
        "compressed": True,
        "blobs": {"codes": {"offset": 0, "nbytes": 50 * MB}},
    }}}
    trace = [
        TraceEvent("read", "read", "storage", 0.0, 0.5,
                   tensor_key="w", nbytes=50 * MB),
        TraceEvent("decode", "decode", "cuda", 0.5, 2.5,
                   ("read",), "w", 100 * MB),
    ]
    choice = {"w": RepresentationOption(
        "w", "decoded_ram", ram_bytes=100 * MB,
        storage_bytes=50 * MB)}

    slow = _ReplayScorer(
        manifest, [trace], h2d_gbps=0.1,
        fragmentation_penalty_weight=0.0)
    fast = _ReplayScorer(
        manifest, [trace], h2d_gbps=1.0,
        fragmentation_penalty_weight=0.0)

    assert slow.replay(choice) == pytest.approx(1.0)
    assert fast.replay(choice) == pytest.approx(0.1)


def _three_trace_live_plan(**live):
    traces = [
        _trace({"a": 0.1, "b": 0.2, "c": 4.0, "d": 3.0}),
        _trace({"a": 0.2, "b": 0.1, "c": 3.5, "d": 3.2}),
        _trace({"a": 0.1, "b": 0.2, "c": 3.8, "d": 3.1}),
    ]
    return optimize_h65_plan(
        _manifest(), traces,
        vram_budget_gb=0.20, ram_budget_gb=0.03, h2d_gbps=1.0,
        decode_slice_elems=1, search_iterations=0,
        minimum_predicted_improvement=0.0,
        require_causal_trace=False, **live)


def test_h65_requires_paired_live_blocks_after_replay_gate():
    result = _three_trace_live_plan()

    assert result.report.fallback_to_control
    assert "awaits 2 paired live validation blocks" in result.report.fallback_reason


def test_h65_deploys_only_when_every_live_block_clears_gate():
    result = _three_trace_live_plan(
        live_control_seconds=[10.0, 10.0],
        live_candidate_seconds=[9.0, 8.8],
        live_validation_eligible=True)

    assert not result.report.fallback_to_control
    assert result.report.conservative_live_improvement == pytest.approx(0.1)


def test_h65_falls_back_when_one_live_block_regresses():
    result = _three_trace_live_plan(
        live_control_seconds=(10.0, 10.0),
        live_candidate_seconds=(9.0, 10.1),
        live_validation_eligible=True)

    assert result.report.fallback_to_control
    assert "worst paired live improvement" in result.report.fallback_reason


def test_traffic_topology_is_reported_but_not_a_hidden_search_constraint():
    control = PlanGeometry(100, 10, 4, 2, 0, 2, 4)
    candidate = PlanGeometry(105, 11, 5, 1, 1, 2, 5)

    assert _guard_failures(control, candidate) == ()
    assert set(_guard_failures(
        control, candidate,
        maximum_disk_byte_increase=0.0,
        maximum_disk_call_increase=0.0,
        maximum_layer_call_increase=0.0)) == {
            "disk_bytes_increased",
            "disk_read_calls_increased",
            "maximum_layer_read_calls_increased",
        }


def test_compressed_ram_can_be_disabled_for_placement_only_ablation():
    result = _three_trace_live_plan(
        enable_compressed_ram=False, require_live_validation=False)

    assert result.report.compressed_ram_enabled is False
    assert "compressed_ram" not in result.report.candidate_choices
