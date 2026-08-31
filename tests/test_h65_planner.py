from afterimage.runtime.critical_path import TraceEvent
from afterimage.runtime.h65_planner import optimize_h65_plan, plan_geometry


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
        minimum_predicted_improvement=0.0)

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
        minimum_predicted_improvement=0.50)

    assert result.report.fallback_to_control
    assert result.report.fallback_reason
    assert result.report.control_choices == {
        name: sum(option.name == name for option in result.plan.choices.values())
        for name in {option.name for option in result.plan.choices.values()}
    }
    assert plan_geometry(manifest, result.plan.choices) == result.report.control_geometry
