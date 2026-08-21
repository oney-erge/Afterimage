import pytest

from afterimage.runtime.critical_path import (
    CriticalPathProfile, TraceEvent, TraceRecorder, critical_path, replay_speedup,
)


def _events():
    return [
        TraceEvent("read", "read", "disk", 0, 4, tensor_key="w"),
        TraceEvent("decode", "decode", "gpu", 4, 6, ("read",), tensor_key="w"),
        TraceEvent("other", "compute", "cpu", 0, 3),
        TraceEvent("matmul", "compute", "gpu", 6, 11, ("decode", "other"), tensor_key="w"),
    ]


def test_longest_path_and_counterfactual_replay():
    report = critical_path(_events())
    assert report.duration_s == pytest.approx(11)
    assert report.event_ids == ("read", "decode", "matmul")
    assert report.is_critical("decode")
    assert replay_speedup(_events(), {"read": 2}) == pytest.approx(11 / 9)


def test_missing_dependency_and_cycle_are_rejected():
    with pytest.raises(ValueError, match="missing"):
        critical_path([TraceEvent("a", "x", "r", 0, 1, ("missing",))])
    with pytest.raises(ValueError, match="cycle"):
        critical_path([
            TraceEvent("a", "x", "r", 0, 1, ("b",)),
            TraceEvent("b", "x", "r", 0, 1, ("a",)),
        ])


def test_profile_round_trip(tmp_path):
    profile = CriticalPathProfile.from_traces([_events()])
    assert profile.tensors["w"].critical_s == pytest.approx(11)
    # Removing six seconds of preparation exposes the three-second side
    # path, so the actual makespan improvement is only three seconds.
    assert profile.tensors["w"].counterfactual_s == pytest.approx(3)
    path = tmp_path / "profile.json"
    profile.save(path)
    loaded = CriticalPathProfile.load(path)
    assert loaded.score("w", "critical_path") == pytest.approx(3)


def test_recorder_orders_same_resource_events():
    recorder = TraceRecorder()
    with recorder.span("read", "disk"):
        pass
    recorder.record("read", "disk", 1.0, 2.0)
    assert recorder.events[1].dependencies == (recorder.events[0].id,)


def test_trace_recorder_round_trip(tmp_path):
    recorder = TraceRecorder()
    recorder.record("read", "disk", 1.0, 2.0, tensor_key="w")
    path = tmp_path / "trace.json"
    recorder.save(path)
    assert TraceRecorder.load(path) == recorder.events


def test_engine_trace_reset_drops_startup_dependency_ids():
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    engine = StreamingLosslessModel.__new__(StreamingLosslessModel)
    engine.trace = TraceRecorder()
    startup_id = engine.trace.record("transfer", "cuda", 0.0, 1.0)
    engine._last_read_event = {"w": startup_id}
    engine._layer_prepare_events = {0: [startup_id]}
    engine._layer_compute_start = {0: 0.0}

    engine._clear_startup_trace()

    assert engine.trace.events == []
    assert engine._last_read_event == {}
    assert engine._layer_prepare_events == {}
    assert engine._layer_compute_start == {}
