import random

import pytest

from afterimage.runtime.critical_path import (
    CriticalPathProfile, TraceEvent, TraceRecorder, compile_topology, critical_path,
    critical_path_fast, replay_speedup,
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


def _report_fields_equal(a, b):
    assert a.duration_s == pytest.approx(b.duration_s)
    assert a.event_ids == b.event_ids
    assert a.earliest_start.keys() == b.earliest_start.keys()
    for k in a.earliest_start:
        assert a.earliest_start[k] == pytest.approx(b.earliest_start[k])
        assert a.latest_start[k] == pytest.approx(b.latest_start[k])
        assert a.slack_s[k] == pytest.approx(b.slack_s[k])


def test_critical_path_fast_matches_critical_path_on_the_reference_graph():
    """The precompiled path (replay_planner.py's CEM/QUBO hot loop) must
    return the exact same report as the dict-based critical_path() it
    replaces there -- same makespan, same critical path, same slack."""
    events = _events()
    topo = compile_topology(events)
    _report_fields_equal(critical_path(events), critical_path_fast(topo))
    overrides = {"read": 0.0, "matmul": 2.5, "not-a-real-id": 9.0}
    _report_fields_equal(critical_path(events, overrides), critical_path_fast(topo, overrides))


def test_critical_path_fast_rejects_the_same_malformed_graphs():
    with pytest.raises(ValueError, match="missing"):
        compile_topology([TraceEvent("a", "x", "r", 0, 1, ("missing",))])
    with pytest.raises(ValueError, match="cycle"):
        compile_topology([
            TraceEvent("a", "x", "r", 0, 1, ("b",)),
            TraceEvent("b", "x", "r", 0, 1, ("a",)),
        ])
    with pytest.raises(ValueError, match="unique"):
        compile_topology([TraceEvent("a", "x", "r", 0, 1), TraceEvent("a", "x", "r", 0, 1)])


def test_critical_path_fast_empty_and_single_event():
    assert critical_path_fast(compile_topology([])).duration_s == 0.0
    solo = [TraceEvent("solo", "read", "disk", 0.0, 3.0)]
    _report_fields_equal(critical_path(solo), critical_path_fast(compile_topology(solo)))


def _random_dag(rng, n_events, n_resources, max_extra_deps=2):
    """Mirrors how TraceRecorder actually builds dependencies: one chain per
    resource plus a few random cross-resource edges to an earlier event, so
    the graph stays acyclic while still exercising multi-parent tie-breaks."""
    events = []
    resource_tail = {}
    for i in range(n_events):
        resource = "r%d" % rng.randrange(n_resources)
        deps = []
        if resource in resource_tail:
            deps.append(resource_tail[resource])
        for _ in range(rng.randrange(max_extra_deps + 1)):
            if events:
                cand = events[rng.randrange(len(events))].id
                if cand not in deps:
                    deps.append(cand)
        rng.shuffle(deps)
        ev = TraceEvent("ev%d" % i, "read", resource, 0.0, rng.uniform(0.0, 5.0), tuple(deps))
        events.append(ev)
        resource_tail[resource] = ev.id
    return events


def test_critical_path_fast_matches_critical_path_on_random_dags():
    """compile_topology()'s index-based Kahn's sort and CSR parent/child
    order must reproduce critical_path()'s tie-breaking (first-wins-tie on
    both the forward-pass parent choice and the makespan-owning event) bit
    for bit, not just agree on the total duration."""
    rng = random.Random(20260822)
    for trial in range(60):
        events = _random_dag(rng, n_events=rng.randrange(1, 120), n_resources=rng.randrange(1, 5))
        topo = compile_topology(events)
        _report_fields_equal(critical_path(events), critical_path_fast(topo))

        overrides = {}
        for _ in range(rng.randrange(0, len(events) + 2)):
            if events and rng.random() < 0.5:
                overrides[events[rng.randrange(len(events))].id] = 0.0
            elif events:
                overrides[events[rng.randrange(len(events))].id] = rng.uniform(0.0, 10.0)
        _report_fields_equal(critical_path(events, overrides), critical_path_fast(topo, overrides))


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
