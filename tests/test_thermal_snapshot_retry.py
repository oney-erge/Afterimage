from scripts import run_bounded_suite as rbs


def test_missing_snapshot_retries_and_uses_observed_value(monkeypatch):
    responses = iter([None, "1000,2000,45,100,600,0x0,0,0,0,0,0"])
    waits = []
    monkeypatch.setattr(rbs, "command_output", lambda _: next(responses))
    monkeypatch.setattr(rbs.time, "sleep", waits.append)
    assert rbs.gpu_thermal_snapshot()["temperature_c"] == "45"
    assert waits == [0.5]


def test_missing_snapshot_remains_unobservable_after_bounded_retries(monkeypatch):
    calls = []
    monkeypatch.setattr(rbs, "command_output", lambda _: calls.append(1))
    monkeypatch.setattr(rbs.time, "sleep", lambda _: None)
    assert rbs.gpu_thermal_snapshot() == {}
    assert len(calls) == 4
