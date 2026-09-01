import pytest

from scripts.run_bounded_suite import (
    _installed_accelerate_title,
    _installed_airllm_title,
    _installed_deepspeed_title,
    aggregate,
)


def _row(case_id, seconds, tokens=4, repeat=0, **extra):
    row = {
        "case_id": case_id,
        "wall_seconds": seconds * tokens,
        "output_tokens": tokens,
        "seconds_per_token": seconds,
        "peak_vram_gb": 3.9,
        "expected_match": True,
        "cache_drop_succeeded": True,
        "repeat": repeat,
    }
    row.update(extra)
    return row


def test_airllm_method_title_reflects_the_actually_installed_version(monkeypatch):
    """The airllm Method's title used to be the hardcoded literal "AirLLM
    3.1.0", printed as the run's "METHOD: ..." log line and written into
    every result JSON's title field regardless of which airllm was actually
    running -- so upgrading the installed package (as this project did to
    airllm 3.2.0) silently mislabeled every subsequent result. The title is
    now computed from the installed package at call time."""
    def fake_version(name):
        assert name == "airllm"
        return "9.9.9"

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    assert _installed_airllm_title() == "AirLLM 9.9.9"


def test_airllm_method_title_degrades_gracefully_when_not_installed(monkeypatch):
    def raising_version(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", raising_version)
    title = _installed_airllm_title()
    assert "unknown" in title.lower()


def test_accelerate_method_title_reflects_the_actually_installed_version(monkeypatch):
    """Mirrors the airllm title fix above: the accelerate Method's title
    must read the installed package version at call time, not a literal
    baked in when the module first imports, or upgrading the package would
    silently mislabel every subsequent result the same way airllm's did."""
    def fake_version(name):
        assert name == "accelerate"
        return "1.14.0"

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    assert _installed_accelerate_title() == "Hugging Face Accelerate 1.14.0"


def test_accelerate_method_title_degrades_gracefully_when_not_installed(monkeypatch):
    def raising_version(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", raising_version)
    title = _installed_accelerate_title()
    assert "unknown" in title.lower()


def test_deepspeed_method_title_reflects_the_actually_installed_version(monkeypatch):
    def fake_version(name):
        assert name == "deepspeed"
        return "0.18.0"

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    assert _installed_deepspeed_title() == "DeepSpeed ZeRO-Inference 0.18.0"


def test_aggregate_reports_no_dispersion_for_a_single_repeat():
    """A one-repeat run is what this suite did for its entire history. It
    must stay valid and must NOT grow fake spread fields, because a single
    observation per cell has no spread to report."""
    rows = [_row("a", 10.0), _row("b", 20.0)]
    summary = aggregate(rows)
    assert summary["repeats_completed"] == 1
    assert "repeat_stdev_seconds_per_token" not in summary
    assert "per_repeat_seconds_per_token" not in summary


def test_aggregate_reports_spread_across_repeats():
    """Each repeat is a complete sweep of every case, so its seconds/token
    is total_wall/total_tokens for that repeat alone, and the spread across
    repeats is the number that says whether two methods really differ."""
    rows = [
        _row("a", 10.0, repeat=0), _row("b", 20.0, repeat=0),   # repeat 0: 15.0
        _row("a", 12.0, repeat=1), _row("b", 22.0, repeat=1),   # repeat 1: 17.0
        _row("a", 11.0, repeat=2), _row("b", 21.0, repeat=2),   # repeat 2: 16.0
    ]
    summary = aggregate(rows)
    assert summary["repeats_completed"] == 3
    per_repeat = summary["per_repeat_seconds_per_token"]
    assert per_repeat[0]["seconds_per_token"] == pytest.approx(15.0)
    assert per_repeat[1]["seconds_per_token"] == pytest.approx(17.0)
    assert per_repeat[2]["seconds_per_token"] == pytest.approx(16.0)
    assert summary["repeat_median_seconds_per_token"] == pytest.approx(16.0)
    assert summary["repeat_min_seconds_per_token"] == pytest.approx(15.0)
    assert summary["repeat_max_seconds_per_token"] == pytest.approx(17.0)
    assert summary["repeat_stdev_seconds_per_token"] == pytest.approx(1.0)
    assert summary["repeat_relative_stdev"] == pytest.approx(1.0 / 16.0)
    assert summary["all_repeats_complete"] is True


def test_aggregate_withholds_stdev_below_three_repeats():
    """Two points have a defined stdev arithmetically but it carries no
    useful information about run-to-run noise, so it is not reported."""
    rows = [
        _row("a", 10.0, repeat=0), _row("b", 20.0, repeat=0),
        _row("a", 12.0, repeat=1), _row("b", 22.0, repeat=1),
    ]
    summary = aggregate(rows)
    assert summary["repeats_completed"] == 2
    assert "repeat_median_seconds_per_token" in summary
    assert "repeat_stdev_seconds_per_token" not in summary


def test_aggregate_flags_a_repeat_truncated_by_the_deadline():
    """A repeat cut short has fewer cases and is not comparable to a full
    sweep. Averaging them silently would mix different case mixes; the flag
    is what stops a reader treating that as a clean replication."""
    rows = [
        _row("a", 10.0, repeat=0), _row("b", 20.0, repeat=0),
        _row("a", 12.0, repeat=1),  # deadline hit before case "b"
    ]
    summary = aggregate(rows)
    assert summary["repeats_completed"] == 2
    assert summary["all_repeats_complete"] is False


def test_aggregate_treats_legacy_rows_without_a_repeat_field_as_one_repeat():
    """Every result JSON committed before --repeats existed has no `repeat`
    key. Those must keep aggregating exactly as they always did."""
    rows = [
        {"case_id": "a", "wall_seconds": 40.0, "output_tokens": 4,
         "seconds_per_token": 10.0, "peak_vram_gb": 3.9,
         "expected_match": True, "cache_drop_succeeded": True},
    ]
    summary = aggregate(rows)
    assert summary["repeats_completed"] == 1
    assert summary["seconds_per_token"] == pytest.approx(10.0)


def test_process_read_bytes_returns_zero_when_proc_self_io_is_unavailable(monkeypatch):
    """Non-Linux hosts (this dev machine is one) have no /proc/self/io at
    all; the function must degrade to 0, not raise."""
    from scripts.run_bounded_suite import process_read_bytes

    assert isinstance(process_read_bytes(), int)
    assert process_read_bytes() >= 0


def test_process_read_bytes_parses_the_real_proc_self_io_format(tmp_path, monkeypatch):
    from scripts import run_bounded_suite as bounded

    fake_io = tmp_path / "io"
    fake_io.write_text(
        "rchar: 123456\n"
        "wchar: 7890\n"
        "syscr: 42\n"
        "syscw: 10\n"
        "read_bytes: 987654321\n"
        "write_bytes: 0\n"
        "cancelled_write_bytes: 0\n",
        encoding="utf-8")

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/self/io":
            return real_open(fake_io, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    assert bounded.process_read_bytes() == 987654321


def test_aggregate_reports_median_process_read_bytes_per_token():
    rows = [_row("a", 10.0, process_read_bytes_per_token=1000),
            _row("b", 10.0, process_read_bytes_per_token=3000),
            _row("c", 10.0, process_read_bytes_per_token=2000)]
    summary = aggregate(rows)
    assert summary["process_read_bytes_per_token_median"] == 2000


def test_aggregate_io_traffic_is_none_for_legacy_rows_without_the_field():
    """Rows written before process_read_bytes_per_token existed must read
    as "not measured", not silently as zero traffic."""
    summary = aggregate([_row("a", 10.0)])
    assert summary["process_read_bytes_per_token_median"] is None


def test_throttle_decoding_matches_nvml_reason_bits():
    """The raw reason field is hex; nobody scanning a result file decodes it
    by eye, so the boolean is what actually surfaces the confound."""
    from scripts.run_bounded_suite import is_throttled

    assert is_throttled({"throttle_reasons_active": "0x0000000000000000"}) is False
    # GpuIdle alone is a normal state, not a performance throttle.
    assert is_throttled({"throttle_reasons_active": "0x0000000000000001"}) is False
    # SwThermalSlowdown -- the one measured on this project's own reference
    # machine while an uncooled campaign degraded 1.51x across repeats.
    assert is_throttled({"throttle_reasons_active": "0x0000000000000020"}) is True
    assert is_throttled({"throttle_reasons_active": "0x0000000000000004"}) is True  # SwPowerCap
    assert is_throttled({"throttle_reasons_active": "0x0000000000000040"}) is True  # HwThermal
    # Unknown / unavailable must be None, never a confident False.
    assert is_throttled({"throttle_reasons_active": "[N/A]"}) is None
    assert is_throttled({}) is None
    assert is_throttled({"throttle_reasons_active": "not-hex"}) is None


def test_aggregate_reports_thermal_integrity():
    from scripts.run_bounded_suite import aggregate

    clean = [_row("a", 10.0, gpu_thermal={"throttled": False, "temperature_c": "55"}),
             _row("b", 11.0, gpu_thermal={"throttled": False, "temperature_c": "58"})]
    summary = aggregate(clean)
    assert summary["thermally_clean"] is True
    assert summary["thermally_throttled_cells"] == 0
    assert summary["gpu_temperature_c_max"] == 58.0

    dirty = clean + [_row("c", 30.0, gpu_thermal={"throttled": True, "temperature_c": "87"})]
    summary = aggregate(dirty)
    assert summary["thermally_clean"] is False
    assert summary["thermally_throttled_cells"] == 1


def test_aggregate_thermal_integrity_is_none_when_unobserved():
    """Legacy results and non-NVIDIA hosts have no thermal data; that must
    read as 'unknown', not as 'clean'."""
    from scripts.run_bounded_suite import aggregate

    summary = aggregate([_row("a", 10.0)])
    assert summary["thermally_throttled_cells"] is None
    assert "thermally_clean" not in summary


def test_cool_down_reports_recovery_only_when_cool_and_unthrottled(monkeypatch):
    """Regression test for the exact failure mode measured on this project's
    reference machine: a throttled GPU can read a LOW temperature (it is
    generating less heat because it is clocked down), so a temperature-only
    gate would misreport recovery. cool_down() must require the throttle
    flag to also be clear."""
    from scripts import run_bounded_suite as bounded

    # Cool AND clear -- must report reached, on the very first snapshot, so
    # this exercises the real wait loop without waiting on real time.
    monkeypatch.setattr(bounded, "gpu_thermal_snapshot", lambda: {
        "temperature_c": "59", "throttle_reasons_active": "0x0000000000000000",
        "throttled": False})
    result = bounded.cool_down(0.0, max_temperature_c=65.0)
    assert result["cooldown_reached_target"] is True


def test_cool_down_never_reports_recovery_from_temperature_alone(monkeypatch):
    """Same failure mode, but the GPU never actually clears (a stuck driver
    state). Must give up at the hard ceiling rather than loop forever, and
    must not fake-report recovery just because it gave up. Time itself is
    mocked so this does not spend real wall-clock time on the ceiling."""
    from scripts import run_bounded_suite as bounded

    monkeypatch.setattr(bounded, "gpu_thermal_snapshot", lambda: {
        "temperature_c": "59", "throttle_reasons_active": "0x0000000000000020",
        "throttled": True})
    monkeypatch.setattr(bounded.time, "sleep", lambda _seconds: None)
    clock = {"t": 0.0}

    def fake_perf_counter():
        clock["t"] += 120.0  # jump 2 minutes per poll; ceiling trips in ~6 polls
        return clock["t"]

    monkeypatch.setattr(bounded.time, "perf_counter", fake_perf_counter)
    result = bounded.cool_down(0.0, max_temperature_c=65.0)
    assert result["cooldown_reached_target"] is False


def test_cool_down_checks_throttle_even_with_no_cooldown_flags_at_all(monkeypatch):
    """The exact gap this fix closes: benchmark.sh's canonical invocation
    passes neither --cooldown-seconds nor --cooldown-max-temp-c, so callers
    reach cool_down(0.0, None). The old code returned {} immediately in that
    case without ever calling gpu_thermal_snapshot() -- a genuinely
    throttled GPU was invisible unless a caller separately opted in to a
    temperature target. The throttle check must now be unconditional."""
    from scripts import run_bounded_suite as bounded

    monkeypatch.setattr(bounded, "gpu_thermal_snapshot", lambda: {
        "temperature_c": "59", "throttle_reasons_active": "0x0000000000000020",
        "throttled": True})
    monkeypatch.setattr(bounded.time, "sleep", lambda _seconds: None)
    clock = {"t": 0.0}

    def fake_perf_counter():
        clock["t"] += 120.0
        return clock["t"]

    monkeypatch.setattr(bounded.time, "perf_counter", fake_perf_counter)
    result = bounded.cool_down(0.0, None)
    assert result["cooldown_reached_target"] is False


def test_cool_down_returns_immediately_when_not_thermally_throttled(monkeypatch):
    """The common case must stay cheap: a thermally clear GPU with the fully
    default call still returns on the first snapshot, not after waiting."""
    from scripts import run_bounded_suite as bounded

    monkeypatch.setattr(bounded, "gpu_thermal_snapshot", lambda: {
        "temperature_c": "59", "throttle_reasons_active": "0x0000000000000000",
        "throttled": False})
    result = bounded.cool_down(0.0, None)
    assert result["cooldown_reached_target"] is True


def test_cool_down_does_not_wait_for_an_ordinary_software_power_cap(monkeypatch):
    """A laptop power limit is a measured operating condition, not heat that
    an idle wait can clear. Only the thermal bit should hold the cooldown."""
    from scripts import run_bounded_suite as bounded

    calls = {"n": 0}

    def snapshot():
        calls["n"] += 1
        return {
            "temperature_c": "59",
            "throttle_reasons_active": "0x0000000000000004",
            "throttled": True,
            "thermal_throttled": False,
            "power_limited": True,
        }

    monkeypatch.setattr(bounded, "gpu_thermal_snapshot", snapshot)
    result = bounded.cool_down(0.0, max_temperature_c=65.0)

    assert result["cooldown_reached_target"] is True
    assert result["thermal_throttled_after_cooldown"] is False
    assert result["power_limited_after_cooldown"] is True
    assert calls["n"] == 2  # readiness sample plus final audit sample


def test_cool_down_seconds_floor_also_requires_thermal_clear(monkeypatch):
    """Previously the seconds-only floor path (--cooldown-seconds with no
    --cooldown-max-temp-c) never consulted thermal state at all, so it
    would wait out the floor and report done even on a still-hot GPU.
    The floor must now be a minimum, not a substitute for the throttle
    check."""
    from scripts import run_bounded_suite as bounded

    calls = {"n": 0}

    def snapshot():
        calls["n"] += 1
        # Throttled for the first two polls, then clears.
        throttled = calls["n"] <= 2
        return {"temperature_c": "59",
                "throttle_reasons_active": "0x0000000000000020" if throttled else "0x0",
                "throttled": throttled}

    monkeypatch.setattr(bounded, "gpu_thermal_snapshot", snapshot)
    monkeypatch.setattr(bounded.time, "sleep", lambda _seconds: None)
    result = bounded.cool_down(1.0, None)
    assert result["cooldown_reached_target"] is True
    assert calls["n"] >= 3  # had to poll past the still-throttled snapshots


def test_cool_down_unknown_throttle_reading_does_not_block(monkeypatch):
    """No nvidia-smi / non-NVIDIA host: is_throttled() is None, not False.
    That must read as "proceed", never as a confident "still throttled"."""
    from scripts import run_bounded_suite as bounded

    monkeypatch.setattr(bounded, "gpu_thermal_snapshot", lambda: {})
    result = bounded.cool_down(0.0, None)
    assert result["cooldown_reached_target"] is True
