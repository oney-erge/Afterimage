import pytest

from scripts.run_bounded_suite import _installed_airllm_title, aggregate


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
