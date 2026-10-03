import json
import pathlib

import pytest

from scripts.analyze_h65_32token_followup import analyze, interval, render

ROOT = pathlib.Path(__file__).resolve().parent.parent
D8 = (ROOT / "evidence" / "h65-paper1" / "laptop-rtx3080" / "D8-qwen-32token"
      / "qwen3-14b-h65-matched-decode-practical-20260909-r2-32tok.json")
PROTOCOL = (ROOT / "docs" / "h65" / "protocols"
            / "PROTOCOL-h65-32token-calibration-followup-v3-20261003.md")
CASES = ("a", "b", "c", "d")


def row(block, case, wall, tokens=(1, 2, 3), phys=8e9, logical=18.0):
    return {"repeat": block, "case_id": case, "wall_seconds": wall,
            "seconds_per_token": wall / 32, "output_token_ids": list(tokens),
            "process_read_bytes_per_token": phys, "gb_read_per_token": logical,
            "peak_vram_gb": 2.0, "host_rss_peak_gb": 4.0}


def method(name, walls_by_block, **kwargs):
    return {"method_id": name, "rows": [
        row(block, case, wall, **kwargs)
        for block, wall in enumerate(walls_by_block) for case in CASES]}


def result(*methods):
    return {"blocks_requested": 3, "methods": list(methods)}


def test_faster_arm_with_a_tight_interval_is_called_faster():
    control = method("ctl", [600.0, 610.0, 590.0])
    arm = method("arm", [500.0, 505.0, 495.0])
    summary = analyze(result(control, arm), "ctl", ["arm"])["arms"]["arm"]
    assert summary["speed_ratio_vs_control"] == pytest.approx(1.2, abs=0.01)
    assert summary["ci95"][0] > 1.0
    assert summary["verdict"] == "faster than the control"
    assert summary["request_wins"] == summary["requests"] == 12


def test_slower_arm_and_inconclusive_arm():
    control = method("ctl", [600.0, 610.0, 590.0])
    slower = method("slow", [700.0, 705.0, 695.0])
    noisy = method("noisy", [500.0, 700.0, 590.0])
    arms = analyze(result(control, slower, noisy), "ctl", ["slow", "noisy"])["arms"]
    assert arms["slow"]["verdict"] == "slower than the control"
    assert arms["noisy"]["verdict"] == "inconclusive"


def test_fewer_than_three_complete_blocks_is_never_a_verdict():
    control = method("ctl", [600.0, 610.0])
    arm = method("arm", [400.0, 410.0])
    summary = analyze(result(control, arm), "ctl", ["arm"])
    assert summary["complete_blocks"] == [0, 1]
    assert summary["arms"]["arm"]["verdict"].startswith("insufficient blocks")


def test_a_block_missing_for_one_method_is_excluded_for_all():
    control = method("ctl", [600.0, 610.0, 590.0])
    arm = method("arm", [500.0, 505.0, 495.0])
    arm["rows"] = [r for r in arm["rows"] if not (r["repeat"] == 2 and r["case_id"] == "d")]
    summary = analyze(result(control, arm), "ctl", ["arm"])
    assert summary["complete_blocks"] == [0, 1]


def test_different_output_tokens_invalidate_the_arm_whatever_its_speed():
    control = method("ctl", [600.0, 610.0, 590.0])
    arm = method("arm", [300.0, 300.0, 300.0], tokens=(9, 9, 9))
    summary = analyze(result(control, arm), "ctl", ["arm"])["arms"]["arm"]
    assert not summary["tokens_identical_to_control"]
    assert summary["verdict"].startswith("invalid")


def test_cache_served_share_is_requested_minus_ssd_bytes():
    control = method("ctl", [600.0] * 3)
    arm = method("arm", [500.0] * 3, phys=6e9, logical=12.0)
    assert analyze(result(control, arm), "ctl", ["arm"])["arms"]["arm"][
        "cache_served_share"] == pytest.approx(0.5)


def test_unknown_method_names_the_methods_the_result_has():
    with pytest.raises(KeyError, match="has ctl"):
        analyze(result(method("ctl", [1.0] * 3)), "ctl", ["nope"])


def test_interval_matches_a_hand_computed_three_block_case():
    low, high = interval([1.10, 1.20, 1.30])
    # mean log 0.18000, sd log 0.08355, t(2 df, 95%) 4.303 -> +/- 0.20757
    assert low == pytest.approx(0.9728, abs=1e-3)
    assert high == pytest.approx(1.4734, abs=1e-3)


def test_published_d8_shows_the_disk_control_beating_the_one_token_plan():
    """The follow-up exists because of this result; the analysis must reproduce it."""
    summary = analyze(json.loads(D8.read_text(encoding="utf-8")),
                      "disk-frozen", ["h65-selected"])
    arm = summary["arms"]["h65-selected"]
    assert summary["complete_blocks"] == [0, 1, 2]
    assert arm["speed_ratio_vs_control"] == pytest.approx(17.93 / 22.95, abs=0.01)
    assert arm["verdict"] == "slower than the control"
    assert arm["tokens_identical_to_control"]
    assert arm["cache_served_share"] < 0.10
    control = analyze(json.loads(D8.read_text(encoding="utf-8")),
                      "exact-min", ["disk-frozen"])["arms"]["disk-frozen"]
    assert control["cache_served_share"] > 0.5
    assert "slower than the control" in render(summary)


def test_the_frozen_protocol_states_the_rule_the_script_applies():
    text = " ".join(PROTOCOL.read_text(encoding="utf-8").split())
    for required in ("two-sided 95%", "lower bound", "at least 3", "token ids",
                     "disk-frozen", "calibration_long", "simple-v4-r0",
                     "up to two more times", "above 5%", "17.93", "h65-cal1-ram0",
                     "predicted"):
        assert required in text, "protocol no longer states %r" % required


def test_control_drift_between_blocks_is_flagged():
    control = method("ctl", [600.0, 610.0, 700.0])
    arm = method("arm", [500.0, 505.0, 580.0])
    summary = analyze(result(control, arm), "ctl", ["arm"], 600.0 / 32)
    stability = summary["control_stability"]
    assert stability["drifted"]
    assert stability["relative_difference_from_expected"] > 0
    assert "DRIFTED" in render(summary)


def test_published_d8_control_is_stable_and_thermal_exposure_is_reported():
    summary = analyze(json.loads(D8.read_text(encoding="utf-8")),
                      "disk-frozen", ["h65-selected"], 17.93)
    assert not summary["control_stability"]["drifted"]
    assert abs(summary["control_stability"]["relative_difference_from_expected"]) < 0.01
    exposure = summary["arms"]["h65-selected"]["thermal_exposure"]
    assert exposure["cells"] == 3
    assert exposure["power_limit_seconds"] > 0


def test_an_arm_that_reads_nothing_from_storage_has_no_cache_share():
    """Caught by the small-model rehearsal: a fully VRAM-resident arm requests
    zero bytes, which divided by zero and crashed the scoring step."""
    control = method("ctl", [600.0] * 3)
    resident = method("resident", [100.0] * 3, phys=0.0, logical=0.0)
    summary = analyze(result(control, resident), "ctl", ["resident"])
    assert summary["arms"]["resident"]["cache_served_share"] is None
    assert "resident" in render(summary)


def test_a_storage_counter_that_reads_zero_everywhere_is_unavailable_not_fully_cached():
    """The follow-up ran on a WSL kernel with no /proc/self/io: every row said 0
    bytes read while requesting ~18 GB a token, which must not print as '0 GB, 100%
    cache'."""
    control = method("ctl", [600.0] * 3, phys=0.0, logical=18.0)
    arm = method("arm", [500.0] * 3, phys=0.0, logical=12.0)
    summary = analyze(result(control, arm), "ctl", ["arm"])["arms"]["arm"]
    assert summary["ssd_gb_per_token"] is None
    assert summary["cache_served_share"] is None
    assert "n/a" in render(analyze(result(control, arm), "ctl", ["arm"]))
