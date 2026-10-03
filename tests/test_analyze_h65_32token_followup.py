import json
import pathlib

import pytest

from scripts.analyze_h65_32token_followup import analyze, interval, render

ROOT = pathlib.Path(__file__).resolve().parent.parent
D8 = (ROOT / "evidence" / "h65-paper1" / "laptop-rtx3080" / "D8-qwen-32token"
      / "qwen3-14b-h65-matched-decode-practical-20260909-r2-32tok.json")
PROTOCOL = (ROOT / "docs" / "h65" / "protocols"
            / "PROTOCOL-h65-32token-calibration-followup-20261003.md")
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
    assert summary["ci90"][0] > 1.0
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
    # mean log 0.18000, sd log 0.08355, t(2 df, 90%) 2.920 -> +/- 0.14086
    assert low == pytest.approx(1.0399, abs=1e-3)
    assert high == pytest.approx(1.3783, abs=1e-3)


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
    for required in ("two-sided 90%", "lower bound", "at least 3", "token ids",
                     "disk-frozen", "calibration_long"):
        assert required in text, "protocol no longer states %r" % required
