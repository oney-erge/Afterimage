import pytest

from scripts.power_analysis import paired_log_ratios, required_pairs, retrospective_power


def _trial(arm, block, case_seconds):
    return {"arm": arm, "block": block, "rows": [
        {"case_id": case_id, "seconds_per_token": seconds}
        for case_id, seconds in case_seconds.items()]}


def test_paired_log_ratios_matches_by_block_and_case_and_signs_toward_candidate():
    result = {"trials": [
        _trial("control", 0, {"a": 10.0, "b": 20.0}),
        _trial("candidate", 0, {"a": 8.0, "b": 20.0}),  # a: candidate faster; b: tied
        _trial("control", 1, {"a": 10.0}),
        _trial("candidate", 1, {"a": 12.0}),  # candidate slower
    ]}
    ratios = paired_log_ratios(result)
    assert len(ratios) == 3
    # control/candidate = 10/8 > 1 for the faster pair, so its log-ratio is positive
    assert any(r > 0 for r in ratios)
    assert any(r < 0 for r in ratios)
    assert any(r == pytest.approx(0.0) for r in ratios)


def test_paired_log_ratios_ignores_unmatched_rows():
    result = {"trials": [
        _trial("control", 0, {"a": 10.0}),
        _trial("candidate", 0, {"b": 10.0}),  # different case_id, no pair
    ]}
    assert paired_log_ratios(result) == []


def test_required_pairs_increases_with_variance_and_decreases_with_effect_size():
    low_sigma = required_pairs(0.05, minimum_effect=0.08, power_z=0.8416212335729143)
    high_sigma = required_pairs(0.20, minimum_effect=0.08, power_z=0.8416212335729143)
    assert high_sigma > low_sigma > 0

    small_effect = required_pairs(0.10, minimum_effect=0.05, power_z=0.8416212335729143)
    large_effect = required_pairs(0.10, minimum_effect=0.20, power_z=0.8416212335729143)
    assert small_effect > large_effect > 0


def test_retrospective_power_increases_with_n_and_is_bounded():
    small_n = retrospective_power(4, sigma=0.10, minimum_effect=0.05)
    large_n = retrospective_power(200, sigma=0.10, minimum_effect=0.05)
    assert 0.0 <= small_n <= large_n <= 1.0


def test_retrospective_power_matches_a_known_normal_quantile():
    # By construction: if n is exactly the required-pairs count for 80%
    # power, retrospective power at that n should recover ~80%.
    sigma, effect = 0.10, 0.05
    n = required_pairs(sigma, effect, power_z=0.8416212335729143)
    power = retrospective_power(round(n), sigma, effect)
    assert power == pytest.approx(0.80, abs=0.01)
