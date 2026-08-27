"""paired_block_log_ratios() is the block-level statistic this project's own
methodology review recommended in place of pooling every (block, case) pair
as if it were an independent observation: one d_b per randomized block
rather than one point per case. These tests exercise it directly with
synthetic rows, without touching CUDA or any real model.
"""
from __future__ import annotations

import math

import pytest

from scripts.run_paper_comparison import (
    CONTROL_METHOD,
    DEFAULT_METHODS,
    _shuffled_order,
    paired_block_log_ratios,
)


def _row(case_id: str, seconds_per_token: float, block: int) -> dict:
    return {"case_id": case_id, "seconds_per_token": seconds_per_token, "repeat": block}


def test_control_is_one_of_the_default_methods():
    """paired_block_log_ratios compares every other method against
    CONTROL_METHOD; the default method set must actually include it or the
    comparison silently produces nothing."""
    assert CONTROL_METHOD in DEFAULT_METHODS


def test_candidate_exactly_twice_as_fast_gives_log_two_and_speedup_two():
    rows_by_method = {
        "exact-min": [_row("a", 10.0, 0), _row("b", 20.0, 0),
                      _row("a", 12.0, 1), _row("b", 22.0, 1)],
        "spec-fixed": [_row("a", 5.0, 0), _row("b", 10.0, 0),
                       _row("a", 6.0, 1), _row("b", 11.0, 1)],
    }
    result = paired_block_log_ratios(rows_by_method, "exact-min", "spec-fixed")
    assert result["blocks_compared"] == 2
    assert result["median_candidate_speedup_vs_control"] == pytest.approx(2.0, rel=1e-6)
    assert result["median_log_ratio"] == pytest.approx(math.log(2.0), rel=1e-6)


def test_candidate_slower_gives_speedup_below_one():
    rows_by_method = {
        "exact-min": [_row("a", 10.0, 0)],
        "airllm": [_row("a", 30.0, 0)],
    }
    result = paired_block_log_ratios(rows_by_method, "exact-min", "airllm")
    assert result["median_candidate_speedup_vs_control"] < 1.0


def test_reports_one_median_per_block_not_pooled_across_blocks():
    """The whole point of this statistic: dispersion is over blocks, so a
    3-case, 2-block run must report exactly 2 values, not 6."""
    rows_by_method = {
        "exact-min": [_row("a", 10.0, 0), _row("b", 10.0, 0), _row("c", 10.0, 0),
                      _row("a", 10.0, 1), _row("b", 10.0, 1), _row("c", 10.0, 1)],
        "dfloat11": [_row("a", 9.0, 0), _row("b", 11.0, 0), _row("c", 10.0, 0),
                     _row("a", 8.0, 1), _row("b", 12.0, 1), _row("c", 10.0, 1)],
    }
    result = paired_block_log_ratios(rows_by_method, "exact-min", "dfloat11")
    assert result["blocks_compared"] == 2
    assert len(result["block_log_ratios"]) == 2


def test_stdev_across_blocks_only_reported_from_two_blocks_up():
    rows_by_method = {
        "exact-min": [_row("a", 10.0, 0)],
        "accelerate": [_row("a", 10.0, 0)],
    }
    result = paired_block_log_ratios(rows_by_method, "exact-min", "accelerate")
    assert result["blocks_compared"] == 1
    assert "log_ratio_stdev_across_blocks" not in result


def test_a_block_with_no_shared_cases_is_skipped_not_treated_as_zero():
    """A block where the candidate failed every case (no rows at all) must
    not silently contribute a fabricated log-ratio of 0."""
    rows_by_method = {
        "exact-min": [_row("a", 10.0, 0), _row("a", 10.0, 1)],
        "airllm": [_row("a", 10.0, 0)],  # block 1 missing entirely
    }
    result = paired_block_log_ratios(rows_by_method, "exact-min", "airllm")
    assert result["blocks_compared"] == 1


def test_no_shared_blocks_at_all_reports_zero_not_a_crash():
    result = paired_block_log_ratios(
        {"exact-min": [_row("a", 10.0, 0)], "airllm": []}, "exact-min", "airllm")
    assert result == {"blocks_compared": 0}


def test_shuffled_order_is_a_permutation_not_a_subset():
    import random

    methods = list(DEFAULT_METHODS)
    order = _shuffled_order(methods, random.Random(0))
    assert sorted(order) == sorted(methods)
    assert order is not methods  # does not mutate/alias the input list


def test_shuffled_order_uses_the_given_random_instance_deterministically():
    import random

    methods = list(DEFAULT_METHODS)
    first = _shuffled_order(methods, random.Random(42))
    second = _shuffled_order(methods, random.Random(42))
    assert first == second
