"""build_amortization_curve() is the pure reduction step of H19 (Candidate-
Amortization Hypothesis): raw per-cell rows -> the median/min/max latency
curve plus each point's ratio to the N=1 baseline, which is the number that
actually answers the hypothesis. These tests exercise it directly with
synthetic rows, no CUDA or real model involved.
"""
from __future__ import annotations

import pytest

from scripts.run_h19_candidate_sweep import build_amortization_curve


def _row(n, seconds, repeat=0):
    return {"candidate_positions": n, "verification_sweep_seconds": seconds,
           "repeat": repeat}


def test_reports_one_point_per_distinct_candidate_count():
    rows = [_row(1, 8.0), _row(8, 8.1), _row(64, 8.8)]
    curve = build_amortization_curve(rows)
    assert [point["candidate_positions"] for point in curve] == [1, 8, 64]


def test_points_are_sorted_ascending_regardless_of_row_order():
    rows = [_row(64, 8.8), _row(1, 8.0), _row(8, 8.1)]
    curve = build_amortization_curve(rows)
    assert [point["candidate_positions"] for point in curve] == [1, 8, 64]


def test_median_min_max_across_repeats():
    rows = [_row(8, 8.0, repeat=0), _row(8, 8.4, repeat=1), _row(8, 8.2, repeat=2)]
    curve = build_amortization_curve(rows)
    point = curve[0]
    assert point["median_seconds"] == pytest.approx(8.2)
    assert point["min_seconds"] == pytest.approx(8.0)
    assert point["max_seconds"] == pytest.approx(8.4)
    assert point["samples"] == 3


def test_relative_to_n1_is_one_at_the_baseline_itself():
    rows = [_row(1, 8.0), _row(128, 9.6)]
    curve = build_amortization_curve(rows)
    assert curve[0]["relative_to_n1"] == pytest.approx(1.0)


def test_relative_to_n1_reflects_a_cheap_amortization_regime():
    """The exact result H19 is looking for: candidate positions that cost
    barely more than N=1, which is what would justify moving on to tree-
    based speculation strategies at that budget."""
    rows = [_row(1, 8.0), _row(128, 9.6)]
    curve = build_amortization_curve(rows)
    assert curve[1]["relative_to_n1"] == pytest.approx(9.6 / 8.0)


def test_relative_to_n1_is_absent_when_n1_was_not_measured():
    """Without an N=1 baseline in the data, relative_to_n1 has nothing to
    divide by -- it must be omitted, not silently computed against the
    wrong reference point."""
    rows = [_row(8, 8.1), _row(64, 8.8)]
    curve = build_amortization_curve(rows)
    assert all("relative_to_n1" not in point for point in curve)


def test_empty_rows_returns_an_empty_curve_not_a_crash():
    assert build_amortization_curve([]) == []
