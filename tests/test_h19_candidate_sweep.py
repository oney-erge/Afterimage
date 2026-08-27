"""build_amortization_curve and find_knee are pure CPU functions -- no
CUDA, no engine -- so this suite verifies the reduction/gate logic
directly against synthetic per-cell rows rather than needing real
hardware measurements. What matters: curves stay separated per case
(never pooled across different prompt lengths), throughput/marginal-cost
columns are arithmetically correct, and find_knee implements the actual
G1 gate (largest N within a real overhead threshold of N=1), not a
threshold so loose it passes trivially.
"""
from __future__ import annotations

import pytest

from scripts.run_h19_candidate_sweep import build_amortization_curve, find_knee


def _rows(case_id: str, points: list[tuple[int, float]], repeats: int = 1) -> list[dict]:
    return [
        {"case_id": case_id, "candidate_positions": n, "verification_sweep_seconds": cost}
        for n, cost in points for _ in range(repeats)
    ]


def test_curves_are_kept_separate_per_case_not_pooled():
    """Two different-length prompts must never be averaged together --
    see the module's own reasoning on why absolute latency across prompts
    is not comparable."""
    rows = _rows("short", [(1, 1.0)]) + _rows("long", [(1, 5.0)])
    curves = build_amortization_curve(rows)
    assert set(curves) == {"short", "long"}
    assert curves["short"][0]["median_seconds"] == pytest.approx(1.0)
    assert curves["long"][0]["median_seconds"] == pytest.approx(5.0)


def test_relative_to_n1_uses_the_literal_n1_point_not_the_smallest_measured():
    """A sweep that never measured N=1 at all must not silently treat
    its smallest measured N as the baseline."""
    rows = _rows("c", [(2, 1.0), (4, 1.5)])  # no N=1 point
    curves = build_amortization_curve(rows)
    assert all("relative_to_n1" not in p for p in curves["c"])


def test_relative_to_n1_is_computed_correctly_when_baseline_exists():
    rows = _rows("c", [(1, 2.0), (4, 3.0)])
    curves = build_amortization_curve(rows)
    by_n = {p["candidate_positions"]: p for p in curves["c"]}
    assert by_n[1]["relative_to_n1"] == pytest.approx(1.0)
    assert by_n[4]["relative_to_n1"] == pytest.approx(1.5)


def test_throughput_is_candidates_over_median_seconds():
    rows = _rows("c", [(8, 2.0)])
    curves = build_amortization_curve(rows)
    point = curves["c"][0]
    assert point["throughput_candidates_per_second"] == pytest.approx(4.0)


def test_marginal_cost_compares_against_exactly_half_the_candidates():
    rows = _rows("c", [(4, 1.0), (8, 1.5)])
    curves = build_amortization_curve(rows)
    by_n = {p["candidate_positions"]: p for p in curves["c"]}
    assert by_n[8]["marginal_cost_seconds_vs_half"] == pytest.approx(0.5)


def test_marginal_cost_is_none_when_half_the_count_was_not_measured():
    rows = _rows("c", [(1, 1.0), (3, 1.2)])  # 3's half (1.5) was never measured
    curves = build_amortization_curve(rows)
    by_n = {p["candidate_positions"]: p for p in curves["c"]}
    assert by_n[3]["marginal_cost_seconds_vs_half"] is None
    # N=1 has no "half" at all (0 is not a valid candidate count)
    assert by_n[1]["marginal_cost_seconds_vs_half"] is None


def test_median_min_max_reflect_repeated_measurements():
    rows = _rows("c", [(1, 1.0), (1, 3.0), (1, 2.0)])
    curves = build_amortization_curve(rows)
    point = curves["c"][0]
    assert point["median_seconds"] == pytest.approx(2.0)
    assert point["min_seconds"] == pytest.approx(1.0)
    assert point["max_seconds"] == pytest.approx(3.0)
    assert point["samples"] == 3


# ------------------------------------------------------------------- find_knee

def test_find_knee_stops_at_the_last_point_within_threshold():
    curve = [
        {"candidate_positions": 1, "relative_to_n1": 1.0},
        {"candidate_positions": 8, "relative_to_n1": 1.05},
        {"candidate_positions": 16, "relative_to_n1": 1.09},
        {"candidate_positions": 32, "relative_to_n1": 1.30},
    ]
    assert find_knee(curve, overhead_threshold=1.10) == 16


def test_find_knee_is_not_a_trivially_passing_gate():
    """A knee that just returns the largest measured N regardless of
    actual overhead would make G1 unfalsifiable -- confirm a curve that
    genuinely grows past threshold early gets a small knee, not the
    largest N measured."""
    curve = [
        {"candidate_positions": 1, "relative_to_n1": 1.0},
        {"candidate_positions": 2, "relative_to_n1": 1.5},  # already over threshold
        {"candidate_positions": 1024, "relative_to_n1": 1.5},
    ]
    knee = find_knee(curve, overhead_threshold=1.10)
    assert knee == 1
    assert knee != 1024


def test_find_knee_returns_none_without_a_baseline():
    # No point in the curve has candidate_positions==1 at all -- there is
    # no N=1 baseline to compare anything against.
    assert find_knee([{"candidate_positions": 4, "relative_to_n1": 4.0}]) is None
    # A curve missing the "relative_to_n1" key entirely (build_
    # amortization_curve never computed one, e.g. because N=1 itself was
    # never measured) must also return None, not raise.
    assert find_knee([{"candidate_positions": 4}]) is None
