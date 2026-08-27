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
    pareto_frontier,
    paired_block_log_ratios,
    vram_regime,
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


# ---------------------------------------------------------------- vram_regime

def test_vram_regime_buckets_around_this_repos_own_configured_budgets():
    # exact-min's 1.80 GB and spec-fixed's 2.70 GB both belong in ~2 GB.
    assert vram_regime(1.80) == "~2 GB"
    assert vram_regime(2.70) == "~2 GB"
    # exact-resident's 4.00 GB belongs in ~4 GB.
    assert vram_regime(4.00) == "~4 GB"


def test_vram_regime_falls_through_to_other_above_the_second_ceiling():
    label = vram_regime(9.5)
    assert label.startswith("other")
    assert "9.5" in label


# ------------------------------------------------------------ pareto_frontier

def test_pareto_frontier_drops_a_point_beaten_on_both_axes():
    points = [
        {"method_id": "slow-and-big", "peak_vram_gb": 4.0, "seconds_per_token": 20.0},
        {"method_id": "fast-and-small", "peak_vram_gb": 2.0, "seconds_per_token": 10.0},
    ]
    frontier = pareto_frontier(points)
    assert [p["method_id"] for p in frontier] == ["fast-and-small"]


def test_pareto_frontier_keeps_a_genuine_tradeoff():
    """More VRAM buys real speed here, so both points belong on the
    frontier -- this is the case the whole function exists for."""
    points = [
        {"method_id": "cheap", "peak_vram_gb": 2.0, "seconds_per_token": 20.0},
        {"method_id": "pricier-but-faster", "peak_vram_gb": 4.0, "seconds_per_token": 10.0},
    ]
    frontier = pareto_frontier(points)
    assert {p["method_id"] for p in frontier} == {"cheap", "pricier-but-faster"}


def test_pareto_frontier_is_sorted_by_vram_ascending():
    points = [
        {"method_id": "big", "peak_vram_gb": 4.0, "seconds_per_token": 5.0},
        {"method_id": "small", "peak_vram_gb": 2.0, "seconds_per_token": 30.0},
    ]
    frontier = pareto_frontier(points)
    assert [p["method_id"] for p in frontier] == ["small", "big"]


def test_pareto_frontier_handles_an_exact_tie_without_dropping_both():
    """Two points with identical coordinates must not eliminate each other
    -- the domination test requires a strict improvement on at least one
    axis, not just "at least as good"."""
    points = [
        {"method_id": "a", "peak_vram_gb": 3.0, "seconds_per_token": 15.0},
        {"method_id": "b", "peak_vram_gb": 3.0, "seconds_per_token": 15.0},
    ]
    frontier = pareto_frontier(points)
    assert {p["method_id"] for p in frontier} == {"a", "b"}


def test_pareto_frontier_of_an_empty_list_is_empty():
    assert pareto_frontier([]) == []


# ------------------------------------------------------- _run_cell_in_subprocess

def test_run_cell_in_subprocess_returns_the_workers_written_json(tmp_path):
    """Uses a stand-in "worker" (any script matching the --config/--out
    contract) rather than the real GPU worker, so this exercises the
    subprocess plumbing -- launch, read stdout/stderr, parse the output
    file -- without CUDA or a real model."""
    import scripts.run_paper_comparison as rpc

    fake_worker = tmp_path / "fake_worker.py"
    fake_worker.write_text(
        "import argparse, json, pathlib\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--config', required=True)\n"
        "p.add_argument('--out', required=True)\n"
        "a = p.parse_args()\n"
        "config = json.loads(pathlib.Path(a.config).read_text())\n"
        "pathlib.Path(a.out).write_text(json.dumps({"
        "'rows': [{'case_id': config['method_id']}], 'metadata': {}, "
        "'peak_host_rss_bytes': 12345, 'error': None, 'traceback': None}))\n",
        encoding="utf-8")

    old_worker_script = rpc.WORKER_SCRIPT
    rpc.WORKER_SCRIPT = fake_worker
    try:
        result = rpc._run_cell_in_subprocess(
            {"method_id": "exact-min"}, tmp_path, timeout_s=30.0)
    finally:
        rpc.WORKER_SCRIPT = old_worker_script

    assert result["error"] is None
    assert result["rows"] == [{"case_id": "exact-min"}]
    assert result["peak_host_rss_bytes"] == 12345


def test_run_cell_in_subprocess_reports_a_timeout_as_a_failure_not_a_crash(tmp_path):
    import scripts.run_paper_comparison as rpc

    slow_worker = tmp_path / "slow_worker.py"
    slow_worker.write_text(
        "import time\n"
        "time.sleep(10)\n",
        encoding="utf-8")

    old_worker_script = rpc.WORKER_SCRIPT
    rpc.WORKER_SCRIPT = slow_worker
    try:
        result = rpc._run_cell_in_subprocess(
            {"method_id": "exact-min"}, tmp_path, timeout_s=0.2)
    finally:
        rpc.WORKER_SCRIPT = old_worker_script

    assert "timed out" in result["error"]
    assert result["rows"] == []


def test_run_cell_in_subprocess_reports_missing_output_as_a_failure(tmp_path):
    """A worker that crashes before writing --out (e.g. an import error)
    must not be mistaken for a cell that produced zero rows cleanly."""
    import scripts.run_paper_comparison as rpc

    crashing_worker = tmp_path / "crashing_worker.py"
    crashing_worker.write_text("raise SystemExit(3)\n", encoding="utf-8")

    old_worker_script = rpc.WORKER_SCRIPT
    rpc.WORKER_SCRIPT = crashing_worker
    try:
        result = rpc._run_cell_in_subprocess(
            {"method_id": "exact-min"}, tmp_path, timeout_s=30.0)
    finally:
        rpc.WORKER_SCRIPT = old_worker_script

    assert result["error"] is not None
    assert "no output" in result["error"]


def test_run_cell_in_subprocess_reports_malformed_json_as_a_failure(tmp_path):
    import scripts.run_paper_comparison as rpc

    garbage_worker = tmp_path / "garbage_worker.py"
    garbage_worker.write_text(
        "import argparse, pathlib\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--config', required=True)\n"
        "p.add_argument('--out', required=True)\n"
        "a = p.parse_args()\n"
        "pathlib.Path(a.out).write_text('not valid json{{{')\n",
        encoding="utf-8")

    old_worker_script = rpc.WORKER_SCRIPT
    rpc.WORKER_SCRIPT = garbage_worker
    try:
        result = rpc._run_cell_in_subprocess(
            {"method_id": "exact-min"}, tmp_path, timeout_s=30.0)
    finally:
        rpc.WORKER_SCRIPT = old_worker_script

    assert result["error"] is not None
    assert "not valid JSON" in result["error"]
