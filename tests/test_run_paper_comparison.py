"""paired_block_log_ratios() is the block-level statistic this project's own
methodology review recommended in place of pooling every (block, case) pair
as if it were an independent observation: one d_b per randomized block
rather than one point per case. These tests exercise it directly with
synthetic rows, without touching CUDA or any real model.
"""
from __future__ import annotations

import json
import math
import pathlib
import types

import pytest

from scripts.run_bounded_suite import METHODS
from scripts.run_paper_comparison import (
    CONTROL_METHOD,
    DEFAULT_METHODS,
    DEFAULT_TOKEN_LENGTHS,
    DEFAULT_TOKEN_LENGTHS_BY_SUITE,
    _shuffled_order,
    afterimage_plan_method,
    budget_label,
    budget_method_variants,
    completed_cells,
    derive_ttft_decode_metrics,
    pareto_frontier,
    paired_block_log_ratios,
    paper_eligibility,
    run_one_token_length,
    snapshot_afterimage_plan_methods,
    token_exactness,
    vram_regime,
    workload_for,
)


def _row(case_id: str, seconds_per_token: float, block: int) -> dict:
    return {"case_id": case_id, "seconds_per_token": seconds_per_token, "repeat": block}


def test_control_is_one_of_the_default_methods():
    """paired_block_log_ratios compares every other method against
    CONTROL_METHOD; the default method set must actually include it or the
    comparison silently produces nothing."""
    assert CONTROL_METHOD in DEFAULT_METHODS


def test_headline_defaults_use_two_runnable_external_offload_baselines():
    assert "accelerate" in DEFAULT_METHODS
    assert "deepspeed-zero-inference" in DEFAULT_METHODS
    assert "dfloat11" not in DEFAULT_METHODS


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


# -------------------------------------------------------- budget_method_variants

def test_budget_label_formats_without_trailing_zeros():
    assert budget_label(2.0) == "2"
    assert budget_label(2.5) == "2.5"
    assert budget_label(3.25) == "3.25"


def test_budget_method_variants_pins_afterimage_to_the_exact_budget():
    variants = budget_method_variants(2.0)
    assert variants["exact-2gb"].overrides["vram_budget_gb"] == 2.0
    assert variants["exact-2gb"].kind == "afterimage"


def test_budget_method_variants_pins_accelerate_to_the_exact_budget_in_mb():
    variants = budget_method_variants(2.0)
    assert variants["accelerate-2gb"].overrides["gpu_memory"] == "2048MB"
    assert variants["accelerate-2gb"].kind == "accelerate"


def test_budget_method_variants_preserves_every_other_override_from_the_base_method():
    """Only the budget knob changes -- decode_slice_elems, io_prefetch_depth
    etc. must carry over from exact-min/accelerate unchanged, or this
    would silently also be testing a different configuration, not just a
    different budget."""
    base_exact_overrides = dict(METHODS["exact-min"].overrides)
    variants = budget_method_variants(3.0)
    for key, value in base_exact_overrides.items():
        if key != "vram_budget_gb":
            assert variants["exact-3gb"].overrides[key] == value


def test_budget_method_variants_ids_do_not_collide_with_static_methods():
    variants = budget_method_variants(4.0)
    assert set(variants) & set(METHODS) == set()


def test_afterimage_plan_method_uses_plan_budgets_and_hash(tmp_path):
    plan_path = tmp_path / "h65.json"
    plan_path.write_text(json.dumps({
        "schema_version": 2,
        "feasible": True,
        "choices": {"model.weight": {"name": "decoded_ram"}},
        "vram_budget_bytes": 4_000_000_000,
        "ram_budget_bytes": 8_000_000_000,
        "predicted_prepare_s": 12.5,
    }), encoding="utf-8")

    method_id, method, provenance = afterimage_plan_method(
        "h65-selected=%s" % plan_path)

    assert method_id == "h65-selected"
    assert method.overrides["representation_policy"] == "multi_state"
    assert method.overrides["vram_budget_gb"] == 4.0
    assert method.overrides["ram_budget_gb"] == 8.0
    assert provenance["choice_count"] == 1
    assert len(provenance["source_sha256"]) == 64


def test_afterimage_plan_method_rejects_infeasible_plan(tmp_path):
    plan_path = tmp_path / "infeasible.json"
    plan_path.write_text(json.dumps({
        "schema_version": 2,
        "feasible": False,
        "choices": {"model.weight": {}},
        "vram_budget_bytes": 4_000_000_000,
        "ram_budget_bytes": 8_000_000_000,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="not feasible"):
        afterimage_plan_method("h65-selected=%s" % plan_path)


def test_afterimage_plan_method_accepts_an_explicit_legacy_policy(tmp_path):
    plan_path = tmp_path / "h6.json"
    plan_path.write_text(json.dumps({
        "schema_version": 2,
        "feasible": True,
        "choices": {"model.weight": {"name": "compressed_disk"}},
        "vram_budget_bytes": 4_000_000_000,
        "ram_budget_bytes": 8_000_000_000,
    }), encoding="utf-8")

    _method_id, method, provenance = afterimage_plan_method(
        "legacy-h6=per_tensor:%s" % plan_path)

    assert method.overrides["representation_policy"] == "per_tensor"
    assert provenance["representation_policy"] == "per_tensor"


def test_snapshot_afterimage_plan_method_is_content_addressed(tmp_path, monkeypatch):
    plan_path = tmp_path / "h65.json"
    plan_path.write_text(json.dumps({
        "schema_version": 2,
        "feasible": True,
        "choices": {"model.weight": {"name": "decoded_vram"}},
        "vram_budget_bytes": 4_000_000_000,
        "ram_budget_bytes": 8_000_000_000,
    }), encoding="utf-8")
    method_id, method, provenance = afterimage_plan_method(
        "h65-test-snapshot=%s" % plan_path)
    monkeypatch.setitem(METHODS, method_id, method)

    frozen = snapshot_afterimage_plan_methods(
        [provenance], tmp_path / "results", "qwen-smoke")

    snapshot = pathlib.Path(frozen[0]["snapshot_path"])
    assert snapshot.read_bytes() == plan_path.read_bytes()
    assert provenance["source_sha256"][:12] in snapshot.name
    assert METHODS[method_id].overrides["representation_plan_state"] == str(snapshot)


# -------------------------------------------------------------- workload_for

def test_default_token_lengths_include_a_literal_ttft_probe():
    assert 1 in DEFAULT_TOKEN_LENGTHS


def test_suite_defaults_do_not_force_short_factual_answers_to_decode_length():
    assert DEFAULT_TOKEN_LENGTHS_BY_SUITE["evaluation"] == (1, 4)
    assert DEFAULT_TOKEN_LENGTHS_BY_SUITE["paper_generation"] == (1, 32, 128)


def test_workload_for_labels_one_token_as_ttft_not_short_cold_start():
    """The exact correction this closes: a 4-token result must never be
    mistaken for TTFT. Only n_tokens=1 is a literal time-to-first-token
    measurement."""
    assert workload_for(1) == "ttft"
    assert workload_for(4) == "short_cold_start"
    assert workload_for(32) == "short_cold_start"
    assert workload_for(100) == "decode"
    assert workload_for(128) == "decode"


# ------------------------------------------------------ derive_ttft_decode_metrics

def _method_entry(method_id, rows):
    return {"method_id": method_id, "rows": rows}


def _wall_row(case_id, wall_seconds, block=0):
    return {"case_id": case_id, "repeat": block, "wall_seconds": wall_seconds}


def test_derive_ttft_decode_metrics_computes_the_marginal_decode_rate():
    # TTFT: 0.5s for token 1. Full 128-token run: 64.5s total.
    # Marginal decode rate = (128-1) tokens / (64.5-0.5)s = 127/64 = 1.984 tok/s.
    ttft_result = {"methods": [_method_entry("exact-min", [_wall_row("a", 0.5)])]}
    decode_result = {"max_new_tokens": 128,
                     "methods": [_method_entry("exact-min", [_wall_row("a", 64.5)])]}
    derived = derive_ttft_decode_metrics(ttft_result, decode_result)
    assert derived["exact-min"]["paired_observations"] == 1
    assert derived["exact-min"]["ttft_seconds_median"] == pytest.approx(0.5)
    assert derived["exact-min"]["decode_tokens_per_second_median"] == pytest.approx(
        127 / 64.0)


def test_derive_ttft_decode_metrics_requires_n_tokens_above_one():
    result = derive_ttft_decode_metrics({}, {"max_new_tokens": 1})
    assert "error" in result


def test_derive_ttft_decode_metrics_only_pairs_shared_block_case_keys():
    ttft_result = {"methods": [_method_entry("airllm", [_wall_row("a", 0.5, block=0)])]}
    decode_result = {"max_new_tokens": 32,
                     "methods": [_method_entry("airllm", [_wall_row("a", 16.0, block=1)])]}
    derived = derive_ttft_decode_metrics(ttft_result, decode_result)
    assert derived["airllm"]["paired_observations"] == 0
    assert derived["airllm"]["decode_tokens_per_second_median"] is None


def test_derive_ttft_decode_metrics_handles_a_method_missing_from_the_ttft_run():
    ttft_result = {"methods": []}
    decode_result = {"max_new_tokens": 32,
                     "methods": [_method_entry("airllm", [_wall_row("a", 16.0)])]}
    derived = derive_ttft_decode_metrics(ttft_result, decode_result)
    assert derived["airllm"]["paired_observations"] == 0


# ------------------------------------------------------------- token_exactness

def _exactness_row(case_id, token_ids, block=0):
    return {"case_id": case_id, "repeat": block, "output_token_ids": token_ids}


def test_token_exactness_reports_full_agreement():
    rows_by_method = {
        "exact-min": [_exactness_row("a", [1, 2, 3]), _exactness_row("b", [4, 5, 6])],
        "exact-resident": [_exactness_row("a", [1, 2, 3]), _exactness_row("b", [4, 5, 6])],
    }
    result = token_exactness(rows_by_method, "exact-min", "exact-resident")
    assert result == {"compared_sequences": 2, "matching_sequences": 2,
                      "all_tokens_identical": True, "first_mismatch": None}


def test_token_exactness_finds_the_first_differing_position_not_just_a_count():
    rows_by_method = {
        "exact-min": [_exactness_row("a", [1, 2, 3, 4])],
        "spec-fixed": [_exactness_row("a", [1, 2, 9, 4])],
    }
    result = token_exactness(rows_by_method, "exact-min", "spec-fixed")
    assert result["all_tokens_identical"] is False
    assert result["first_mismatch"]["position"] == 2
    assert result["first_mismatch"]["case_id"] == "a"


def test_token_exactness_reports_only_the_first_mismatch_not_every_one():
    rows_by_method = {
        "exact-min": [_exactness_row("a", [1, 2]), _exactness_row("b", [3, 4])],
        "spec-fixed": [_exactness_row("a", [1, 9]), _exactness_row("b", [3, 9])],
    }
    result = token_exactness(rows_by_method, "exact-min", "spec-fixed")
    assert result["matching_sequences"] == 0
    assert result["first_mismatch"]["case_id"] == "a"  # not "b"


def test_token_exactness_only_compares_shared_block_case_keys():
    rows_by_method = {
        "exact-min": [_exactness_row("a", [1, 2], block=0)],
        "spec-fixed": [_exactness_row("a", [1, 2], block=1)],  # different block
    }
    result = token_exactness(rows_by_method, "exact-min", "spec-fixed")
    assert result["compared_sequences"] == 0
    assert result["all_tokens_identical"] is False


def test_token_exactness_with_no_rows_at_all_does_not_crash():
    result = token_exactness({}, "exact-min", "spec-fixed")
    assert result["compared_sequences"] == 0
    assert result["all_tokens_identical"] is False


# ----------------------------------------------- completed_cells / paper_eligibility

def test_completed_cells_only_counts_error_free_cells():
    result = {"cells": [
        {"block": 0, "method": "airllm", "error": None},
        {"block": 0, "method": "exact-min", "error": "boom"},
        {"block": 1, "method": "airllm", "error": None},
    ]}
    assert completed_cells(result) == {(0, "airllm"), (1, "airllm")}


def test_paper_eligibility_true_when_every_required_cell_succeeded():
    result = {"cells": [
        {"block": b, "method": m, "error": None}
        for b in range(2) for m in ("airllm", "exact-min")]}
    eligible, reason = paper_eligibility(result, blocks=2, selected=["airllm", "exact-min"])
    assert eligible is True
    assert "complete" in reason


def test_paper_eligibility_false_and_names_what_is_missing():
    result = {"cells": [{"block": 0, "method": "airllm", "error": None}]}
    eligible, reason = paper_eligibility(result, blocks=2, selected=["airllm", "exact-min"])
    assert eligible is False
    assert "missing 3 of 4" in reason


def test_paper_eligibility_catches_a_block_that_never_started_at_all():
    """A block cut short by the time budget before it began never appears
    in method_order_per_block or cells at all -- eligibility must still
    catch it via the required range(blocks) x selected set, not just by
    looking for failure records."""
    result = {"cells": []}
    eligible, _reason = paper_eligibility(result, blocks=3, selected=["airllm"])
    assert eligible is False


def test_capacity_failure_is_accounted_but_not_headline_paper_eligible():
    result = {"cells": [{
        "block": 0, "method": "dfloat11", "error": "CUDA out of memory",
        "metadata": {"capacity_failure": True},
    }]}
    eligible, reason = paper_eligibility(result, blocks=1, selected=["dfloat11"])
    assert eligible is False
    assert "missing 1 of 1" in reason


# ------------------------------------------------- run_one_token_length (integration)

def _fake_args(**overrides):
    base = dict(
        model="Qwen/Qwen3-14B", dfloat11_model="DFloat11/Qwen3-14B-DF11",
        draft_model="Qwen/Qwen3-0.6B", store="/tmp/store", blocks=2,
        warmup_tokens=0, cooldown_seconds=0.0, cooldown_max_temp_c=None,
        time_budget_minutes_per_length=60.0, seed=0, resume=False,
        require_complete=False, require_thermally_clean=False,
        prompt_suite="evaluation",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _fake_rendered(case_ids=("a", "b")):
    return [{"case": types.SimpleNamespace(id=cid), "prompt": "prompt-%s" % cid,
            "input_tokens": 5, "tokenizer": None} for cid in case_ids]


def _cell_row(method_id, case_id, block, seconds_per_token=10.0, tokens=4):
    return {"case_id": case_id, "method": method_id, "repeat": block,
           "output_tokens": tokens, "wall_seconds": seconds_per_token * tokens,
           "seconds_per_token": seconds_per_token, "peak_vram_gb": 2.0,
           "expected_match": True, "cache_drop_succeeded": True,
           "output_token_ids": [1, 2, 3, 4][:tokens]}


@pytest.fixture
def stub_environment_manifest(monkeypatch):
    import scripts.run_paper_comparison as rpc
    monkeypatch.setattr(rpc, "environment_manifest", lambda *a, **k: {"stub": True})
    monkeypatch.setattr(
        rpc, "cool_down",
        lambda *a, **k: {"cooldown_reached_target": True,
                         "throttled_after_cooldown": False})


class TestRunOneTokenLengthIntegration:
    def test_a_fully_successful_run_is_paper_eligible_and_writes_the_file(
            self, tmp_path, monkeypatch, stub_environment_manifest):
        import scripts.run_paper_comparison as rpc

        def fake_cell(config, work_dir, timeout_s):
            return {"rows": [_cell_row(config["method_id"], cid, config["block"])
                             for cid in ("a", "b")],
                   "metadata": {"initialization_seconds": 1.5},
                   "peak_host_rss_bytes": 1_000_000, "thermal_monitoring": None,
                   "error": None}

        monkeypatch.setattr(rpc, "_run_cell_in_subprocess", fake_cell)
        args = _fake_args()
        out_path = tmp_path / "result.json"
        result = run_one_token_length(
            args, tokenizer=None, rendered=_fake_rendered(),
            selected=["exact-min", "airllm"], out_path=out_path,
            repo_root=tmp_path, n_tokens=4, dirty=None, work_dir=tmp_path)

        assert out_path.exists()
        assert result["paper_eligible"] is True
        assert result["status"] == "complete"
        assert not out_path.with_suffix(".json.partial").exists()

    def test_require_complete_withholds_the_file_when_a_method_always_fails(
            self, tmp_path, monkeypatch, stub_environment_manifest):
        import scripts.run_paper_comparison as rpc

        def fake_cell(config, work_dir, timeout_s):
            if config["method_id"] == "airllm":
                return {"rows": [], "metadata": {}, "peak_host_rss_bytes": None,
                       "thermal_monitoring": None, "error": "simulated failure"}
            return {"rows": [_cell_row(config["method_id"], cid, config["block"])
                             for cid in ("a", "b")],
                   "metadata": {}, "peak_host_rss_bytes": None,
                   "thermal_monitoring": None, "error": None}

        monkeypatch.setattr(rpc, "_run_cell_in_subprocess", fake_cell)
        args = _fake_args(require_complete=True)
        out_path = tmp_path / "result.json"
        result = run_one_token_length(
            args, tokenizer=None, rendered=_fake_rendered(),
            selected=["exact-min", "airllm"], out_path=out_path,
            repo_root=tmp_path, n_tokens=4, dirty=None, work_dir=tmp_path)

        assert not out_path.exists()
        assert result["paper_eligible"] is False
        partial = out_path.with_suffix(out_path.suffix + ".partial")
        assert partial.exists()

    def test_resume_skips_already_succeeded_cells_and_only_retries_the_failure(
            self, tmp_path, monkeypatch, stub_environment_manifest):
        import scripts.run_paper_comparison as rpc

        dispatched = []

        def failing_airllm_cell(config, work_dir, timeout_s):
            dispatched.append((config["block"], config["method_id"]))
            if config["method_id"] == "airllm":
                return {"rows": [], "metadata": {}, "peak_host_rss_bytes": None,
                       "thermal_monitoring": None, "error": "simulated failure"}
            return {"rows": [_cell_row(config["method_id"], cid, config["block"])
                             for cid in ("a", "b")],
                   "metadata": {}, "peak_host_rss_bytes": None,
                   "thermal_monitoring": None, "error": None}

        monkeypatch.setattr(rpc, "_run_cell_in_subprocess", failing_airllm_cell)
        out_path = tmp_path / "result.json"
        first_args = _fake_args(require_complete=True)
        run_one_token_length(
            first_args, tokenizer=None, rendered=_fake_rendered(),
            selected=["exact-min", "airllm"], out_path=out_path,
            repo_root=tmp_path, n_tokens=4, dirty=None, work_dir=tmp_path)
        assert not out_path.exists()
        first_dispatch_count = len(dispatched)

        def now_succeeding_cell(config, work_dir, timeout_s):
            dispatched.append((config["block"], config["method_id"]))
            return {"rows": [_cell_row(config["method_id"], cid, config["block"])
                             for cid in ("a", "b")],
                   "metadata": {}, "peak_host_rss_bytes": None,
                   "thermal_monitoring": None, "error": None}

        monkeypatch.setattr(rpc, "_run_cell_in_subprocess", now_succeeding_cell)
        resume_args = _fake_args(require_complete=True, resume=True)
        result = run_one_token_length(
            resume_args, tokenizer=None, rendered=_fake_rendered(),
            selected=["exact-min", "airllm"], out_path=out_path,
            repo_root=tmp_path, n_tokens=4, dirty=None, work_dir=tmp_path)

        assert out_path.exists()
        assert result["paper_eligible"] is True
        # Only the previously-failed airllm cells were redispatched, not the
        # exact-min cells that already succeeded the first time.
        second_run_dispatches = dispatched[first_dispatch_count:]
        assert all(method_id == "airllm" for _block, method_id in second_run_dispatches)
        assert len(second_run_dispatches) == first_args.blocks

    def test_resume_without_a_partial_file_behaves_like_a_normal_first_run(
            self, tmp_path, monkeypatch, stub_environment_manifest):
        import scripts.run_paper_comparison as rpc

        def fake_cell(config, work_dir, timeout_s):
            return {"rows": [_cell_row(config["method_id"], cid, config["block"])
                             for cid in ("a", "b")],
                   "metadata": {}, "peak_host_rss_bytes": None,
                   "thermal_monitoring": None, "error": None}

        monkeypatch.setattr(rpc, "_run_cell_in_subprocess", fake_cell)
        args = _fake_args(resume=True)
        out_path = tmp_path / "result.json"
        result = run_one_token_length(
            args, tokenizer=None, rendered=_fake_rendered(),
            selected=["exact-min"], out_path=out_path, repo_root=tmp_path,
            n_tokens=4, dirty=None, work_dir=tmp_path)
        assert out_path.exists()
        assert result["paper_eligible"] is True

    def test_resume_with_mismatched_settings_refuses_rather_than_silently_merging(
            self, tmp_path, monkeypatch, stub_environment_manifest):
        import scripts.run_paper_comparison as rpc

        def fake_cell(config, work_dir, timeout_s):
            return {"rows": [], "metadata": {}, "peak_host_rss_bytes": None,
                   "thermal_monitoring": None, "error": "simulated failure"}

        monkeypatch.setattr(rpc, "_run_cell_in_subprocess", fake_cell)
        out_path = tmp_path / "result.json"
        run_one_token_length(
            _fake_args(require_complete=True), tokenizer=None,
            rendered=_fake_rendered(), selected=["exact-min"], out_path=out_path,
            repo_root=tmp_path, n_tokens=4, dirty=None, work_dir=tmp_path)

        with pytest.raises(ValueError, match="different settings"):
            run_one_token_length(
                _fake_args(require_complete=True, resume=True, blocks=99),
                tokenizer=None, rendered=_fake_rendered(), selected=["exact-min"],
                out_path=out_path, repo_root=tmp_path, n_tokens=4, dirty=None,
                work_dir=tmp_path)

    def test_metadata_and_thermal_data_are_preserved_in_the_final_report(
            self, tmp_path, monkeypatch, stub_environment_manifest):
        """The exact bug this fix closes: the worker always returned rich
        per-cell metadata and thermal readings, but the orchestrator used
        to drop all of it before writing the final result."""
        import scripts.run_paper_comparison as rpc

        def fake_cell(config, work_dir, timeout_s):
            return {"rows": [_cell_row(config["method_id"], "a", config["block"])],
                   "metadata": {"initialization_seconds": 4.2},
                   "peak_host_rss_bytes": 500_000_000,
                    "thermal_monitoring": {
                        "samples_collected": 5, "sm_clock_mhz_min": 1200.0,
                        "sm_clock_mhz_median": 1800.0, "temperature_c_max": 72.0,
                        "any_throttle_during_measurement": False,
                        "any_thermal_throttle_during_measurement": False,
                        "any_power_limit_during_measurement": True},
                   "error": None}

        monkeypatch.setattr(rpc, "_run_cell_in_subprocess", fake_cell)
        out_path = tmp_path / "result.json"
        result = run_one_token_length(
            _fake_args(blocks=1), tokenizer=None, rendered=_fake_rendered(("a",)),
            selected=["exact-min"], out_path=out_path, repo_root=tmp_path,
            n_tokens=4, dirty=None, work_dir=tmp_path)

        entry = result["methods"][0]
        assert entry["initialization_seconds_median"] == 4.2
        assert entry["peak_host_rss_bytes"] == 500_000_000
        assert entry["thermal_across_all_cells"]["sm_clock_mhz_min"] == 1200.0
        assert entry["thermal_across_all_cells"]["temperature_c_max"] == 72.0
        assert entry["thermal_across_all_cells"]["any_throttle_during_measurement"] is False
        assert entry["thermal_across_all_cells"]["any_thermal_throttle_during_measurement"] is False
        assert entry["thermal_across_all_cells"]["any_power_limit_during_measurement"] is True
