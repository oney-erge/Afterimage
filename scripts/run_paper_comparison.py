#!/usr/bin/env python3
"""Randomized-block comparison for the paper's headline claim:
Afterimage vs AirLLM vs Hugging Face Accelerate vs DeepSpeed ZeRO-Inference,
on identical prompts and hardware, at multiple output-token lengths, with
real inter-method randomization instead of a fixed method order.

Why this exists instead of just raising --repeats on run_bounded_suite.py's
canonical driver
--------------------------------------------------------------------------
The canonical driver (benchmark.sh -> run_bounded_suite.py) loads one
method, sweeps every case for every repeat, then moves to the next method
-- so "repeat" measures noise *within* a method's own run, but never
untangles method quality from *when in the campaign* a method happened to
run. A GPU that heats up over a multi-hour campaign makes later methods
look slower purely from run order, and the fix cannot be "measure order
effects away with more repeats of the same order" -- the order itself has
to vary. This script instead makes one full sweep of ONE randomly-ordered
method its unit of replication (a "block"): every block gives every
selected method a turn, in a freshly shuffled order, each starting from a
freshly loaded model and an explicit untimed warm-up. Reported dispersion
is therefore across blocks, matching the "d_b = median_p[log(control) -
log(candidate)] per independent randomized block" statistic this project's
own methodology review recommended over pooling every (block, case) pair
as if it were an independent observation.

Every (block, method) cell runs in its own fresh subprocess
--------------------------------------------------------------------------
See run_paper_comparison_worker.py's module docstring for the full
rationale. In short: a resident draft model (spec-fixed), DFloat11's
custom CUDA kernels, or AirLLM's own internal state are not guaranteed to
release GPU/allocator state when a method finishes in-process, and Python's
allocator does not reliably return freed host pages to the OS either -- so
"peak VRAM"/"peak host RAM" measured mid-process for the Nth method in a
block is contaminated by whatever the first N-1 methods left behind. A
fresh OS process per cell is the only isolation guarantee for both.

This is a real engineering trade-off, not a free upgrade: block-major,
subprocess-isolated execution reloads every method's model once per
(block, method) cell instead of once for the whole run (5-6 methods x
--blocks reloads instead of 5-6, plus one interpreter start each), which is
why --blocks defaults to 3 ("each pass 3 times") rather than the 8-12
blocks a confirmatory paper claim should eventually use -- raise --blocks
for that once a pilot run's block-to-block variance is known.

Requires (WSL2/Linux only, matching run_bounded_suite.py and benchmark.sh):
CUDA, a prepared Afterimage store for --model, and the optional-dependency
group `bench` installed (`pip install -e .[bench]`). Missing packages fail
preflight before the first multi-hour cell rather than creating a partial
comparison that cannot satisfy --require-complete.

Usage:
    python scripts/run_paper_comparison.py \\
        --store /root/afterimage/store_14b \\
        --out-dir results/paper-comparison

Or via the one-click wrapper: ./paper_benchmark.sh
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import random
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from afterimage.bench.prompt_suite import PROMPT_SUITE_VERSION, prompt_cases
from scripts.run_bounded_suite import (
    METHODS,
    aggregate,
    checkpoint,
    command_output,
    cool_down,
    environment_manifest,
    load_tokenizer,
    log,
    render_cases,
)
from scripts import run_bounded_suite as bounded

WORKER_SCRIPT = pathlib.Path(__file__).resolve().parent / "run_paper_comparison_worker.py"

# Headline methods must produce timings on the reference host. DFloat11 is
# retained as an opt-in capacity appendix because its Qwen3-14B checkpoint
# cannot initialize on the 8 GB reference GPU. A predictable OOM is useful
# applicability evidence, but it is not a performance baseline and must not
# occupy an empty row in the headline table. Accelerate and DeepSpeed are the
# two runnable external offload baselines in addition to AirLLM.
DEFAULT_METHODS = ("airllm", "accelerate", "deepspeed-zero-inference",
                   "exact-min", "exact-resident", "spec-fixed")
# 1 is a genuine TTFT probe (see workload_for()); 128 is long enough to
# exercise a full k=8 fixed-speculation chain repeatedly, which 4 never
# could (speculative_decoding's own k_request is capped at
# max_new_tokens - n_generated -- see streaming_engine.py). Do not report
# the 4-token pass as TTFT: it measures short cold-start latency, a
# different and still-useful thing, but not literal time-to-first-token.
DEFAULT_TOKEN_LENGTHS_BY_SUITE = {
    "evaluation": (1, 4),
    "paper_generation": (1, 32, 128),
}
# Backwards-compatible public constant used by tests and external scripts.
# The CLI selects the suite-specific value above when --token-lengths is not
# supplied, so it never forces a factual one-word prompt to 128 tokens.
DEFAULT_TOKEN_LENGTHS = DEFAULT_TOKEN_LENGTHS_BY_SUITE["evaluation"]


def workload_for(n_tokens: int) -> str:
    """Labels which of the paper's three distinct workloads a token length
    belongs to, so a result file is never ambiguous about what it measured
    -- in particular so a 4-token cold-start pass is never mistaken for (or
    mislabeled as) a literal TTFT measurement, which only n_tokens=1 is.
    """
    if n_tokens == 1:
        return "ttft"
    if n_tokens >= 100:
        return "decode"
    return "short_cold_start"


def budget_label(budget_gb: float) -> str:
    return ("%.2f" % budget_gb).rstrip("0").rstrip(".")


def budget_method_variants(budget_gb: float) -> dict[str, object]:
    """Synthesizes method variants pinned to exactly ``budget_gb`` of VRAM,
    for every system in METHODS that exposes a single, direct memory-
    budget config knob -- Afterimage's own vram_budget_gb, and
    Accelerate's gpu_memory string.

    This is "the config at that budget", not a search over every other
    free parameter (decode_slice_elems, io_prefetch_depth, ...) at that
    budget: Afterimage and Accelerate each already expose their budget as
    one direct dial with no other tunable knob in METHODS' current
    overrides, so "the fastest valid configuration at this budget" and
    "the configuration at this budget" coincide today. A real per-budget
    hyperparameter search over the other knobs is real future work, noted
    here rather than silently assumed equivalent.

    AirLLM and DFloat11 are not included: neither exposes a comparable
    direct VRAM-budget dial in this project's integration (see run_airllm/
    run_dfloat11 in run_bounded_suite.py) -- they contribute whatever
    peak_vram_gb they naturally land on, exactly as the plan's own "AirLLM
    and DFloat11 can contribute their naturally achieved memory points"
    describes.
    """
    label = budget_label(budget_gb)
    variants = {}
    exact = METHODS["exact-min"]
    exact_id = "exact-%sgb" % label
    variants[exact_id] = dataclasses.replace(
        exact, id=exact_id, title="Afterimage exact streaming at %s GB" % label,
        overrides={**exact.overrides, "vram_budget_gb": budget_gb})
    accelerate = METHODS["accelerate"]
    accelerate_id = "accelerate-%sgb" % label
    variants[accelerate_id] = dataclasses.replace(
        accelerate, id=accelerate_id,
        title="Hugging Face Accelerate at %s GB" % label,
        overrides={**accelerate.overrides,
                  "gpu_memory": "%dMB" % round(budget_gb * 1024)})
    return variants

# The Afterimage method every other method is compared against. exact-min
# is the "reference_execution_equivalent" control -- greedy-exact, no
# speculation, no residency heuristics -- so a speedup measured against it
# isolates what a given method or mechanism contributes, not a second
# confounding Afterimage feature.
CONTROL_METHOD = "exact-min"

DEPENDENCY_PACKAGE = {"airllm": "airllm", "accelerate": "accelerate",
                      "dfloat11": "dfloat11", "dfloat11-gpu-resident": "dfloat11",
                      "deepspeed-zero-inference": "deepspeed"}

# Boundaries for vram_regime() sit at the midpoints between this repo's own
# configured Afterimage budgets: exact-min 1.80 GB, spec-fixed 2.70 GB,
# exact-resident 4.00 GB (see METHODS in run_bounded_suite.py) -- not
# arbitrary round numbers.
_VRAM_REGIME_2GB_CEILING = 3.35
_VRAM_REGIME_4GB_CEILING = 6.0


def _shuffled_order(methods: list[str], rng: random.Random) -> list[str]:
    order = list(methods)
    rng.shuffle(order)
    return order


def vram_regime(peak_vram_gb: float) -> str:
    """Buckets a measured peak VRAM figure into one of this project's two
    standard low-memory comparison regimes, so results at genuinely
    different memory budgets are never presented in one table as though
    memory were held equal. A method's *configured* budget and its
    *measured* peak_vram_gb can differ (AirLLM/Accelerate/DFloat11 are not
    configured to an Afterimage-style budget at all), so this buckets the
    measured number, which is what a reader actually cares about.
    """
    if peak_vram_gb <= _VRAM_REGIME_2GB_CEILING:
        return "~2 GB"
    if peak_vram_gb <= _VRAM_REGIME_4GB_CEILING:
        return "~4 GB"
    return "other (%.2f GB)" % peak_vram_gb


def pareto_frontier(points: list[dict]) -> list[dict]:
    """The Pareto-optimal subset of (peak_vram_gb, seconds_per_token)
    points -- lower is better on both axes. A point is dropped only if some
    other point is at least as good on both axes and strictly better on at
    least one. This is the honest way to compare methods that were never
    configured to the same memory budget (see vram_regime): a method at
    higher VRAM only belongs in the frontier if nothing cheaper is also at
    least as fast, and a method at lower VRAM only belongs if nothing
    equally cheap beats its time.

    Each point must carry "peak_vram_gb" and "seconds_per_token"; every
    other key is passed through unchanged.
    """
    frontier = []
    for i, point in enumerate(points):
        dominated = any(
            j != i
            and other["peak_vram_gb"] <= point["peak_vram_gb"]
            and other["seconds_per_token"] <= point["seconds_per_token"]
            and (other["peak_vram_gb"] < point["peak_vram_gb"]
                 or other["seconds_per_token"] < point["seconds_per_token"])
            for j, other in enumerate(points))
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda p: p["peak_vram_gb"])


def derive_ttft_decode_metrics(ttft_result: dict, decode_result: dict) -> dict:
    """Derives a decode-only tokens/sec estimate per method from a paired
    (1-token, N-token) measurement at the same (block, case), rather than
    instrumenting per-token callbacks inside four different libraries that
    do not share one streaming API (AirLLM/Accelerate/DFloat11 vs.
    Afterimage's own generate_adaptive(on_token=...)): measure T(1) and
    T(N) as two separate campaigns (workload_for() labels them "ttft" and
    "decode") and derive the marginal decode rate as
    (N - 1) / (wall_N - wall_1). This is the standard TTFT/TPOT
    decomposition used across current inference-serving literature,
    applied here as two matched full runs instead of one instrumented one.

    ``ttft_result``/``decode_result`` are the loaded JSON result dicts
    written by run_one_token_length() for max_new_tokens=1 and
    max_new_tokens>=100 respectively.
    """
    n_decode = decode_result.get("max_new_tokens")
    if not n_decode or n_decode <= 1:
        return {"error": "decode_result's max_new_tokens must be > 1"}

    def rows_by_key(result):
        return {entry["method_id"]: {(row["repeat"], row["case_id"]): row["wall_seconds"]
                                     for row in entry["rows"]}
               for entry in result.get("methods", [])}

    ttft_by_method = rows_by_key(ttft_result)
    decode_by_method = rows_by_key(decode_result)
    out = {}
    for method_id, decode_by_key_map in decode_by_method.items():
        ttft_by_key_map = ttft_by_method.get(method_id, {})
        shared = sorted(set(decode_by_key_map) & set(ttft_by_key_map))
        if not shared:
            out[method_id] = {"paired_observations": 0, "ttft_seconds_median": None,
                              "decode_tokens_per_second_median": None}
            continue
        ttft_values = [ttft_by_key_map[key] for key in shared]
        decode_tps_values = [
            (n_decode - 1) / (decode_by_key_map[key] - ttft_by_key_map[key])
            for key in shared if decode_by_key_map[key] > ttft_by_key_map[key]]
        out[method_id] = {
            "paired_observations": len(shared),
            "ttft_seconds_median": statistics.median(ttft_values),
            "decode_tokens_per_second_median": (
                statistics.median(decode_tps_values) if decode_tps_values else None),
        }
    return out


def paired_block_log_ratios(rows_by_method: dict[str, list[dict]],
                            control_id: str, candidate_id: str) -> dict:
    """The block-level paired statistic this project's own methodology
    review recommended in place of pooling every (block, case) pair as if
    it were an independent observation: one d_b per randomized block,

        d_b = median_p [ log(seconds_per_token_control) -
                         log(seconds_per_token_candidate) ]

    over cases p shared between control and candidate within block b.
    Positive d_b means the candidate was faster than the control in that
    block. Blocks are the independent replication unit here (each is a
    fresh model load, fresh warm-up, and a randomly-drawn position in that
    block's method order); cases give within-block workload diversity, not
    additional independence -- so inference belongs over d_1..d_B, not over
    every individual case timing.
    """
    control_by_block: dict[int, dict[str, float]] = {}
    for row in rows_by_method.get(control_id, []):
        control_by_block.setdefault(row["repeat"], {})[row["case_id"]] = (
            row["seconds_per_token"])
    candidate_by_block: dict[int, dict[str, float]] = {}
    for row in rows_by_method.get(candidate_id, []):
        candidate_by_block.setdefault(row["repeat"], {})[row["case_id"]] = (
            row["seconds_per_token"])
    block_ds = []
    for block in sorted(set(control_by_block) & set(candidate_by_block)):
        shared = sorted(set(control_by_block[block]) & set(candidate_by_block[block]))
        if not shared:
            continue
        diffs = [math.log(control_by_block[block][case]) -
                 math.log(candidate_by_block[block][case]) for case in shared]
        block_ds.append(statistics.median(diffs))
    if not block_ds:
        return {"blocks_compared": 0}
    result = {
        "blocks_compared": len(block_ds),
        "block_log_ratios": block_ds,
        "median_log_ratio": statistics.median(block_ds),
        "median_candidate_speedup_vs_control": math.exp(statistics.median(block_ds)),
    }
    if len(block_ds) >= 2:
        result["log_ratio_stdev_across_blocks"] = statistics.stdev(block_ds)
    # Deliberately not a confidence interval: the honest caveat this
    # project's own protocol exists to enforce (see _repeat_dispersion's
    # docstring in run_bounded_suite.py) applies here too, more so with
    # --blocks 3. Report the raw per-block values so a reader can judge for
    # themselves rather than trusting a summary that implies more
    # replication than actually happened.
    return result


def token_exactness(rows_by_method: dict[str, list[dict]],
                    control_id: str, candidate_id: str) -> dict:
    """Whether the candidate's generated token IDs exactly match the
    control's, case by case, at every shared (block, case_id).

    A raw match-count alone hides whether two runs diverged at token 1
    (a real correctness bug) or token 100 (plausibly late sampling/kernel
    nondeterminism); first_mismatch reports the earliest differing
    position across every mismatching pair found, which is what actually
    matters for debugging a regression.

    Only Afterimage's own alternate methods (exact-resident, spec-fixed)
    carry a hard exactness *contract* against exact-min -- see each
    Method's declared ``exactness`` in run_bounded_suite.py, which is the
    thing to check before treating a mismatch here as a bug rather than a
    diagnostic curiosity. AirLLM/Accelerate/DFloat11 are independent
    engines; bf16 arithmetic is not strictly associative, so even "greedy"
    decoding across genuinely different kernels/libraries can legitimately
    diverge over a long enough generation without either implementation
    being wrong.
    """
    control_by_key = {(row["repeat"], row["case_id"]): row
                      for row in rows_by_method.get(control_id, [])}
    compared = 0
    matches = 0
    first_mismatch = None
    for row in rows_by_method.get(candidate_id, []):
        key = (row["repeat"], row["case_id"])
        control_row = control_by_key.get(key)
        if control_row is None:
            continue
        compared += 1
        control_ids = control_row["output_token_ids"]
        candidate_ids = row["output_token_ids"]
        if control_ids == candidate_ids:
            matches += 1
        elif first_mismatch is None:
            position = next(
                (i for i, (a, b) in enumerate(zip(control_ids, candidate_ids)) if a != b),
                min(len(control_ids), len(candidate_ids)))
            first_mismatch = {"block": key[0], "case_id": key[1], "position": position,
                              "control_length": len(control_ids),
                              "candidate_length": len(candidate_ids)}
    return {
        "compared_sequences": compared,
        "matching_sequences": matches,
        "all_tokens_identical": compared > 0 and matches == compared,
        "first_mismatch": first_mismatch,
    }


def _run_cell_in_subprocess(config: dict, work_dir: pathlib.Path,
                            timeout_s: float) -> dict:
    """Runs exactly one (block, method) cell in a fresh subprocess. See
    run_paper_comparison_worker.py's module docstring for why: isolating
    GPU/allocator state and host-RAM high-water marks between methods is
    not achievable from inside one long-lived process, no matter how
    carefully each method's own code calls del/gc.collect()/empty_cache().
    """
    config_path = work_dir / "cell_config.json"
    out_path = work_dir / "cell_result.json"
    out_path.unlink(missing_ok=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER_SCRIPT), "--config", str(config_path),
             "--out", str(out_path)],
            cwd=str(pathlib.Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"rows": [], "metadata": {}, "peak_host_rss_bytes": None,
               "error": "worker timed out after %.1fs" % timeout_s,
               "traceback": repr(exc)}
    if proc.stdout:
        log(proc.stdout.rstrip())
    if proc.stderr:
        log(proc.stderr.rstrip())
    if not out_path.exists():
        return {"rows": [], "metadata": {}, "peak_host_rss_bytes": None,
               "error": "worker produced no output (exit %d)" % proc.returncode,
               "traceback": proc.stderr or None}
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"rows": [], "metadata": {}, "peak_host_rss_bytes": None,
               "error": "worker output was not valid JSON: %r" % exc,
               "traceback": traceback.format_exc()}


def completed_cells(result: dict) -> set[tuple[int, str]]:
    """(block, method_id) pairs that already have a successful (error-free)
    cell recorded -- what --resume treats as done and skips."""
    return {(cell["block"], cell["method"]) for cell in result.get("cells", [])
           if cell.get("error") is None}


def capacity_failed_cells(result: dict) -> set[tuple[int, str]]:
    """(block, method_id) pairs where the method predictably could not
    initialize within the available hardware (see run_paper_comparison_
    worker.is_capacity_failure) -- a real, reportable OUTCOME, not missing
    data. A method that deterministically cannot fit a VRAM budget will
    fail every block the same way; treating that as "still needed" would
    make --resume retry a doomed OOM forever. It still does not make the
    performance matrix paper-eligible; run capacity studies without
    --require-complete and keep them outside the headline table.
    """
    return {(cell["block"], cell["method"]) for cell in result.get("cells", [])
           if cell.get("error") is not None
           and isinstance(cell.get("metadata"), dict)
           and cell["metadata"].get("capacity_failure")}


def paper_eligibility(result: dict, blocks: int, selected: list[str]) -> tuple[bool, str]:
    """Whether every requested (block, method) cell for this token length
    produced rows. Capacity failures remain explicit outcomes in ``methods``
    but do not make a performance matrix paper-eligible: the headline table
    must use runnable alternatives instead of presenting an empty "failed"
    competitor row. A paper claim built on a matrix with silent gaps (a method
    that failed every block for an unexplained reason, a block cut short
    by the time budget) is not a claim about what the flags requested, it
    is a claim about whatever happened to finish. required is exactly
    range(blocks) x selected, not "whatever showed up in
    method_order_per_block", so a block that never started at all (time
    budget exhausted before it began) is caught too, not just cells that
    started and then failed.
    """
    required = {(block, method_id) for block in range(blocks) for method_id in selected}
    have = completed_cells(result)
    missing = sorted(required - have)
    if not missing:
        return True, "complete: every requested (block, method) cell succeeded"
    preview = ", ".join("block %d/%s" % pair for pair in missing[:5])
    more = " (+%d more)" % (len(missing) - 5) if len(missing) > 5 else ""
    return False, "missing %d of %d required cells: %s%s" % (
        len(missing), len(required), preview, more)


def run_one_token_length(args, tokenizer, rendered: list[dict],
                         selected: list[str], out_path: pathlib.Path,
                         repo_root: pathlib.Path, n_tokens: int,
                         dirty: str | None, work_dir: pathlib.Path) -> dict:
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    if out_path.exists():
        if args.resume:
            log("\n%s already complete, skipping" % out_path)
            return json.loads(out_path.read_text(encoding="utf-8"))
        raise FileExistsError("refusing to overwrite immutable result: %s" % out_path)

    resuming = False
    if partial.exists():
        if not args.resume:
            raise FileExistsError(
                "partial result already exists: %s (pass --resume to continue it, "
                "or remove it to start over)" % partial)
        result = json.loads(partial.read_text(encoding="utf-8"))
        mismatched = [
            (name, (result.get(name, False) if name == "require_thermally_clean"
                    else result.get(name)), value) for name, value in
             (("max_new_tokens", n_tokens), ("blocks_requested", args.blocks),
              ("seed", args.seed), ("selected_methods", selected),
              ("require_thermally_clean", args.require_thermally_clean))
             if (result.get(name, False) if name == "require_thermally_clean"
                 else result.get(name)) != value]
        # prompt_suite predates this field in older .partial files; a
        # missing key means "evaluation" (the only split that existed
        # then), not "leave unspecified and refuse to resume".
        if result.get("prompt_suite", "evaluation") != args.prompt_suite:
            mismatched.append(
                ("prompt_suite", result.get("prompt_suite", "evaluation"),
                 args.prompt_suite))
        if mismatched:
            raise ValueError(
                "cannot resume %s: it was started with different settings than this "
                "invocation: %s" % (partial, mismatched))
        result.setdefault("cells", [])
        result.setdefault("rows_by_method", {method_id: [] for method_id in selected})
        # A .partial file normally still has rows_by_method (checkpoint()
        # saves the in-progress result as-is, before finalize_result()'s
        # own del rows_by_method ever runs). But this project's own
        # established way to retry a failed method in an already-FINISHED
        # run is to rename <name>.json -> <name>.json.partial and rerun
        # with --resume (see docs/REPRODUCE.md) -- and a finished file has
        # already been through finalize_result(), which moves each
        # method's real per-row measurements into methods[i]["rows"] and
        # deletes rows_by_method entirely. Without this backfill, resuming
        # from a renamed-finished file silently starts every method's
        # rows_by_method empty; only methods actually re-executed in this
        # invocation get real rows back, and finalize_result() then
        # rebuilds "methods" from that mostly-empty rows_by_method,
        # overwriting the previously-real rows/summary/paired_vs_control
        # for every method that was correctly skipped as already-done.
        # Caught by hand tonight (Qwen3-14B TTFT table: 5 of 6 methods
        # showed outcome="error" with 0 rows in the final artifact after a
        # DeepSpeed-only retry, despite every one of their cells recording
        # error=None) -- recovered that run's real numbers from its stdout
        # log rather than trusting the corrupted artifact. This backfill
        # is the actual fix so future retries do not depend on the log
        # surviving to recover from the same loss.
        for method_entry in result.get("methods", []):
            method_id = method_entry.get("method_id")
            if method_id in result["rows_by_method"] and not result["rows_by_method"][method_id]:
                result["rows_by_method"][method_id] = list(method_entry.get("rows") or [])
        for method_id in selected:
            result["rows_by_method"].setdefault(method_id, [])
        resuming = True
        already_done_count = len(completed_cells(result))
        already_capacity_failed_count = len(capacity_failed_cells(result))
        log("\nRESUMING %s: %d cell(s) already complete, %d predeclared capacity "
           "failure(s) (not retried)" % (partial, already_done_count,
                                         already_capacity_failed_count))
    else:
        result = {
            "schema_version": 3,
            "kind": "paper_comparison_randomized_block",
            "status": "running",
            "exploratory": args.blocks < 8,
            "evidence_level": "L1_mechanism_screen" if args.blocks < 2 else "L2_regulated_exploratory",
            "prompt_suite_version": PROMPT_SUITE_VERSION,
            "prompt_suite": args.prompt_suite,
            "evaluation_case_ids": [item["case"].id for item in rendered],
            "max_new_tokens": n_tokens,
            "workload": workload_for(n_tokens),
            "blocks_requested": args.blocks,
            "warmup_tokens": args.warmup_tokens,
            "cooldown_seconds": args.cooldown_seconds,
            "cooldown_max_temp_c": args.cooldown_max_temp_c,
            "control_method": CONTROL_METHOD,
            "cache_regime": "cold page cache before every timed cell",
            "cell_isolation": "one fresh subprocess per (block, method) cell",
            "model": args.model,
            "dfloat11_model": args.dfloat11_model,
            "draft_model": args.draft_model,
            "store": args.store,
            "selected_methods": selected,
            "seed": args.seed,
            "require_thermally_clean": args.require_thermally_clean,
            "environment": environment_manifest(repo_root, tokenizer,
                                                store=pathlib.Path(args.store)),
            "reproducible_from_commit": not bool(dirty),
            "method_order_per_block": [],
            "rows_by_method": {method_id: [] for method_id in selected},
            "cells": [],
            "failures": [],
        }
    checkpoint(partial, result)

    started = time.perf_counter()
    deadline = started + args.time_budget_minutes_per_length * 60
    rng = random.Random(args.seed)
    case_ids = [item["case"].id for item in rendered]
    # A predeclared capacity failure (see capacity_failed_cells) is skipped
    # on resume the same as a real success -- it deterministically fails
    # the same way every time (the hardware still does not have enough
    # VRAM), so retrying it only burns time without producing new
    # information.
    already_done = completed_cells(result) | capacity_failed_cells(result)

    for block in range(args.blocks):
        if time.perf_counter() >= deadline:
            result["failures"].append(
                {"block": block, "error": "not started: time budget exhausted"})
            continue
        # Re-derive the order for every block even when resuming, so the
        # shared rng advances exactly as it did the first time and later
        # blocks' orders stay identical to what a completed run already
        # recorded -- only whether a cell is *dispatched* changes.
        order = _shuffled_order(selected, rng)
        if not resuming or block >= len(result["method_order_per_block"]):
            result["method_order_per_block"].append(order)
        log("\nBLOCK %d/%d  order: %s" % (block + 1, args.blocks, ", ".join(order)))
        for method_id in order:
            method = METHODS[method_id]
            if (block, method_id) in already_done:
                log("  SKIP (resumed): %s" % method.title)
                continue
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                result["failures"].append({
                    "block": block, "method": method_id,
                    "error": "not started: time budget exhausted"})
                continue
            # An inter-method cooldown too, not only the inter-cell one
            # inside each run_* function -- loading/decompressing a 14B
            # checkpoint is itself GPU/IO work that can leave heat behind
            # before the first timed cell of the next method.
            bounded.COOLDOWN_SECONDS = args.cooldown_seconds
            bounded.COOLDOWN_MAX_TEMPERATURE_C = args.cooldown_max_temp_c
            cool_down(args.cooldown_seconds, args.cooldown_max_temp_c)

            log("  METHOD: %s" % method.title)
            cell_config = {
                "method_id": method_id, "model": args.model,
                "dfloat11_model": args.dfloat11_model, "draft_model": args.draft_model,
                "store": args.store, "n_tokens": n_tokens, "block": block,
                "warmup_tokens": args.warmup_tokens,
                "cooldown_seconds": args.cooldown_seconds,
                "cooldown_max_temp_c": args.cooldown_max_temp_c,
                "require_thermally_clean": args.require_thermally_clean,
                "seconds_remaining": deadline - time.perf_counter(),
                "case_ids": case_ids, "prompt_suite": args.prompt_suite,
                # The worker runs in a fresh subprocess and does not inherit
                # this process's METHODS dict -- a method registered here
                # only at runtime (budget_method_variants()'s exact-<N>gb/
                # accelerate-<N>gb entries) would not exist in the worker's
                # own copy at all. Sending the resolved spec directly, for
                # every method (not only dynamic ones), means the worker
                # never has to assume its own METHODS matches this
                # process's -- see run_cell()'s handling of these fields.
                "method_title": method.title, "method_kind": method.kind,
                "method_overrides": method.overrides,
                "method_exactness": method.exactness,
            }
            # +180s slack beyond the nominal remaining budget: the worker's
            # own deadline check is what actually truncates a cell's case
            # sweep, this is only a backstop against a genuinely hung
            # subprocess (a stuck download, a wedged driver), not the
            # mechanism that enforces --time-budget-minutes-per-length.
            cell_result = _run_cell_in_subprocess(
                cell_config, work_dir, timeout_s=remaining + 180.0)
            result["rows_by_method"][method_id].extend(cell_result.get("rows") or [])
            result["cells"].append({
                "block": block, "method": method_id,
                "metadata": cell_result.get("metadata") or {},
                "peak_host_rss_bytes": cell_result.get("peak_host_rss_bytes"),
                "thermal_monitoring": cell_result.get("thermal_monitoring"),
                "thermal_measurement_monitoring": cell_result.get(
                    "thermal_measurement_monitoring"),
                "error": cell_result.get("error")})
            if cell_result.get("error"):
                result["failures"].append({
                    "block": block, "method": method_id,
                    "error": cell_result["error"],
                    "traceback": cell_result.get("traceback")})
                log("  FAILED: %s" % cell_result["error"])
            else:
                already_done.add((block, method_id))
            result["elapsed_seconds"] = time.perf_counter() - started
            checkpoint(partial, result)

    methods_out = []
    pareto_points = []
    for method_id in selected:
        rows = result["rows_by_method"][method_id]
        summary = aggregate(rows) if rows else {}
        # At max_new_tokens=1, seconds_per_token and wall-clock time to the
        # first token are the same number by construction (wall_seconds /
        # 1). Presenting that as "N s/token" invites treating a one-token
        # measurement as a steady-state decode rate, which it is not --
        # ttft_seconds is the same value under the name a reader should
        # actually use for it. See run_paper_comparison.py's workload_for.
        if result.get("workload") == "ttft" and summary.get("seconds_per_token") is not None:
            summary["ttft_seconds"] = summary["seconds_per_token"]
        method_cells = [cell for cell in result["cells"] if cell["method"] == method_id]
        successful_method_cells = [
            cell for cell in method_cells if cell.get("error") is None]
        # Failed attempts stay in cells/failures as audit evidence, but their
        # RSS, initialization, and thermal state must not contaminate the
        # summary of the successful retry that supplied the reported rows.
        peak_rss_values = [cell["peak_host_rss_bytes"] for cell in successful_method_cells
                           if cell["peak_host_rss_bytes"] is not None]
        init_seconds_values = [
            cell["metadata"]["initialization_seconds"] for cell in successful_method_cells
            if isinstance(cell.get("metadata"), dict)
            and cell["metadata"].get("initialization_seconds") is not None]
        thermal_summaries = [cell["thermal_monitoring"] for cell in successful_method_cells
                             if cell.get("thermal_monitoring")]
        # New artifacts classify thermal/power events from paired counter
        # snapshots that bracket timed inference only. Fall back to the older
        # whole-cell summary solely for resumability of pre-change partials.
        measurement_summaries = [
            cell.get("thermal_measurement_monitoring") or cell.get("thermal_monitoring")
            for cell in successful_method_cells
            if cell.get("thermal_measurement_monitoring") or cell.get("thermal_monitoring")]
        sm_clock_mins = [t["sm_clock_mhz_min"] for t in thermal_summaries
                         if t.get("sm_clock_mhz_min") is not None]
        temp_maxes = [t["temperature_c_max"] for t in thermal_summaries
                     if t.get("temperature_c_max") is not None]
        thermal_throttle_flags = [
            t["any_thermal_throttle_during_measurement"] for t in measurement_summaries
            if t.get("any_thermal_throttle_during_measurement") is not None]
        power_limit_flags = [
            t["any_power_limit_during_measurement"] for t in measurement_summaries
            if t.get("any_power_limit_during_measurement") is not None]
        throttle_flags = [
            (t["any_throttle_during_measurement"]
             if t.get("any_throttle_during_measurement") is not None
             else bool(t.get("any_thermal_throttle_during_measurement") or
                       t.get("any_power_limit_during_measurement")))
            for t in measurement_summaries
            if (t.get("any_throttle_during_measurement") is not None or
                t.get("any_thermal_throttle_during_measurement") is not None or
                t.get("any_power_limit_during_measurement") is not None)]
        energy_values = [t["energy_joules_estimate"] for t in thermal_summaries
                         if t.get("energy_joules_estimate") is not None]
        # Sum, not mean: each cell's energy_joules_estimate already covers
        # that whole cell (every case in one block for this method), so
        # summing across cells and dividing by every token this method
        # produced gives joules/token over the full campaign, not just one
        # cell. This total also includes each cell's untimed warm-up and
        # the inter-case cooldown pauses inside it (ThermalSampler wraps
        # the whole cell dispatch, not just the timed generate() calls),
        # so it is deliberately a conservative, inclusive estimate rather
        # than an idealized decode-only figure -- see run_paper_comparison
        # _worker.thermal_monitor_summary's own energy_joules_estimate
        # docstring for the sampling-precision caveat underneath it.
        total_energy_j = sum(energy_values) if energy_values else None
        total_tokens_for_energy = summary.get("total_output_tokens")
        energy_j_per_token = (
            total_energy_j / total_tokens_for_energy
            if total_energy_j is not None and total_tokens_for_energy else None)
        capacity_failure_cells = [
            cell for cell in method_cells if cell.get("error") is not None
            and isinstance(cell.get("metadata"), dict)
            and cell["metadata"].get("capacity_failure")]
        # A capacity failure is a finding ("this method cannot fit the
        # available hardware"), never a table row that just says "failed" --
        # this is the field a paper's methods table should key its
        # capacity-failure vs. measured-result rows on, instead of
        # inferring it from an empty rows list, which also covers "not
        # attempted yet" and "failed for an unrelated bug" identically.
        if rows:
            outcome = "measured"
        elif method_cells and len(capacity_failure_cells) == len(method_cells):
            outcome = "capacity_failure"
        elif method_cells:
            outcome = "error"
        else:
            outcome = "not_attempted"
        entry = {
            "method_id": method_id, "title": METHODS[method_id].title,
            "declared_exactness": METHODS[method_id].exactness,
            "rows": rows, "summary": summary,
            "cells": method_cells,
            "outcome": outcome,
            "capacity_failure_blocks": len(capacity_failure_cells),
            "peak_host_rss_bytes": max(peak_rss_values) if peak_rss_values else None,
            "initialization_seconds_median": (
                statistics.median(init_seconds_values) if init_seconds_values else None),
            "thermal_across_all_cells": {
                "sm_clock_mhz_min": min(sm_clock_mins) if sm_clock_mins else None,
                "temperature_c_max": max(temp_maxes) if temp_maxes else None,
                "any_throttle_during_measurement": (
                    any(throttle_flags) if throttle_flags else None),
                "any_thermal_throttle_during_measurement": (
                    any(thermal_throttle_flags) if thermal_throttle_flags else None),
                "thermal_throttled_cells": sum(bool(value) for value in thermal_throttle_flags),
                "any_power_limit_during_measurement": (
                    any(power_limit_flags) if power_limit_flags else None),
                "power_limited_cells": sum(bool(value) for value in power_limit_flags),
                # The worker-level sampler starts before method setup and ends
                # after cleanup. These are deliberately named inclusive so a
                # paper cannot mistake setup+warmup+generation energy for a
                # decode-only hardware counter.
                "energy_scope": "cell setup, warmup, generation, cooldown, and cleanup",
                "inclusive_total_energy_joules_estimate": total_energy_j,
                "inclusive_energy_joules_per_output_token_estimate": energy_j_per_token,
            },
        }
        if method_id != CONTROL_METHOD:
            entry["paired_vs_control"] = paired_block_log_ratios(
                result["rows_by_method"], CONTROL_METHOD, method_id)
            entry["token_exactness_vs_control"] = token_exactness(
                result["rows_by_method"], CONTROL_METHOD, method_id)
        if summary.get("peak_vram_gb") is not None and summary.get("seconds_per_token") is not None:
            entry["vram_regime"] = vram_regime(summary["peak_vram_gb"])
            pareto_points.append({
                "method_id": method_id, "peak_vram_gb": summary["peak_vram_gb"],
                "seconds_per_token": summary["seconds_per_token"],
                "vram_regime": entry["vram_regime"]})
        methods_out.append(entry)
    result["methods"] = methods_out
    result["pareto_frontier"] = pareto_frontier(pareto_points)
    del result["rows_by_method"]

    eligible, eligibility_reason = paper_eligibility(result, args.blocks, selected)
    result["paper_eligible"] = eligible
    result["paper_eligibility_reason"] = eligibility_reason
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = ("time_capped" if result["elapsed_seconds"] >=
                        args.time_budget_minutes_per_length * 60 else "complete")
    result["completed_at_unix"] = time.time()
    checkpoint(partial, result)
    if args.require_complete and not eligible:
        log("\nNOT writing immutable result: --require-complete and paper_eligible is "
            "False (%s). Partial state is saved at %s -- rerun with --resume once the "
            "missing cells can complete." % (eligibility_reason, partial))
        result["status"] = "incomplete_kept_as_partial"
        return result
    partial.replace(out_path)
    log("\nwrote immutable result %s (paper_eligible=%s)" % (out_path, eligible))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=bounded.MODEL)
    parser.add_argument("--draft-model", default=bounded.DRAFT_MODEL)
    parser.add_argument("--dfloat11-model", default=bounded.DFLOAT11_MODEL)
    parser.add_argument("--store", default=bounded.STORE)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                        help="comma-separated method IDs; choices: %s" %
                             ",".join(sorted(METHODS)))
    parser.add_argument(
        "--vram-budgets", default=None,
        help="comma-separated GB values (e.g. '2,3,4'); for each one, adds an "
             "exact-<N>gb and accelerate-<N>gb method pinned to exactly that "
             "VRAM budget, on top of --methods, so the headline comparison is "
             "never one table mixing configs at different memory budgets as "
             "though memory were held equal. Combine with vram_regime/"
             "pareto_frontier in the result JSON for the (VRAM, seconds/token) "
             "plot this is for. AirLLM and DFloat11 contribute whatever "
             "peak_vram_gb they naturally land on instead -- see "
             "budget_method_variants()'s docstring for why only Afterimage and "
             "Accelerate can be pinned this way today.")
    parser.add_argument(
        "--prompt-suite", default="evaluation",
        choices=["evaluation", "paper_generation"],
        help="'evaluation' is paper-short-v1 (the four short factual cases) -- "
             "use it for the ttft/short_cold_start workloads. 'paper_generation' "
             "is paper-generation-v1 (explanation/summarization/code/analytical, "
             "each eliciting a real ~120-180-word answer) -- use it for the "
             "100-128-token decode workload, since forcing the short factual "
             "cases to keep generating past their one-token answer produces a "
             "strange speculative-decoding workload. Run the script twice, once "
             "per suite, rather than mixing both into one --token-lengths sweep.")
    parser.add_argument("--token-lengths", default=None,
                        help="comma-separated --max-new-tokens values run back to "
                             "back, one immutable result file each. Short lengths "
                             "measure cold-start latency; only a length at least as "
                             "large as any spec_k in --methods actually exercises a "
                             "full speculative chain (spec-fixed uses spec_k=8). "
                             "Default: 1,4 for evaluation; 1,32,128 for "
                             "paper_generation.")
    parser.add_argument("--blocks", type=int, default=3,
                        help="randomized-order full sweeps per token length (default "
                             "3, i.e. 'each pass 3 times'). Each block reloads every "
                             "selected method's model once, in a freshly shuffled "
                             "order -- raise this for a confirmatory claim (8-12 is "
                             "the range this project's own methodology review "
                             "suggests, set from a pilot block's variance) at the "
                             "cost of that many more model reloads.")
    parser.add_argument("--warmup-tokens", type=int, default=8,
                        help="untimed generate() call per (block, method) before "
                             "the timed sweep, absorbing first-call CUDA/Triton/"
                             "allocator compilation so it does not land inside the "
                             "measured seconds/token. 0 disables it.")
    parser.add_argument("--cooldown-seconds", type=float, default=20.0)
    parser.add_argument("--cooldown-max-temp-c", type=float, default=75.0)
    parser.add_argument(
        "--require-thermally-clean", action="store_true",
        help="keep a cell resumable instead of accepting its timing when continuous "
             "monitoring observes a thermal slowdown, or cannot observe thermal "
             "status. Ordinary software power capping remains measured and reported "
             "but does not reject the cell.")
    parser.add_argument("--case-ids", default=None,
                        help="comma-separated evaluation case IDs; default is all")
    parser.add_argument("--time-budget-minutes-per-length", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0,
                        help="method-order shuffle seed; fixed by default so a run "
                             "is reproducible given the same --blocks and --methods, "
                             "not so every block gets the same order (each block "
                             "advances the same Random instance).")
    parser.add_argument("--out-dir", default="results/paper-comparison")
    parser.add_argument("--run-label", default=None,
                        help="filename component; default is the model name plus "
                             "today's date")
    parser.add_argument(
        "--allow-dirty-tree", action="store_true",
        help="proceed even with uncommitted changes; the result is recorded but "
             "not reproducible from its git_commit alone")
    parser.add_argument(
        "--resume", action="store_true",
        help="continue an existing .partial file instead of refusing to run "
             "because one is already there. A cell only counts as already done "
             "if it completed without error; a length whose immutable .json "
             "output already exists is skipped entirely. Must be run with the "
             "same --model/--methods/--blocks/--seed as the run being resumed, "
             "or it refuses (those determine the exact cell matrix and the "
             "randomized block orders, and a --resume under different settings "
             "would silently merge two different experiments into one file).")
    parser.add_argument(
        "--require-complete", action="store_true",
        help="do not finalize a token length's immutable result unless every "
             "requested (block, method) cell actually succeeded (paper_eligible "
             "is True). An incomplete run's state is still saved to the .partial "
             "file so --resume can pick it up later; only the final rename to "
             "the immutable .json is withheld. Without this flag, a run that "
             "hits its time budget partway through still finalizes normally, "
             "with paper_eligible recorded honestly as False.")
    args = parser.parse_args()

    selected = [part.strip() for part in args.methods.split(",") if part.strip()]
    unknown = sorted(set(selected) - set(METHODS))
    if unknown:
        parser.error("unknown methods: %s" % ", ".join(unknown))
    if CONTROL_METHOD not in selected:
        parser.error("--methods must include %r (the comparison control)" % CONTROL_METHOD)
    if args.vram_budgets:
        try:
            budgets = [float(part.strip()) for part in args.vram_budgets.split(",")
                      if part.strip()]
        except ValueError:
            parser.error("--vram-budgets must be a comma-separated list of numbers")
        if any(budget <= 0 for budget in budgets):
            parser.error("--vram-budgets values must be positive")
        for budget in budgets:
            variants = budget_method_variants(budget)
            METHODS.update(variants)
            for method_id in variants:
                if method_id not in selected:
                    selected.append(method_id)
                if method_id.startswith("accelerate-"):
                    DEPENDENCY_PACKAGE[method_id] = "accelerate"
    token_lengths_arg = args.token_lengths or ",".join(
        map(str, DEFAULT_TOKEN_LENGTHS_BY_SUITE[args.prompt_suite]))
    try:
        token_lengths = [int(part.strip()) for part in token_lengths_arg.split(",")
                         if part.strip()]
    except ValueError:
        parser.error("--token-lengths must be a comma-separated list of integers")
    if not token_lengths or any(n < 1 for n in token_lengths):
        parser.error("--token-lengths values must be positive")
    if args.blocks < 1:
        parser.error("--blocks must be positive")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must not be negative")

    # Only used directly by this process for load_tokenizer/render_cases/
    # environment_manifest below; every actual method execution happens in
    # a worker subprocess, which receives model/store/draft_model/
    # dfloat11_model explicitly through its own cell config instead of
    # inheriting these globals.
    bounded.MODEL = args.model
    bounded.STORE = args.store

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the hardware comparison")

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    dirty = command_output(["git", "-C", str(repo_root), "status", "--short"])
    if dirty and not args.allow_dirty_tree:
        raise RuntimeError(
            "refusing to run with uncommitted changes (git status --short is "
            "non-empty): a result's git_commit only reproduces the code that "
            "produced it if the tree was clean. Commit or stash first, or pass "
            "--allow-dirty-tree for a deliberately non-reproducible local run.\n"
            + dirty)

    missing = [method_id for method_id in selected
              if method_id in DEPENDENCY_PACKAGE
              and bounded.package_version(DEPENDENCY_PACKAGE[method_id]) is None]
    if missing:
        raise RuntimeError(
            "missing packages for selected methods %s; install with "
            "`pip install -e .[bench]` before starting the campaign" %
            ", ".join(missing))

    tokenizer = load_tokenizer(args.model)
    evaluation_cases = prompt_cases(args.prompt_suite)
    if args.case_ids:
        requested = [part.strip() for part in args.case_ids.split(",") if part.strip()]
        by_id = {case.id: case for case in evaluation_cases}
        unknown_cases = sorted(set(requested) - set(by_id))
        if unknown_cases:
            parser.error("unknown evaluation cases: %s" % ", ".join(unknown_cases))
        evaluation_cases = tuple(by_id[case_id] for case_id in requested)
    rendered = render_cases(tokenizer, evaluation_cases)
    for item in rendered:
        item["tokenizer"] = tokenizer

    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.run_label or (
        args.model.split("/")[-1].lower() + "-" + time.strftime("%Y-%m-%d"))

    written = []
    incomplete = []
    out_path_by_length: dict[int, pathlib.Path] = {}
    with tempfile.TemporaryDirectory(prefix="afterimage-paper-cell-") as work_dir_name:
        work_dir = pathlib.Path(work_dir_name)
        for n_tokens in token_lengths:
            out_path = out_dir / ("%s-%dtok.json" % (label, n_tokens))
            log("\n%s\nTOKEN LENGTH %d (%s) [%s]\n%s" %
                ("=" * 60, n_tokens, out_path, workload_for(n_tokens), "=" * 60))
            run_one_token_length(args, tokenizer, rendered, selected, out_path,
                                 repo_root, n_tokens, dirty, work_dir)
            if out_path.exists():
                written.append(str(out_path))
                out_path_by_length[n_tokens] = out_path
            else:
                incomplete.append(str(out_path))

    if written:
        log("\nWrote: %s" % ", ".join(written))
    if incomplete:
        log("Withheld (incomplete, --require-complete set): %s -- rerun with "
            "--resume once the missing cells can complete." % ", ".join(incomplete))

    if 1 in out_path_by_length:
        decode_length = max((n for n in out_path_by_length if n >= 100), default=None)
        if decode_length is not None:
            ttft_result = json.loads(out_path_by_length[1].read_text(encoding="utf-8"))
            decode_result = json.loads(
                out_path_by_length[decode_length].read_text(encoding="utf-8"))
            derived = derive_ttft_decode_metrics(ttft_result, decode_result)
            derived_path = out_dir / ("%s-ttft-decode-derived.json" % label)
            derived_path.write_text(json.dumps({
                "kind": "ttft_decode_derived", "ttft_source": str(out_path_by_length[1]),
                "decode_source": str(out_path_by_length[decode_length]),
                "decode_max_new_tokens": decode_length, "methods": derived,
            }, indent=2, sort_keys=True), encoding="utf-8")
            log("Wrote derived TTFT/decode-TPS metrics: %s" % derived_path)

    log("Rebuild the results index with: python scripts/build_results_index.py "
        "> results/INDEX.md")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
