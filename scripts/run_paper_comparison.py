#!/usr/bin/env python3
"""One-click, randomized-block comparison for the paper's headline claim:
Afterimage vs AirLLM vs Hugging Face Accelerate vs DFloat11, on identical
prompts and hardware, at multiple output-token lengths, with real inter-
method randomization instead of a fixed method order.

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

This is a real engineering trade-off, not a free upgrade: block-major
execution reloads every method's model once per block instead of once for
the whole run (5-6 methods x --blocks reloads instead of 5-6), which is why
--blocks defaults to 3 ("each pass 3 times") rather than the 8-12 blocks a
confirmatory paper claim should eventually use -- raise --blocks for that
once a pilot run's block-to-block variance is known.

Requires (WSL2/Linux only, matching run_bounded_suite.py and benchmark.sh):
CUDA, a prepared Afterimage store for --model, and the optional-dependency
group `bench` installed (`pip install -e .[bench]`, or by hand:
`pip install airllm "accelerate>=1.0" "dfloat11[cuda12]"`). Any method whose
package is missing is skipped with a recorded reason, not a hard failure,
so a partial dependency set still produces a usable result.

Usage:
    python scripts/run_paper_comparison.py \\
        --store /root/afterimage/store_14b \\
        --out-dir results/paper-comparison

Or via the one-click wrapper: ./paper_benchmark.sh
"""
from __future__ import annotations

import argparse
import gc
import math
import pathlib
import random
import statistics
import sys
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
    run_accelerate,
    run_afterimage,
    run_airllm,
    run_dfloat11,
)
from scripts import run_bounded_suite as bounded

# The core headline comparison: one representative from each execution
# family (published disk-offload baseline, published GPU-resident
# compression baseline, and Afterimage's own exact-streaming controls).
DEFAULT_METHODS = ("airllm", "accelerate", "dfloat11", "exact-min",
                   "exact-resident", "spec-fixed")
DEFAULT_TOKEN_LENGTHS = (4, 32, 128)

# The Afterimage method every other method is compared against. exact-min
# is the "reference_execution_equivalent" control -- greedy-exact, no
# speculation, no residency heuristics -- so a speedup measured against it
# isolates what a given method or mechanism contributes, not a second
# confounding Afterimage feature.
CONTROL_METHOD = "exact-min"

DEPENDENCY_PACKAGE = {"airllm": "airllm", "accelerate": "accelerate",
                      "dfloat11": "dfloat11"}


def _shuffled_order(methods: list[str], rng: random.Random) -> list[str]:
    order = list(methods)
    rng.shuffle(order)
    return order


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


def run_one_token_length(args, tokenizer, rendered: list[dict],
                         selected: list[str], out_path: pathlib.Path,
                         repo_root: pathlib.Path, n_tokens: int,
                         dirty: str | None) -> dict:
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    if out_path.exists():
        raise FileExistsError("refusing to overwrite immutable result: %s" % out_path)
    if partial.exists():
        raise FileExistsError("partial result already exists: %s" % partial)

    result = {
        "schema_version": 1,
        "kind": "paper_comparison_randomized_block",
        "status": "running",
        "exploratory": args.blocks < 8,
        "evidence_level": "L1_mechanism_screen" if args.blocks < 2 else "L2_regulated_exploratory",
        "prompt_suite_version": PROMPT_SUITE_VERSION,
        "evaluation_case_ids": [item["case"].id for item in rendered],
        "max_new_tokens": n_tokens,
        "blocks_requested": args.blocks,
        "warmup_tokens": args.warmup_tokens,
        "cooldown_seconds": args.cooldown_seconds,
        "cooldown_max_temp_c": args.cooldown_max_temp_c,
        "control_method": CONTROL_METHOD,
        "cache_regime": "cold page cache before every timed cell",
        "model": args.model,
        "dfloat11_model": args.dfloat11_model,
        "draft_model": args.draft_model,
        "store": args.store,
        "selected_methods": selected,
        "seed": args.seed,
        "environment": environment_manifest(repo_root, tokenizer,
                                            store=pathlib.Path(args.store)),
        "reproducible_from_commit": not bool(dirty),
        "method_order_per_block": [],
        "rows_by_method": {method_id: [] for method_id in selected},
        "failures": [],
    }
    checkpoint(partial, result)

    started = time.perf_counter()
    deadline = started + args.time_budget_minutes_per_length * 60
    rng = random.Random(args.seed)
    draft_model = None

    for block in range(args.blocks):
        if time.perf_counter() >= deadline:
            result["failures"].append(
                {"block": block, "error": "not started: time budget exhausted"})
            continue
        order = _shuffled_order(selected, rng)
        result["method_order_per_block"].append(order)
        log("\nBLOCK %d/%d  order: %s" % (block + 1, args.blocks, ", ".join(order)))
        for method_id in order:
            method = METHODS[method_id]
            if time.perf_counter() >= deadline:
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

            # Every run_* function's own rows_checkpoint callback receives
            # the *complete* rows-so-far list for this one (block, method)
            # call, starting from an empty list each call (repeats=1) --
            # never a delta. pre_block_count marks where this block's rows
            # start in the accumulated per-method list, so both the
            # interim callback and the final sync below can replace exactly
            # this block's slice, idempotently, instead of appending on top
            # of what the callback already wrote (which would double-count
            # every row once the call returns).
            pre_block_count = len(result["rows_by_method"][method_id])

            def checkpoint_cb(rows, method_id=method_id, mark=pre_block_count):
                result["rows_by_method"][method_id] = (
                    result["rows_by_method"][method_id][:mark] + rows)
                result["elapsed_seconds"] = time.perf_counter() - started
                checkpoint(partial, result)

            log("  METHOD: %s" % method.title)
            rows = []
            try:
                if method.kind == "airllm":
                    rows, metadata = run_airllm(
                        method, rendered, n_tokens, deadline, checkpoint_cb,
                        repeats=1, repeat_offset=block, warmup_tokens=args.warmup_tokens)
                elif method.kind == "accelerate":
                    rows, metadata = run_accelerate(
                        method, rendered, n_tokens, deadline, checkpoint_cb,
                        repeats=1, repeat_offset=block, warmup_tokens=args.warmup_tokens)
                elif method.kind == "dfloat11":
                    rows, metadata = run_dfloat11(
                        method, rendered, n_tokens, deadline, checkpoint_cb,
                        repeats=1, repeat_offset=block, warmup_tokens=args.warmup_tokens)
                else:
                    if method_id == "spec-fixed" and draft_model is None:
                        from afterimage.runtime.streaming_engine import load_draft_model
                        log("  loading resident draft model %s" % args.draft_model)
                        draft_model = load_draft_model(args.draft_model, device="cuda")
                    rows, metadata = run_afterimage(
                        method, rendered, n_tokens, deadline,
                        draft_model=draft_model if method_id == "spec-fixed" else None,
                        burn_in_rendered=rendered[:1] if args.warmup_tokens > 0 else None,
                        burn_in_tokens=args.warmup_tokens,
                        rows_checkpoint=checkpoint_cb, repeats=1, repeat_offset=block)
                result["rows_by_method"][method_id] = (
                    result["rows_by_method"][method_id][:pre_block_count] + rows)
            except Exception as exc:
                result["failures"].append({
                    "block": block, "method": method_id, "error": repr(exc),
                    "traceback": traceback.format_exc()})
                log("  FAILED: %r" % exc)
            result["elapsed_seconds"] = time.perf_counter() - started
            checkpoint(partial, result)

    if draft_model is not None:
        del draft_model
        gc.collect()
        torch.cuda.empty_cache()

    methods_out = []
    for method_id in selected:
        rows = result["rows_by_method"][method_id]
        entry = {"method_id": method_id, "title": METHODS[method_id].title,
                 "declared_exactness": METHODS[method_id].exactness,
                 "rows": rows, "summary": aggregate(rows) if rows else {}}
        if method_id != CONTROL_METHOD:
            entry["paired_vs_control"] = paired_block_log_ratios(
                result["rows_by_method"], CONTROL_METHOD, method_id)
        methods_out.append(entry)
    result["methods"] = methods_out
    del result["rows_by_method"]

    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = ("time_capped" if result["elapsed_seconds"] >=
                        args.time_budget_minutes_per_length * 60 else "complete")
    result["completed_at_unix"] = time.time()
    checkpoint(partial, result)
    partial.replace(out_path)
    log("\nwrote immutable result %s" % out_path)
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
    parser.add_argument("--token-lengths", default=",".join(map(str, DEFAULT_TOKEN_LENGTHS)),
                        help="comma-separated --max-new-tokens values run back to "
                             "back, one immutable result file each. Short lengths "
                             "measure cold-start latency; only a length at least as "
                             "large as any spec_k in --methods actually exercises a "
                             "full speculative chain (spec-fixed uses spec_k=8).")
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
    args = parser.parse_args()

    selected = [part.strip() for part in args.methods.split(",") if part.strip()]
    unknown = sorted(set(selected) - set(METHODS))
    if unknown:
        parser.error("unknown methods: %s" % ", ".join(unknown))
    if CONTROL_METHOD not in selected:
        parser.error("--methods must include %r (the comparison control)" % CONTROL_METHOD)
    try:
        token_lengths = [int(part.strip()) for part in args.token_lengths.split(",")
                         if part.strip()]
    except ValueError:
        parser.error("--token-lengths must be a comma-separated list of integers")
    if not token_lengths or any(n < 1 for n in token_lengths):
        parser.error("--token-lengths values must be positive")
    if args.blocks < 1:
        parser.error("--blocks must be positive")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must not be negative")

    bounded.MODEL = args.model
    bounded.DRAFT_MODEL = args.draft_model
    bounded.DFLOAT11_MODEL = args.dfloat11_model
    bounded.STORE = args.store
    bounded.COOLDOWN_SECONDS = args.cooldown_seconds
    bounded.COOLDOWN_MAX_TEMPERATURE_C = args.cooldown_max_temp_c

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
        log("WARNING: missing packages for %s -- those methods will fail per-block "
            "rather than block the rest of the run. Install with "
            "`pip install -e .[bench]`." % ", ".join(missing))

    tokenizer = load_tokenizer(args.model)
    evaluation_cases = prompt_cases("evaluation")
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
    for n_tokens in token_lengths:
        out_path = out_dir / ("%s-%dtok.json" % (label, n_tokens))
        log("\n%s\nTOKEN LENGTH %d (%s)\n%s" % ("=" * 60, n_tokens, out_path, "=" * 60))
        run_one_token_length(args, tokenizer, rendered, selected, out_path,
                             repo_root, n_tokens, dirty)
        written.append(str(out_path))

    log("\nAll token lengths complete. Wrote: %s" % ", ".join(written))
    log("Rebuild the results index with: python scripts/build_results_index.py "
        "> results/INDEX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
