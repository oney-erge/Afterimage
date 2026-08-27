#!/usr/bin/env python3
"""H19 (Candidate-Amortization Hypothesis): how does the wall-clock cost of
one target verification sweep scale with the number of already-known
candidate positions it verifies?

Deliberately does not generate real speculative trees or run a draft model
-- see StreamingLosslessModel.measure_candidate_sweep_latency's own
docstring (afterimage/runtime/streaming_engine.py) for why the candidate
tokens' values do not matter here, only their count. This script exists
because H19 is the prerequisite this project's own methodology review
identified for every tree-based speculation strategy that could follow it
(exhaustive/SpecInfer/Sequoia/OPT-Tree/SpecExec/cost-aware trees -- see
docs/SPECULATION_TREE_RESEARCH.md): building any of those before knowing
this machine's actual candidate-parallelism knee means guessing a node
budget instead of measuring one.

Called a "Candidate Vectorization/Amortization Curve" deliberately, not a
"Tree Amortization Curve": this measures cost as a function of KNOWN
candidate positions in a normal linear sequence, not arbitrary tree nodes
under real tree-attention masking. That distinction only collapses once
H20 (a real verifier) exists -- see docs/SPECULATION_TREE_RESEARCH.md's
own exactness-boundary section for why the two must not be conflated in
either the code or the writeup.

H19 is deliberately NOT registered in afterimage/experiments.py's live
HYPOTHESES/PROTOCOLS registry: that registry's TestProtocol/EvidenceStage
schema (afterimage/protocols.py) is built around paired candidate-vs-
control comparisons with a strict, validated 1:1 hypothesis<->protocol
mapping, and H19 is a monotonic parameter sweep over one arm, not a paired
comparison -- forcing it into that shape would either misrepresent the
experiment or require extending shared, tightly-coupled infrastructure
that backs the live product's Lab UI. This script runs independently
instead, the same way scripts/benchmark_pinned_h2d.py measures a real
hardware quantity without going through that registry.

Usage:
    python scripts/run_h19_candidate_sweep.py \\
        --store /root/afterimage/store_14b \\
        --out results/h19_candidate_sweep.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from afterimage.bench.prompt_suite import prompt_cases, render_chat_prompt
from scripts.run_bounded_suite import (
    cool_down,
    command_output,
    drop_caches,
    environment_manifest,
    gpu_thermal_snapshot,
    log,
)

DEFAULT_CANDIDATE_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def build_amortization_curve(rows: list[dict]) -> list[dict]:
    """Reduces raw per-cell rows (one per (repeat, case_id, candidate_
    positions)) into the median/min/max latency curve H19 actually
    reports on, PER case_id -- see the module docstring on why absolute
    latency does not compare across prompts, so the curve must not pool
    rows from different-length prompts together. Also reports each
    point's ratio to that same case's N=1 baseline, throughput (N /
    T(N), candidates verified per second of sweep time), and marginal
    cost (T(N) - T(N/2), the actual per-doubling cost the knee shows up
    in more directly than the N=1 ratio alone).
    """
    by_case: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {}).setdefault(
            row["candidate_positions"], []).append(row["verification_sweep_seconds"])

    curves = {}
    for case_id, by_count in by_case.items():
        curve = [
            {"candidate_positions": n,
             "median_seconds": statistics.median(by_count[n]),
             "min_seconds": min(by_count[n]),
             "max_seconds": max(by_count[n]),
             "samples": len(by_count[n])}
            for n in sorted(by_count)
        ]
        # Must be the point whose candidate_positions is literally 1, not
        # just whichever point sorts first -- a sweep that never measured
        # N=1 has no baseline to divide by, and using the smallest
        # measured N instead would silently mislabel that ratio as
        # "relative to N=1" when it is not.
        baseline_point = next((p for p in curve if p["candidate_positions"] == 1), None)
        by_n = {p["candidate_positions"]: p for p in curve}
        for entry in curve:
            n = entry["candidate_positions"]
            entry["throughput_candidates_per_second"] = n / entry["median_seconds"]
            if baseline_point is not None:
                entry["relative_to_n1"] = entry["median_seconds"] / baseline_point["median_seconds"]
            half = by_n.get(n // 2) if n % 2 == 0 else None
            entry["marginal_cost_seconds_vs_half"] = (
                entry["median_seconds"] - half["median_seconds"] if half else None)
        curves[case_id] = curve
    return curves


def find_knee(curve: list[dict], overhead_threshold: float = 1.10) -> int | None:
    """The largest candidate_positions whose relative_to_n1 is still
    within overhead_threshold of the N=1 baseline -- G1's actual gate
    quantity (see docs/SPECULATION_TREE_RESEARCH.md), not a threshold
    that would pass trivially for a streaming-dominated engine (a naive
    "T(64) <= 1.25 x T(1)" gate is nearly guaranteed to pass whenever
    sweep cost is dominated by weight streaming rather than the marginal
    compute of a few more sequence positions, which tells you nothing).
    Returns None if even N=1 is missing (no baseline to compare against).
    """
    by_n = {p["candidate_positions"]: p for p in curve if "relative_to_n1" in p}
    if 1 not in by_n:
        return None
    knee = 1
    for n in sorted(by_n):
        if by_n[n]["relative_to_n1"] <= overhead_threshold:
            knee = n
    return knee


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--store", required=True)
    parser.add_argument(
        "--candidate-counts",
        default=",".join(map(str, DEFAULT_CANDIDATE_COUNTS)),
        help="comma-separated candidate position counts to sweep. SpecExec's own "
             "ablations go to 2048/4096 for a strong drafter, but this project's "
             "own methodology review is explicit: do not start there -- this "
             "sweep is what tells you where to stop. Order here is irrelevant: "
             "the actual dispatch order is randomized per repeat, see --seed.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="repeated measurements per candidate count, so the "
                             "result reports a median/spread instead of one "
                             "unreplicated number (default 3, this project's "
                             "usual quick-screen repeat count).")
    parser.add_argument("--case-ids", default=None,
                        help="comma-separated evaluation case ids to measure, so "
                             "the knee can be checked across more than one prompt "
                             "length (a single-prompt measurement risks fitting "
                             "G1's threshold to that one prompt's own length). "
                             "Default uses the first TWO evaluation cases; pass "
                             "exactly one id to deliberately run single-prompt.")
    parser.add_argument("--warmup-candidate-count", type=int, default=8,
                        help="an UNTIMED warmup sweep at this candidate count, run "
                             "once before the timed loop begins, so lazy CUDA/"
                             "Triton kernel compilation lands here instead of "
                             "uniquely inflating whichever candidate count "
                             "happens to be measured first -- this is exactly "
                             "the bias run_bounded_suite.py's own warmup-tokens "
                             "mechanism exists to absorb elsewhere in this project.")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for randomizing candidate-count dispatch order "
                             "within each repeat, saved to the result so a given "
                             "run's exact order is reproducible. Randomizing (not "
                             "always sweeping 1,2,4,...,1024 in the same order) "
                             "breaks the correlation between candidate size and "
                             "measurement position -- both thermal state (a GPU "
                             "that has been running longer tends to be hotter) "
                             "and cool_down()'s OWN duration (which depends on "
                             "how much heat the PREVIOUS cell left behind) would "
                             "otherwise confound candidate count with when in the "
                             "campaign it happened to run.")
    parser.add_argument("--vram-budget-gb", type=float, default=None,
                        help="EngineConfig's vram_budget_gb, so the amortization "
                             "curve is measured under the SAME residency regime "
                             "this project would actually deploy, not an "
                             "unconfigured default. Default (unset) uses "
                             "EngineConfig's own default.")
    parser.add_argument("--ram-budget-gb", type=float, default=None,
                        help="EngineConfig's ram_budget_gb, paired with "
                             "--vram-budget-gb. The candidate-amortization curve "
                             "can differ meaningfully between a minimum-memory "
                             "regime and a resident/fast regime, since the ratio "
                             "of weight-movement cost to compute cost changes "
                             "with how much is already resident.")
    parser.add_argument("--cooldown-seconds", type=float, default=15.0)
    parser.add_argument("--cooldown-max-temp-c", type=float, default=75.0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-dirty-tree", action="store_true",
        help="proceed even with uncommitted changes; the result is recorded but "
             "not reproducible from its git_commit alone")
    args = parser.parse_args()

    try:
        candidate_counts = [int(part.strip()) for part in args.candidate_counts.split(",")
                           if part.strip()]
    except ValueError:
        parser.error("--candidate-counts must be a comma-separated list of integers")
    if not candidate_counts or any(n < 1 for n in candidate_counts):
        parser.error("--candidate-counts values must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if 1 not in candidate_counts:
        parser.error("--candidate-counts must include 1 (the N=1 baseline every "
                     "other point is reported relative to)")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this measurement")

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    dirty = command_output(["git", "-C", str(repo_root), "status", "--short"])
    if dirty and not args.allow_dirty_tree:
        raise RuntimeError(
            "refusing to run with uncommitted changes (git status --short is "
            "non-empty): a result's git_commit only reproduces the code that "
            "produced it if the tree was clean. Commit or stash first, or pass "
            "--allow-dirty-tree for a deliberately non-reproducible local run.\n"
            + dirty)

    out = pathlib.Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("refusing to overwrite immutable result: %s" % out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=True)
    all_cases = prompt_cases("evaluation")
    if args.case_ids:
        requested = [part.strip() for part in args.case_ids.split(",") if part.strip()]
        by_id = {case.id: case for case in all_cases}
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            parser.error("unknown evaluation cases: %s" % ", ".join(unknown))
        cases = [by_id[case_id] for case_id in requested]
    else:
        cases = list(all_cases[:2])

    config_kwargs = {}
    if args.vram_budget_gb is not None:
        config_kwargs["vram_budget_gb"] = args.vram_budget_gb
    if args.ram_budget_gb is not None:
        config_kwargs["ram_budget_gb"] = args.ram_budget_gb
    config = EngineConfig(**config_kwargs)

    log("model %s" % args.model)
    log("candidate counts: %s" % candidate_counts)
    log("cases: %s" % [case.id for case in cases])
    log("vram_budget_gb=%r ram_budget_gb=%r" %
       (config.vram_budget_gb, config.ram_budget_gb))

    engine = StreamingLosslessModel(args.model, args.store, device="cuda", config=config)
    rows = []
    prompt_input_tokens = {}
    rng = random.Random(args.seed)
    try:
        for case in cases:
            prompt = render_chat_prompt(tokenizer, case)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            prompt_input_tokens[case.id] = int(ids.shape[1])
            log("\ncase %s (%d input tokens)" % (case.id, ids.shape[1]))

            log("  warmup sweep (candidates=%d, untimed)" % args.warmup_candidate_count)
            engine.measure_candidate_sweep_latency(ids, [args.warmup_candidate_count])
            torch.cuda.synchronize()

            for repeat in range(args.repeats):
                order = list(candidate_counts)
                rng.shuffle(order)
                log("  REPEAT %d/%d  order=%s" % (repeat + 1, args.repeats, order))
                for n in order:
                    cooldown = cool_down(args.cooldown_seconds, args.cooldown_max_temp_c)
                    cache = drop_caches()
                    measured = engine.measure_candidate_sweep_latency(ids, [n])[0]
                    row = {
                        "repeat": repeat, "case_id": case.id,
                        "cache_drop_succeeded": cache[0], "cache_drop_error": cache[1],
                        "gpu_thermal": gpu_thermal_snapshot(), **cooldown, **measured,
                    }
                    rows.append(row)
                    log("    positions=%-5d  sweep=%.3fs  io=%.3fs  decode=%.3fs  "
                        "compute=%.3fs  bytes_read=%.3e" %
                        (n, measured["verification_sweep_seconds"], measured["io_seconds"],
                         measured["decode_seconds"], measured["compute_seconds"],
                         measured["bytes_read"]))
    finally:
        engine.close()

    curves = build_amortization_curve(rows)
    knees = {case_id: find_knee(curve) for case_id, curve in curves.items()}

    result = {
        "schema_version": 2,
        "kind": "h19_candidate_amortization_sweep",
        "hypothesis": "h19-candidate-amortization",
        "exploratory": True,
        "evidence_level": "L1_mechanism_screen",
        "model": args.model,
        "store": args.store,
        "case_ids": [case.id for case in cases],
        "prompt_input_tokens": prompt_input_tokens,
        "candidate_counts_requested": candidate_counts,
        "repeats": args.repeats,
        "warmup_candidate_count": args.warmup_candidate_count,
        "seed": args.seed,
        "vram_budget_gb": config.vram_budget_gb,
        "ram_budget_gb": config.ram_budget_gb,
        "cooldown_seconds": args.cooldown_seconds,
        "cooldown_max_temp_c": args.cooldown_max_temp_c,
        "environment": environment_manifest(repo_root, tokenizer,
                                            store=pathlib.Path(args.store)),
        "reproducible_from_commit": not bool(dirty),
        "rows": rows,
        "candidate_amortization_curves_by_case": curves,
        "candidate_parallelism_knee_by_case": knees,
        "gate": "G1: N_free (the knee) must be >= 32 and stable within 2x across "
               "every measured case -- see docs/SPECULATION_TREE_RESEARCH.md. A "
               "knee fitted to a single prompt length is not yet trusted; run "
               "with --case-ids covering at least two different prompt lengths.",
        "completed_at_unix": time.time(),
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    log("\nwrote %s" % out)
    log("candidate_parallelism_knee_by_case: %s" % knees)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
