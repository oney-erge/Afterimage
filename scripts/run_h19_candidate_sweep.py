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
    """Reduces raw per-cell rows (one per (repeat, candidate_positions))
    into the median/min/max latency curve H19 actually reports on, plus
    each point's ratio to the N=1 baseline -- the number that answers the
    hypothesis: values staying near 1.0 well past N=1 mean candidate
    positions are cheap under this engine's streamed-target regime, the
    SpecExec-motivating result this hypothesis exists to check for (or
    refute) on THIS hardware rather than assume from a different paper's
    GPUs.
    """
    by_count: dict[int, list[float]] = {}
    for row in rows:
        by_count.setdefault(row["candidate_positions"], []).append(
            row["verification_sweep_seconds"])
    curve = [
        {"candidate_positions": n,
         "median_seconds": statistics.median(by_count[n]),
         "min_seconds": min(by_count[n]),
         "max_seconds": max(by_count[n]),
         "samples": len(by_count[n])}
        for n in sorted(by_count)
    ]
    # Must be the point whose candidate_positions is literally 1, not just
    # whichever point sorts first -- a sweep that never measured N=1 has no
    # baseline to divide by, and using the smallest measured N instead
    # would silently mislabel that ratio as "relative to N=1" when it is
    # not.
    baseline_point = next((p for p in curve if p["candidate_positions"] == 1), None)
    if baseline_point is not None:
        baseline = baseline_point["median_seconds"]
        for entry in curve:
            entry["relative_to_n1"] = entry["median_seconds"] / baseline
    return curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--store", required=True)
    parser.add_argument(
        "--candidate-counts",
        default=",".join(map(str, DEFAULT_CANDIDATE_COUNTS)),
        help="comma-separated candidate position counts to sweep, ascending. "
             "SpecExec's own ablations go to 2048/4096 for a strong drafter, "
             "but this project's own methodology review is explicit: do not "
             "start there -- this sweep is what tells you where to stop.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="repeated measurements per candidate count, so the "
                             "result reports a median/spread instead of one "
                             "unreplicated number (default 3, this project's "
                             "usual quick-screen repeat count).")
    parser.add_argument("--case-id", default=None,
                        help="a specific evaluation case id to use as the fixed "
                             "prompt; default uses the first evaluation case. "
                             "The SAME prompt is reused across every candidate "
                             "count in one run -- see the module docstring on "
                             "why absolute latency is not comparable across "
                             "different prompts, only the shape vs. count is.")
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
    cases = prompt_cases("evaluation")
    case = next((c for c in cases if c.id == args.case_id), cases[0]) if args.case_id else cases[0]
    prompt = render_chat_prompt(tokenizer, case)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    log("model %s" % args.model)
    log("case %s (%d input tokens)" % (case.id, ids.shape[1]))
    log("candidate counts: %s" % candidate_counts)

    engine = StreamingLosslessModel(args.model, args.store, device="cuda",
                                    config=EngineConfig())
    rows = []
    try:
        for repeat in range(args.repeats):
            log("\nREPEAT %d/%d" % (repeat + 1, args.repeats))
            for n in candidate_counts:
                cooldown = cool_down(args.cooldown_seconds, args.cooldown_max_temp_c)
                cache = drop_caches()
                measured = engine.measure_candidate_sweep_latency(ids, [n])[0]
                row = {
                    "repeat": repeat, "case_id": case.id,
                    "cache_drop_succeeded": cache[0], "cache_drop_error": cache[1],
                    "gpu_thermal": gpu_thermal_snapshot(), **cooldown, **measured,
                }
                rows.append(row)
                log("  positions=%-5d  sweep=%.3fs  io=%.3fs  decode=%.3fs  "
                    "compute=%.3fs  bytes_read=%.3e" %
                    (n, measured["verification_sweep_seconds"], measured["io_seconds"],
                     measured["decode_seconds"], measured["compute_seconds"],
                     measured["bytes_read"]))
    finally:
        engine.close()

    curve = build_amortization_curve(rows)

    result = {
        "schema_version": 1,
        "kind": "h19_candidate_amortization_sweep",
        "hypothesis": "h19-candidate-amortization",
        "exploratory": True,
        "evidence_level": "L1_mechanism_screen",
        "model": args.model,
        "store": args.store,
        "case_id": case.id,
        "input_tokens": int(ids.shape[1]),
        "candidate_counts_requested": candidate_counts,
        "repeats": args.repeats,
        "cooldown_seconds": args.cooldown_seconds,
        "cooldown_max_temp_c": args.cooldown_max_temp_c,
        "environment": environment_manifest(repo_root, tokenizer,
                                            store=pathlib.Path(args.store)),
        "reproducible_from_commit": not bool(dirty),
        "rows": rows,
        "candidate_amortization_curve": curve,
        "completed_at_unix": time.time(),
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    log("\nwrote %s" % out)
    log("Read candidate_amortization_curve's relative_to_n1 column for the "
        "H19 answer: values that stay near 1.0 well past N=1 mark this "
        "hardware's candidate-parallelism knee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
