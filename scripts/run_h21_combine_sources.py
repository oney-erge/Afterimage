#!/usr/bin/env python3
"""H21 Stage D: join a target corpus with two already-scored candidate
sources (Primary and Scout, from scripts/run_h21_score_source.py) into
the actual H21 analysis: coverage/rescue-recall, equal-total-budget
comparisons, and sustained rescue depth.

No GPU work happens in this script at all -- everything here is pure
Python/JSON joining plus the CPU statistics in afterimage/runtime/
speculation_oracle.py. This is deliberately the ONLY stage that combines
two sources, so scripts/run_h21_score_source.py never has to know about
any source but its own.

Usage:
    python scripts/run_h21_combine_sources.py \\
        --corpus results/h21_target_corpus.json \\
        --primary-scores results/h21_scores_primary.json \\
        --scout-scores results/h21_scores_scout.json \\
        --out results/h21_analysis.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from afterimage.runtime.speculation_oracle import (
    approximate_js_divergence_from_topk,
    compute_equal_budget_metrics,
    compute_oracle_coverage_stats,
    compute_sustained_rescue_depth,
)

SCHEMA_VERSION = 2


def _scores_by_case(payload: dict) -> dict[str, dict]:
    return {entry["case_id"]: entry for entry in payload["scores"]}


def combine(corpus_payload: dict, primary_payload: dict, scout_payload: dict) -> list[dict]:
    primary_by_case = _scores_by_case(primary_payload)
    scout_by_case = _scores_by_case(scout_payload)
    traces = []
    for entry in corpus_payload["corpus"]:
        case_id = entry["case_id"]
        if case_id not in primary_by_case:
            raise ValueError("primary scores are missing case_id=%r" % case_id)
        if case_id not in scout_by_case:
            raise ValueError("scout scores are missing case_id=%r" % case_id)
        p_rows = primary_by_case[case_id]["rows"]
        s_rows = scout_by_case[case_id]["rows"]
        target_tokens = entry["generated_token_ids"]
        if not (len(p_rows) == len(s_rows) == len(target_tokens)):
            raise ValueError(
                "case_id=%r has mismatched position counts: target=%d "
                "primary=%d scout=%d -- all three must come from the SAME "
                "target corpus" % (case_id, len(target_tokens), len(p_rows), len(s_rows)))
        rows = []
        for i, target_token in enumerate(target_tokens):
            p_row, s_row = p_rows[i], s_rows[i]
            rows.append({
                "position": i,
                "target_token": target_token,
                "target_rank_under_primary": p_row["rank"],
                "target_rank_under_scout": s_row["rank"],
                "primary_topk": p_row["topk_ids"],
                "scout_topk": s_row["topk_ids"],
                "primary_entropy": p_row["entropy"],
                "primary_margin": p_row["margin"],
                "scout_entropy": s_row["entropy"],
                "scout_margin": s_row["margin"],
                "approx_js_divergence": approximate_js_divergence_from_topk(
                    p_row["topk_ids"], p_row["topk_probs"],
                    s_row["topk_ids"], s_row["topk_probs"]),
            })
        traces.append({"case_id": case_id, "rows": rows})
    return traces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--primary-scores", required=True)
    parser.add_argument("--scout-scores", required=True)
    parser.add_argument("--coverage-k-values", default="1,2,4,8,16,32,64",
                        help="rank-based coverage is unbounded by --stored-top-k -- "
                             "any k here works as long as it does not exceed the "
                             "smaller of the two sources' --stored-top-k when used "
                             "for the topk-ID-based metrics below (Jaccard, "
                             "equal-budget)")
    parser.add_argument("--equal-budget-total", type=int, default=16,
                        help="total candidate-slot budget for the equal-budget "
                             "comparison; requires both sources' --stored-top-k "
                             "(from run_h21_score_source.py) to be >= this value")
    parser.add_argument("--equal-budget-scout-slots", default="0,4,8,12",
                        help="how many of the total budget's slots to give Scout, "
                             "swept; the rest go to Primary at each point")
    parser.add_argument("--rescue-depth-k", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        coverage_k_values = [int(part.strip()) for part in args.coverage_k_values.split(",")
                             if part.strip()]
    except ValueError:
        parser.error("--coverage-k-values must be a comma-separated list of integers")
    try:
        scout_slots = [int(part.strip()) for part in args.equal_budget_scout_slots.split(",")
                       if part.strip()]
    except ValueError:
        parser.error("--equal-budget-scout-slots must be a comma-separated list of integers")
    if args.equal_budget_total < 1:
        parser.error("--equal-budget-total must be positive")
    if args.rescue_depth_k < 1:
        parser.error("--rescue-depth-k must be positive")

    out = pathlib.Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("refusing to overwrite immutable result: %s" % out)
    out.parent.mkdir(parents=True, exist_ok=True)

    corpus_payload = json.loads(pathlib.Path(args.corpus).resolve().read_text(encoding="utf-8"))
    primary_payload = json.loads(
        pathlib.Path(args.primary_scores).resolve().read_text(encoding="utf-8"))
    scout_payload = json.loads(
        pathlib.Path(args.scout_scores).resolve().read_text(encoding="utf-8"))

    for payload, path in ((primary_payload, args.primary_scores),
                          (scout_payload, args.scout_scores)):
        if payload.get("corpus_target_model") != corpus_payload["target_model"]:
            raise ValueError(
                "%s was scored against target_model=%r, but --corpus is for "
                "target_model=%r -- these must come from the same corpus" %
                (path, payload.get("corpus_target_model"), corpus_payload["target_model"]))

    traces = combine(corpus_payload, primary_payload, scout_payload)
    all_rows = [row for trace in traces for row in trace["rows"]]

    coverage = compute_oracle_coverage_stats(all_rows, coverage_k_values)
    equal_budget = compute_equal_budget_metrics(all_rows, args.equal_budget_total, scout_slots)
    rescue_depth = compute_sustained_rescue_depth(traces, args.rescue_depth_k)

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "h21_combined_analysis",
        "hypothesis": "h21-multi-source-oracle",
        "exploratory": True,
        "evidence_level": "L1_mechanism_screen",
        "target_model": corpus_payload["target_model"],
        "target_revision": corpus_payload.get("target_revision"),
        "decoding_mode": corpus_payload.get("decoding_mode", "greedy"),
        "primary_model": primary_payload["source_model"],
        "primary_revision": primary_payload.get("source_revision"),
        "scout_model": scout_payload["source_model"],
        "scout_revision": scout_payload.get("source_revision"),
        "case_ids": [trace["case_id"] for trace in traces],
        "coverage_k_values": coverage_k_values,
        "equal_budget_total": args.equal_budget_total,
        "equal_budget_scout_slots": scout_slots,
        "rescue_depth_k": args.rescue_depth_k,
        "traces": traces,
        "coverage_stats": coverage,
        "equal_budget_metrics": equal_budget,
        "sustained_rescue_depth": rescue_depth,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote %s" % out)
    print("Read coverage_stats.top_k[k].conditional_rescue_recall for raw rescue "
         "recall, but equal_budget_metrics.points[s].equal_budget_union_gain for "
         "the corrected G3 question: is trading Primary slots for Scout slots at a "
         "FIXED total budget actually worth it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
