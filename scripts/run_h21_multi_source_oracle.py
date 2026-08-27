#!/usr/bin/env python3
"""H21 (Multi-Source Oracle Headroom): before integrating a second drafter
("Scout") into any runtime mechanism, does it actually cover target
continuations the primary drafter misses?

Collects real per-position traces -- no live speculative execution, no new
runtime mechanism -- then answers with afterimage.runtime.
speculation_oracle.compute_oracle_coverage_stats: primary_coverage,
scout_coverage, union_coverage, and the number this hypothesis is actually
about, conditional_rescue_recall = P(target rank <= k under Scout | target
NOT covered by Primary at k). Per docs/SPECULATION_TREE_RESEARCH.md's own
framing: a scout that looks WORSE than the primary on raw accuracy can
still be worth having if it rescues a meaningful fraction of the primary's
specific misses -- and a scout with high standalone accuracy but near-zero
rescue recall is not worth integrating regardless of how good it looks
alone.

Method (per prompt): generate the TARGET's own greedy continuation once
(the reference trajectory). At every position along it, run Primary and
Scout on that same real prefix (teacher-forced on the target's actual
path, not on Primary's or Scout's own guesses) and record where the
target's real next token ranked under each. Each forward pass reprocesses
the growing prefix from scratch (no KV cache) -- this is an offline
analysis tool, not a latency benchmark, so correctness and simplicity beat
speed here.

--scout-model defaults to a same-tokenizer, different-training-depth
model (Qwen3-1.7B against a Qwen3-0.6B primary) -- docs/
SPECULATION_TREE_RESEARCH.md's own "lowest risk first" ordering. Pass a
different --scout-model to test another candidate; a scout that does not
share the target's tokenizer/vocabulary makes top-k/rank comparisons
meaningless as written here (see Universal Assisted Decoding for the
translation this script does not attempt).

Usage:
    python scripts/run_h21_multi_source_oracle.py \\
        --store /root/afterimage/store_14b \\
        --out results/h21_multi_source_oracle.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from afterimage.bench.prompt_suite import prompt_cases, render_chat_prompt
from afterimage.runtime.speculation_oracle import compute_oracle_coverage_stats, rank_of_token
from scripts.run_bounded_suite import (
    command_output,
    drop_caches,
    environment_manifest,
    log,
)


def _topk_ids(logits: torch.Tensor, k: int) -> list[int]:
    return torch.topk(logits, k).indices.tolist()


def collect_trace(engine, primary, scout, tokenizer, prompt: str,
                  max_new_tokens: int, top_k: int) -> list[dict]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(engine.device)
    with torch.no_grad():
        sequence = engine.generate_greedy(ids, max_new_tokens)
    generated = sequence[0, ids.shape[1]:].tolist()

    rows = []
    prefix = ids
    for position, target_token in enumerate(generated):
        with torch.no_grad():
            primary_logits = primary(input_ids=prefix).logits[0, -1, :]
            scout_logits = scout(input_ids=prefix).logits[0, -1, :]
        primary_logits_np = primary_logits.float().cpu().numpy()
        scout_logits_np = scout_logits.float().cpu().numpy()
        rows.append({
            "position": position,
            "target_token": target_token,
            "target_rank_under_primary": rank_of_token(primary_logits_np, target_token),
            "target_rank_under_scout": rank_of_token(scout_logits_np, target_token),
            "primary_topk": _topk_ids(primary_logits, top_k),
            "scout_topk": _topk_ids(scout_logits, top_k),
        })
        prefix = torch.cat(
            [prefix, torch.tensor([[target_token]], device=prefix.device)], dim=1)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--store", required=True)
    parser.add_argument("--primary-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--scout-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-k-values", default="1,2,4,8,16",
                        help="comma-separated k values to report coverage at; "
                             "must not exceed --top-k, since only that many "
                             "candidates are actually recorded per position")
    parser.add_argument("--case-ids", default=None,
                        help="comma-separated evaluation case ids; default is all")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-dirty-tree", action="store_true",
        help="proceed even with uncommitted changes; the result is recorded but "
             "not reproducible from its git_commit alone")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        top_k_values = [int(part.strip()) for part in args.top_k_values.split(",")
                        if part.strip()]
    except ValueError:
        parser.error("--top-k-values must be a comma-separated list of integers")
    if any(k > args.top_k for k in top_k_values):
        parser.error("--top-k-values cannot exceed --top-k (only that many "
                     "candidates are recorded per position)")

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

    from transformers import AutoTokenizer

    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel, load_draft_model

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, fix_mistral_regex=True)
    cases = prompt_cases("evaluation")
    if args.case_ids:
        requested = [part.strip() for part in args.case_ids.split(",") if part.strip()]
        by_id = {case.id: case for case in cases}
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            parser.error("unknown evaluation cases: %s" % ", ".join(unknown))
        cases = tuple(by_id[case_id] for case_id in requested)

    log("target %s" % args.target_model)
    log("primary %s" % args.primary_model)
    log("scout %s" % args.scout_model)

    engine = StreamingLosslessModel(args.target_model, args.store, device="cuda",
                                    config=EngineConfig())
    primary = load_draft_model(args.primary_model, device="cuda")
    scout = load_draft_model(args.scout_model, device="cuda")

    traces = []
    try:
        for case in cases:
            prompt = render_chat_prompt(tokenizer, case)
            drop_caches()
            log("\ncase %s" % case.id)
            rows = collect_trace(engine, primary, scout, tokenizer, prompt,
                                 args.max_new_tokens, args.top_k)
            traces.append({"case_id": case.id, "rows": rows})
            log("  %d positions collected" % len(rows))
    finally:
        del primary, scout
        engine.close()
        torch.cuda.empty_cache()

    all_rows = [row for trace in traces for row in trace["rows"]]
    coverage = compute_oracle_coverage_stats(all_rows, top_k_values)

    result = {
        "schema_version": 1,
        "kind": "h21_multi_source_oracle",
        "hypothesis": "h21-multi-source-oracle",
        "exploratory": True,
        "evidence_level": "L1_mechanism_screen",
        "target_model": args.target_model,
        "primary_model": args.primary_model,
        "scout_model": args.scout_model,
        "store": args.store,
        "max_new_tokens": args.max_new_tokens,
        "top_k_recorded": args.top_k,
        "top_k_values_reported": top_k_values,
        "case_ids": [case.id for case in cases],
        "environment": environment_manifest(repo_root, tokenizer,
                                            store=pathlib.Path(args.store)),
        "reproducible_from_commit": not bool(dirty),
        "traces": traces,
        "coverage_stats": coverage,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    log("\nwrote %s" % out)
    log("Read coverage_stats.top_k[k].conditional_rescue_recall for the H21 "
        "answer: a scout worth integrating rescues a meaningful fraction of "
        "the primary's specific misses, not just scores well on its own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
