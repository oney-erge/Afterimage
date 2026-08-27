#!/usr/bin/env python3
"""H21 Stage A: collect the TARGET's own greedy continuations once, with
no drafter loaded at all.

This is the expensive stage (the 14B streamed target) and the one that
must never be re-run just to compare a different candidate source.
Previously (see git history for the retired run_h21_multi_source_oracle.py)
Primary, Scout, and the streamed target all had to be resident on the GPU
together for the whole collection, which meant every new candidate model
worth trying required regenerating the target trajectory from scratch too.
Splitting collection into stages fixes that: this script's output (a
target corpus: prompts plus their real greedy continuations) is reused by
however many scripts/run_h21_score_source.py runs follow, each loading
exactly one candidate model.

Usage:
    python scripts/run_h21_collect_target_corpus.py \\
        --store /root/afterimage/store_14b \\
        --out results/h21_target_corpus.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from afterimage.bench.prompt_suite import prompt_cases, render_chat_prompt
from scripts.run_bounded_suite import (
    command_output,
    drop_caches,
    environment_manifest,
    log,
    model_revision,
)

SCHEMA_VERSION = 2  # SpeculationTrace v2: see docs/SPECULATION_TREE_RESEARCH.md's
                    # trace-format section for the full field-by-field contract
                    # this and the two scripts downstream of it jointly implement.


def collect_target_corpus(engine, tokenizer, cases, max_new_tokens: int) -> list[dict]:
    corpus = []
    for case in cases:
        prompt = render_chat_prompt(tokenizer, case)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(engine.device)
        drop_caches()
        log("case %s" % case.id)
        with torch.no_grad():
            sequence = engine.generate_greedy(prompt_ids, max_new_tokens)
        generated = sequence[0, prompt_ids.shape[1]:].tolist()
        corpus.append({
            "case_id": case.id,
            "prompt": prompt,
            "prompt_token_ids": prompt_ids[0].tolist(),
            "generated_token_ids": generated,
        })
        log("  %d tokens generated" % len(generated))
    return corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--store", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--case-ids", default=None,
                        help="comma-separated evaluation case ids; default is all")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-dirty-tree", action="store_true",
        help="proceed even with uncommitted changes; the result is recorded but "
             "not reproducible from its git_commit alone")
    args = parser.parse_args()

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
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

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
    engine = StreamingLosslessModel(args.target_model, args.store, device="cuda",
                                    config=EngineConfig())
    try:
        corpus = collect_target_corpus(engine, tokenizer, cases, args.max_new_tokens)
    finally:
        engine.close()
        torch.cuda.empty_cache()

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "h21_target_corpus",
        "target_model": args.target_model,
        "target_revision": model_revision(args.target_model),
        "store": args.store,
        "max_new_tokens": args.max_new_tokens,
        "decoding_mode": "greedy",
        "case_ids": [case.id for case in cases],
        "environment": environment_manifest(repo_root, tokenizer,
                                            store=pathlib.Path(args.store)),
        "reproducible_from_commit": not bool(dirty),
        "corpus": corpus,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    log("\nwrote %s" % out)
    log("Next: score one or more candidate sources against this corpus with "
        "scripts/run_h21_score_source.py --corpus %s --model <hf_id> --role primary|scout" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
