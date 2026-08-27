#!/usr/bin/env python3
"""H21 Stage B/C: teacher-force ONE candidate source model against an
already-collected target corpus (scripts/run_h21_collect_target_corpus.py)
and record where the target's real next token ranked under it, plus the
cheap online features (entropy, top-1/top-2 margin) H22b's baselines need.

The SAME script scores Primary, Scout, or any other candidate -- "role"
is just a label attached to the output for scripts/run_h21_combine_
sources.py to key on, not a code path. This is what makes comparing a new
candidate source (a different Qwen size, a different architecture
entirely) cheap: load one model, run this once, done. The 14B target
never has to be touched again.

Usage:
    python scripts/run_h21_score_source.py \\
        --corpus results/h21_target_corpus.json \\
        --model Qwen/Qwen3-0.6B --role primary \\
        --out results/h21_scores_primary.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from afterimage.runtime.speculation_oracle import (
    entropy_of_logits,
    margin_of_logits,
    rank_of_token,
)
from scripts.run_bounded_suite import command_output, environment_manifest, log, model_revision

SCHEMA_VERSION = 2


def score_source(model, tokenizer, corpus: list[dict], stored_top_k: int) -> list[dict]:
    device = next(model.parameters()).device
    scores = []
    for entry in corpus:
        prefix = torch.tensor([entry["prompt_token_ids"]], device=device)
        rows = []
        for position, target_token in enumerate(entry["generated_token_ids"]):
            with torch.no_grad():
                logits = model(input_ids=prefix).logits[0, -1, :]
            logits_np = logits.float().cpu().numpy()
            topk = torch.topk(logits, stored_top_k)
            # softmax over the FULL vocabulary, then read off just the
            # stored top-k entries' probabilities -- these are each
            # token's real probability mass, not a renormalization over
            # only the stored subset, so their sum also tells a reader
            # how much of the true distribution the stored top-k actually
            # captured (see approximate_js_divergence_from_topk's
            # docstring for why this matters downstream).
            full_probs = torch.softmax(logits, dim=0)
            rows.append({
                "position": position,
                "rank": rank_of_token(logits_np, target_token),
                "entropy": entropy_of_logits(logits_np),
                "margin": margin_of_logits(logits_np),
                "topk_ids": topk.indices.tolist(),
                "topk_probs": full_probs[topk.indices].tolist(),
            })
            prefix = torch.cat(
                [prefix, torch.tensor([[target_token]], device=device)], dim=1)
        scores.append({"case_id": entry["case_id"], "rows": rows})
        log("case %s: %d positions scored" % (entry["case_id"], len(rows)))
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True,
                        help="path to an h21_target_corpus.json from "
                             "run_h21_collect_target_corpus.py")
    parser.add_argument("--model", required=True)
    parser.add_argument("--role", default="candidate",
                        help="a label for this source (e.g. primary, scout) that "
                             "run_h21_combine_sources.py keys results on; purely "
                             "informational, does not change scoring behavior")
    parser.add_argument("--stored-top-k", type=int, default=16,
                        help="how many top token IDs to store per position. "
                             "Coverage/rescue-recall metrics only need the target's "
                             "RANK (unbounded k, since rank<=k is a direct "
                             "comparison), but equal-total-budget metrics and "
                             "Jaccard overlap need the actual stored IDs -- set "
                             "this to at least the largest total_budget you plan "
                             "to evaluate in run_h21_combine_sources.py, default "
                             "16 covers the project's own default budget sweep")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-dirty-tree", action="store_true",
        help="proceed even with uncommitted changes; the result is recorded but "
             "not reproducible from its git_commit alone")
    args = parser.parse_args()

    if args.stored_top_k < 1:
        parser.error("--stored-top-k must be positive")

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

    corpus_path = pathlib.Path(args.corpus).resolve()
    corpus_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    if corpus_payload.get("kind") != "h21_target_corpus":
        parser.error("--corpus does not look like a run_h21_collect_target_corpus.py "
                     "output (missing kind=h21_target_corpus)")

    out = pathlib.Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("refusing to overwrite immutable result: %s" % out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    from afterimage.runtime.streaming_engine import load_draft_model

    tokenizer = AutoTokenizer.from_pretrained(corpus_payload["target_model"],
                                              fix_mistral_regex=True)
    log("role %s: %s" % (args.role, args.model))
    model = load_draft_model(args.model, device="cuda")
    try:
        scores = score_source(model, tokenizer, corpus_payload["corpus"], args.stored_top_k)
    finally:
        del model
        torch.cuda.empty_cache()

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "h21_source_scores",
        "corpus_source": str(corpus_path),
        "corpus_target_model": corpus_payload["target_model"],
        "corpus_target_revision": corpus_payload.get("target_revision"),
        "source_model": args.model,
        "source_revision": model_revision(args.model),
        "role": args.role,
        "stored_top_k": args.stored_top_k,
        "environment": environment_manifest(repo_root, tokenizer),
        "reproducible_from_commit": not bool(dirty),
        "scores": scores,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    log("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
