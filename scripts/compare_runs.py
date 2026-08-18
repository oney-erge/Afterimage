#!/usr/bin/env python3
"""Compares two baseline_*.json runs: token-identity, perplexity delta, and
the memory columns of VALIDATION_PLAN.md #2.

This is the S2 instrument -- the token-identity test that must return
100.000% for any config claiming losslessness.

Usage:
    python scripts/compare_runs.py REFERENCE.json CANDIDATE.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from afterimage.bench.accuracy import token_identity_rate


def gb(v):
    return f"{v/1e9:.2f} GB" if v else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--require-lossless", action="store_true",
                    help="exit non-zero unless token identity is exactly 100%")
    args = ap.parse_args()

    ref = json.loads(pathlib.Path(args.reference).read_text())
    cand = json.loads(pathlib.Path(args.candidate).read_text())

    print("=" * 68)
    print(f"REFERENCE: {ref['model']}  [{ref['dtype']}]")
    print(f"CANDIDATE: {cand['model']}  [{cand['dtype']}]")
    print("=" * 68)

    print("\n--- MEMORY ---")
    print(f"{'':22} {'reference':>14} {'candidate':>14}")
    print(f"{'checkpoint on disk':22} {gb(ref.get('checkpoint_bytes')):>14} {gb(cand.get('checkpoint_bytes')):>14}")
    print(f"{'torch peak VRAM':22} {gb(ref.get('torch_peak_vram_bytes')):>14} {gb(cand.get('torch_peak_vram_bytes')):>14}")
    r_smi = ref.get("smi_delta_gb")
    c_smi = cand.get("smi_delta_gb")
    print(f"{'nvidia-smi delta':22} {f'{r_smi:.2f} GB' if r_smi else 'n/a':>14} {f'{c_smi:.2f} GB' if c_smi else 'n/a':>14}")
    print(f"{'host RSS peak':22} {gb(ref.get('host_rss_peak_bytes')):>14} {gb(cand.get('host_rss_peak_bytes')):>14}")

    if r_smi and c_smi:
        print(f"\n  VRAM reduction: {(1 - c_smi / r_smi) * 100:+.1f}%  "
              f"({r_smi:.2f} GB -> {c_smi:.2f} GB)")

    print("\n--- QUALITY ---")
    r_ppl, c_ppl = ref.get("perplexity"), cand.get("perplexity")
    if r_ppl and c_ppl:
        print(f"  perplexity : {r_ppl:.4f} -> {c_ppl:.4f}  "
              f"({(c_ppl - r_ppl) / r_ppl * 100:+.3f}%)")

    r_tok = ref.get("generated_token_ids")
    c_tok = cand.get("generated_token_ids")
    lossless = None
    if r_tok and c_tok:
        result = token_identity_rate(r_tok, c_tok)
        lossless = result.is_lossless
        print(f"  token identity : {result.token_rate * 100:.3f}%  "
              f"({result.n_tokens_identical}/{result.n_tokens_compared} tokens)")
        print(f"  prompts fully identical : {result.n_prompts_fully_identical}/{result.n_prompts}")
        if result.first_divergence_positions:
            pos = result.first_divergence_positions
            print(f"  first divergence position: min={min(pos)} median="
                  f"{sorted(pos)[len(pos)//2]} max={max(pos)}")
        print(f"\n  LOSSLESS: {'YES' if lossless else 'NO'}")

    print("\n--- SPEED ---")
    r_tps, c_tps = ref.get("generation_tok_per_s"), cand.get("generation_tok_per_s")
    if r_tps and c_tps:
        print(f"  tok/s : {r_tps:.1f} -> {c_tps:.1f}  ({c_tps / r_tps:.2f}x)")

    if args.require_lossless and lossless is not True:
        print("\nFAIL: --require-lossless was set but token identity is not 100%")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
