#!/usr/bin/env python3
"""Measures the lossless-compression headroom of a real model's weights.

Answers: if we never trade accuracy, how many bytes can we still avoid
moving? This is the entropy floor of the actual checkpoint, measured, not
quoted from a paper.

Usage:
    python -u scripts/run_entropy_audit.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from afterimage.probe.entropy import analyze_tensor


def log(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    log(f"loading {args.model} as {args.dtype} (CPU -- this is a weight audit, no inference)")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.eval()

    total_raw = total_comp = 0
    per_layer = {}
    exp_entropies = []

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        W = mod.weight.data
        rep = analyze_tensor(W)
        raw = rep.n_weights * rep.original_bits_per_weight / 8
        comp = rep.n_weights * rep.compressed_bits_per_weight / 8
        total_raw += raw
        total_comp += comp
        exp_entropies.append(rep.exponent_entropy)
        per_layer[name] = {
            "n_weights": rep.n_weights,
            "exponent_entropy": rep.exponent_entropy,
            "mantissa_entropy": rep.mantissa_entropy,
            "sign_entropy": rep.sign_entropy,
            "bits_per_weight": rep.compressed_bits_per_weight,
            "size_fraction": rep.size_fraction,
            "raw_bytes": raw,
            "compressed_bytes": comp,
        }

    n = len(per_layer)
    frac = total_comp / total_raw
    log("")
    log("=" * 66)
    log(f"LOSSLESS COMPRESSION AUDIT  ({args.model}, {args.dtype})")
    log("=" * 66)
    log(f"  linear layers audited : {n}")
    log(f"  raw size              : {total_raw/1e9:.3f} GB")
    log(f"  entropy floor         : {total_comp/1e9:.3f} GB")
    log(f"  size fraction         : {frac*100:.1f}%  ->  {1/frac:.2f}x compression")
    log(f"  bytes saved           : {(total_raw-total_comp)/1e9:.3f} GB "
        f"({(1-frac)*100:.1f}%)")
    log("")
    sample = next(iter(per_layer.values()))
    log(f"  mean exponent entropy : {sum(exp_entropies)/len(exp_entropies):.3f} bits "
        f"(field is {'8' if args.dtype=='bf16' else '5'} bits wide)")
    log(f"  mantissa entropy      : {sample['mantissa_entropy']:.3f} bits "
        f"(near-uniform = incompressible, as expected)")

    spread = sorted(per_layer.items(), key=lambda kv: kv[1]["size_fraction"])
    log("")
    log("  most compressible layers:")
    for k, v in spread[:3]:
        log(f"    {k:52s} {v['size_fraction']*100:.1f}%")
    log("  least compressible layers:")
    for k, v in spread[-3:]:
        log(f"    {k:52s} {v['size_fraction']*100:.1f}%")

    out = pathlib.Path(args.out) if args.out else (
        pathlib.Path.home() / "afterimage" / "results" /
        f"entropy_{args.model.replace('/','__')}_{args.dtype}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "dtype": args.dtype,
        "total_raw_bytes": total_raw, "total_compressed_bytes": total_comp,
        "size_fraction": frac, "per_layer": per_layer,
    }, indent=2))
    log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
