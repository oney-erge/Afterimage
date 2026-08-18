#!/usr/bin/env python3
"""Head-to-head: every compression scheme on the SAME real layers, same
activations, same metric -- so compression ratio and output error are
directly comparable.

Answers two questions at once:
  1. How does Afterimage compare to what it competes with (quantization,
     and AirLLM-style streaming which does no compression at all)?
  2. Do the proposed fixes (activation-weighted SVD; low-rank + quantized
     residual) actually reduce the 60-96% error PHASE0_RESULTS.md measured?

Usage (inside the WSL venv):
    python -u scripts/run_shootout.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from afterimage.probe.approximations import (
    activation_weighted_svd,
    activation_weighted_svd_bytes,
    full_bytes,
    lowrank_plus_quantized_residual,
    lowrank_plus_quantized_residual_bytes,
    pca_projection,
    pca_projection_bytes,
    quantize_grouped,
    quantize_grouped_bytes,
    quantize_grouped_with_outliers,
    quantize_grouped_with_outliers_bytes,
    quantize_uniform,
    quantize_uniform_bytes,
    relative_output_error,
)
from afterimage.probe.hooks import ActivationCapture
from afterimage.probe.workloads import FOCUSED_CODE, LONG_FORM_PROSE, MULTI_TURN_CHAT


def log(m: str) -> None:
    print(m, flush=True)


def pick_layers(model, n_depths=6):
    layers = model.model.layers
    n = len(layers)
    depths = sorted({int(round(i * (n - 1) / (n_depths - 1))) for i in range(n_depths)})
    return [f"model.layers.{d}.mlp.down_proj" for d in depths]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--n-depths", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}  model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    model.eval()

    target_layers = pick_layers(model, args.n_depths)
    log(f"layers: {target_layers}")

    texts = FOCUSED_CODE + MULTI_TURN_CHAT + LONG_FORM_PROSE
    enc = tok(texts, padding="longest", truncation=True, max_length=args.max_len,
              return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with ActivationCapture(model, layer_names=target_layers) as cap:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, use_cache=False)

    # Schemes are chosen to span a comparable compression range so the
    # comparison is about error-at-given-compression, not about who was
    # allowed to use more memory.
    schemes = [
        ("PCA projection r=128  (ORIGINAL Afterimage)",
         lambda W, X: pca_projection(W, X, 128),
         lambda W: pca_projection_bytes(W, 128)),
        ("PCA projection r=256  (ORIGINAL Afterimage)",
         lambda W, X: pca_projection(W, X, 256),
         lambda W: pca_projection_bytes(W, 256)),
        ("Act-weighted SVD r=128  (FIX 1)",
         lambda W, X: activation_weighted_svd(W, X, 128),
         lambda W: activation_weighted_svd_bytes(W, 128)),
        ("Act-weighted SVD r=256  (FIX 1)",
         lambda W, X: activation_weighted_svd(W, X, 256),
         lambda W: activation_weighted_svd_bytes(W, 256)),
        ("Quant per-row 2-bit  (weak baseline)",
         lambda W, X: quantize_uniform(W, 2),
         lambda W: quantize_uniform_bytes(W, 2)),
        ("Quant per-row 4-bit  (weak baseline)",
         lambda W, X: quantize_uniform(W, 4),
         lambda W: quantize_uniform_bytes(W, 4)),
        ("Quant GROUPED-64 2-bit  (real competitor)",
         lambda W, X: quantize_grouped(W, 2, 64),
         lambda W: quantize_grouped_bytes(W, 2, 64)),
        ("Quant GROUPED-64 3-bit  (real competitor)",
         lambda W, X: quantize_grouped(W, 3, 64),
         lambda W: quantize_grouped_bytes(W, 3, 64)),
        ("Quant GROUPED-64 4-bit  (real ~ Q4_K_M)",
         lambda W, X: quantize_grouped(W, 4, 64),
         lambda W: quantize_grouped_bytes(W, 4, 64)),
        ("Quant GROUPED-64 8-bit  (real ~ Q8_0)",
         lambda W, X: quantize_grouped(W, 8, 64),
         lambda W: quantize_grouped_bytes(W, 8, 64)),
        ("Quant GROUPED-64 3-bit + 0.1% outliers fp16",
         lambda W, X: quantize_grouped_with_outliers(W, 3, 64, 0.001),
         lambda W: quantize_grouped_with_outliers_bytes(W, 3, 64, 0.001)),
        ("Quant GROUPED-64 2-bit + 0.1% outliers fp16",
         lambda W, X: quantize_grouped_with_outliers(W, 2, 64, 0.001),
         lambda W: quantize_grouped_with_outliers_bytes(W, 2, 64, 0.001)),
        ("LowRank r=64 + 2-bit residual  (FIX 2)",
         lambda W, X: lowrank_plus_quantized_residual(W, X, 64, 2),
         lambda W: lowrank_plus_quantized_residual_bytes(W, 64, 2)),
        ("LowRank r=128 + 2-bit residual  (FIX 2)",
         lambda W, X: lowrank_plus_quantized_residual(W, X, 128, 2),
         lambda W: lowrank_plus_quantized_residual_bytes(W, 128, 2)),
        ("LowRank r=128 + 3-bit residual  (FIX 2)",
         lambda W, X: lowrank_plus_quantized_residual(W, X, 128, 3),
         lambda W: lowrank_plus_quantized_residual_bytes(W, 128, 3)),
        ("LowRank r=128 + 4-bit residual  (FIX 2)",
         lambda W, X: lowrank_plus_quantized_residual(W, X, 128, 4),
         lambda W: lowrank_plus_quantized_residual_bytes(W, 128, 4)),
    ]

    results = {"model": args.model, "layers": target_layers, "schemes": {}}

    for name, make, size_fn in schemes:
        errs, comps = [], []
        t0 = time.perf_counter()
        for ln in target_layers:
            X = cap.stacked_masked(ln, mask).float()
            W = model.get_submodule(ln).weight.float()
            W_hat = make(W, X)
            errs.append(relative_output_error(W, W_hat, X))
            comps.append(full_bytes(W) / size_fn(W))
        mean_err = sum(errs) / len(errs)
        mean_comp = sum(comps) / len(comps)
        results["schemes"][name] = {
            "mean_relative_output_error": mean_err,
            "mean_compression_x": mean_comp,
            "per_layer_error": errs,
        }
        log(f"  {name:46s} {mean_comp:6.1f}x  err={mean_err*100:6.2f}%  "
            f"({time.perf_counter()-t0:.1f}s)")

    out = pathlib.Path(args.out) if args.out else (
        pathlib.Path.home() / "afterimage" / "results" / "shootout.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
