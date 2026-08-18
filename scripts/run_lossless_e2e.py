#!/usr/bin/env python3
"""End-to-end lossless compression demonstration on a REAL model
(LOSSLESS_ENGINE.md Phase A/B validation, the actual product claim):

  1. Compress every linear layer's weights (sign+mantissa raw, exponent
     entropy-coded).
  2. Decompress every layer on the GPU.
  3. Verify EVERY weight is bit-exact against the original.
  4. Run a real forward pass with reconstructed weights substituted in,
     and confirm the output logits are bit-exact against the untouched
     model on the same input.
  5. Report the size table: original vs compressed, in GB.

Usage:
    python -u scripts/run_lossless_e2e.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from afterimage.runtime.compressed_store import compress_layer, decompress_layer_gpu


def log(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--chunk-size", type=int, default=1024)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}  model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model.eval()

    linears = {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
    log(f"linear layers: {len(linears)}")

    # --- Step 1-3: compress every layer, decompress on GPU, verify bit-exact ---
    total_orig = total_comp = 0
    t0 = time.perf_counter()
    all_exact = True
    worst_layer = None

    for name, mod in linears.items():
        W = mod.weight.data
        layer = compress_layer(W, chunk_size=args.chunk_size)
        recon = decompress_layer_gpu(layer, device=device)

        exact = torch.equal(recon.view(torch.int16), W.view(torch.int16))
        if not exact:
            all_exact = False
            worst_layer = name
            n_diff = int((recon.view(torch.int16) != W.view(torch.int16)).sum().item())
            log(f"  MISMATCH in {name}: {n_diff} / {W.numel()} weights differ")

        total_orig += layer.original_bytes
        total_comp += layer.compressed_bytes

    compress_time = time.perf_counter() - t0

    log("")
    log("=" * 66)
    log("WEIGHT-LEVEL BIT-EXACTNESS (every layer, every weight)")
    log("=" * 66)
    log(f"  layers checked         : {len(linears)}")
    log(f"  ALL WEIGHTS BIT-EXACT  : {all_exact}")
    if not all_exact:
        log(f"  first mismatch in      : {worst_layer}")
    log(f"  compress+decompress+verify time: {compress_time:.1f}s")
    log("")
    log(f"  original size (all linear layers) : {total_orig/1e9:.3f} GB")
    log(f"  compressed size                   : {total_comp/1e9:.3f} GB")
    log(f"  size fraction                     : {total_comp/total_orig*100:.1f}%")
    log(f"  compression ratio                 : {total_orig/total_comp:.2f}x")

    # --- Step 4: real forward pass, reconstructed weights vs original, bit-exact? ---
    log("")
    log("=" * 66)
    log("END-TO-END FORWARD PASS: reconstructed weights vs original")
    log("=" * 66)

    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        logits_original = model(input_ids=ids, use_cache=False).logits.clone()

    # Substitute every linear layer's weight with its compress->decompress
    # round trip, IN PLACE, then run the exact same forward pass again.
    originals = {}
    for name, mod in linears.items():
        originals[name] = mod.weight.data.clone()
        layer = compress_layer(mod.weight.data, chunk_size=args.chunk_size)
        recon = decompress_layer_gpu(layer, device=device)
        mod.weight.data.copy_(recon)

    with torch.no_grad():
        logits_reconstructed = model(input_ids=ids, use_cache=False).logits.clone()

    # restore, so the model object is left exactly as it was found
    for name, mod in linears.items():
        mod.weight.data.copy_(originals[name])

    logits_match = torch.equal(logits_original.view(torch.int16), logits_reconstructed.view(torch.int16))
    max_abs_diff = (logits_original.float() - logits_reconstructed.float()).abs().max().item()

    log(f"  logits bit-exact vs original: {logits_match}")
    log(f"  max abs logit difference    : {max_abs_diff}")

    with torch.no_grad():
        gen_original = model.generate(ids, max_new_tokens=20, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
        for name, mod in linears.items():
            layer = compress_layer(originals[name], chunk_size=args.chunk_size)
            mod.weight.data.copy_(decompress_layer_gpu(layer, device=device))
        gen_reconstructed = model.generate(ids, max_new_tokens=20, do_sample=False,
                                            pad_token_id=tok.pad_token_id)
        for name, mod in linears.items():
            mod.weight.data.copy_(originals[name])

    text_original = tok.decode(gen_original[0, ids.shape[1]:])
    text_reconstructed = tok.decode(gen_reconstructed[0, ids.shape[1]:])
    tokens_match = torch.equal(gen_original, gen_reconstructed)

    log(f"  generated tokens identical   : {tokens_match}")
    log(f"  original text     : {text_original!r}")
    log(f"  reconstructed text: {text_reconstructed!r}")

    log("")
    log("=" * 66)
    log("SUMMARY")
    log("=" * 66)
    log(f"  {total_orig/1e9:.2f} GB -> {total_comp/1e9:.2f} GB "
        f"({total_orig/total_comp:.2f}x smaller), LOSSLESS: {all_exact and logits_match and tokens_match}")

    return 0 if (all_exact and logits_match and tokens_match) else 1


if __name__ == "__main__":
    raise SystemExit(main())
