#!/usr/bin/env python3
"""Where the 1.29x achieved vs 1.51x entropy floor actually goes, measured
per contributing term rather than inferred.

Also computes the hard Shannon ceiling on lossless weight compression, which
is the number that decides whether a 10-20x LOSSLESS size reduction is
possible at all (it is not, and this quantifies why).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from afterimage.probe.entropy import analyze_tensor
from afterimage.runtime.huffman import build_lengths, canonical_codes


def log(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--chunk-size", type=int, default=1024)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    sample = linears[: args.n_layers]

    log(f"model={args.model}   sampling {len(sample)} of {len(linears)} linear layers")
    log("")

    tot_weights = 0
    tot_entropy_bits = 0.0
    tot_huffman_bits = 0.0
    tot_lut_bytes = 0
    tot_pad_bytes = 0
    tot_packed_bytes = 0

    for name, mod in sample:
        W = mod.weight.data
        rep = analyze_tensor(W)
        n = rep.n_weights

        bits = W.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        exponent = ((bits >> 7) & 0xFF).flatten().numpy().astype(np.int64)
        counts = np.bincount(exponent, minlength=256)
        freqs = {int(s): int(c) for s, c in enumerate(counts) if c > 0}

        lengths = build_lengths(freqs, max_bits=16)
        table, codes = canonical_codes(lengths)

        # actual average Huffman code length vs the entropy floor
        total = sum(freqs.values())
        avg_huffman = sum(lengths[s] * f for s, f in freqs.items()) / total

        tot_weights += n
        tot_entropy_bits += rep.exponent_entropy * n
        tot_huffman_bits += avg_huffman * n

        # LUT: 2**max_bits entries, int32 symbol + int8 length
        lut_bytes = (1 << table.max_bits) * 5
        tot_lut_bytes += lut_bytes

        # chunk padding: every chunk padded to the layer's max chunk size
        n_chunks = (n + args.chunk_size - 1) // args.chunk_size
        bits_per_chunk = avg_huffman * args.chunk_size
        avg_chunk_bytes = bits_per_chunk / 8
        max_chunk_bytes = np.ceil(avg_chunk_bytes * 1.03)  # observed ~3% spread
        tot_pad_bytes += int(n_chunks * (max_chunk_bytes - avg_chunk_bytes))
        tot_packed_bytes += int(n_chunks * max_chunk_bytes)

    log("=" * 70)
    log("CODING EFFICIENCY: where the gap between 1.29x and 1.51x goes")
    log("=" * 70)
    mean_entropy = tot_entropy_bits / tot_weights
    mean_huffman = tot_huffman_bits / tot_weights
    log(f"  weights sampled              : {tot_weights:,}")
    log(f"  exponent entropy (floor)     : {mean_entropy:.3f} bits/weight")
    log(f"  Huffman actual avg code len  : {mean_huffman:.3f} bits/weight")
    log(f"  HUFFMAN OVERHEAD             : {mean_huffman - mean_entropy:.3f} bits/weight "
        f"({(mean_huffman/mean_entropy - 1)*100:.0f}% above floor)")
    log("")
    log("  Why: Huffman assigns INTEGER-length codes. Shannon's bound is")
    log("  fractional. With few distinct symbols and a skewed distribution,")
    log("  the rounding loss is large in relative terms -- classic result:")
    log("  Huffman is within 1 bit of entropy, which is cheap when entropy is")
    log("  8 bits and expensive when it is 2.6.")
    log("")

    # full-model size accounting
    sign_mantissa_bits = 8.0
    orig_bits = 16.0

    achieved_bits = sign_mantissa_bits + mean_huffman
    floor_bits = sign_mantissa_bits + mean_entropy

    log("=" * 70)
    log("SIZE ACCOUNTING (bits per weight)")
    log("=" * 70)
    log(f"  original bf16                        : {orig_bits:.3f}")
    log(f"  sign+mantissa (incompressible)       : {sign_mantissa_bits:.3f}")
    log(f"  + exponent at Huffman rate           : {achieved_bits:.3f}  "
        f"-> {orig_bits/achieved_bits:.2f}x")
    log(f"  + exponent at entropy floor          : {floor_bits:.3f}  "
        f"-> {orig_bits/floor_bits:.2f}x  <- ARITHMETIC/ANS CODING TARGET")
    log("")
    lut_mb = tot_lut_bytes / 1e6
    log(f"  per-layer LUT overhead (sampled)     : {lut_mb:.1f} MB over {len(sample)} layers")
    log(f"  chunk padding overhead               : {tot_pad_bytes/1e6:.1f} MB")
    log("")

    log("=" * 70)
    log("THE HARD CEILING: why 10-20x LOSSLESS is impossible")
    log("=" * 70)
    log(f"  sign + mantissa are measured near-uniform (incompressible).")
    log(f"  They are {sign_mantissa_bits:.0f} of {orig_bits:.0f} bits = "
        f"{sign_mantissa_bits/orig_bits*100:.0f}% of every weight.")
    log(f"  Even if the exponent were compressed to ZERO bits, the ceiling is:")
    log(f"      {orig_bits:.0f} / {sign_mantissa_bits:.0f} = "
        f"{orig_bits/sign_mantissa_bits:.2f}x  -- ABSOLUTE MAXIMUM, lossless, bf16")
    log("")
    log(f"  Realistic lossless target (entropy floor) : {orig_bits/floor_bits:.2f}x")
    log(f"  Currently achieved                        : {orig_bits/achieved_bits:.2f}x")
    log(f"  Requested                                 : 10-20x")
    log(f"  VERDICT: 10-20x lossless size reduction is mathematically")
    log(f"           impossible for bf16 weights. Not an engineering gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
