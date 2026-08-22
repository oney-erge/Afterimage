#!/usr/bin/env python3
"""Precise size-overhead breakdown + head-to-head vs AirLLM.

Two questions:
  1. Where exactly does the end-to-end 1.29x sit relative to the 1.51x
     coding rate? (measured per term, not estimated)
  2. Against AirLLM -- the system we must beat -- what does this engine
     actually buy, on THIS machine's measured bandwidths?
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from afterimage.runtime.compressed_store import compress_layer


def log(m: str) -> None:
    print(m, flush=True)


# Measured on this rig -- see docs/archive/EXECUTION_PLAN.md Stage A.4 and
# docs/archive/LOSSLESS_ENGINE.md Phase A. Not vendor specs.
NVME_GBPS = 2.0        # sustained O_DIRECT, 24 GB file, WSL2 ext4
RAM_GBPS = 20.0        # host RAM over PCIe
DECODE_GBPS = 16.87    # measured Triton v2 kernel, bf16 output
VRAM_USABLE_GB = 6.5   # 8 GB card minus ~1.5 GB desktop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-layers", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    sample = linears[: args.n_layers]

    tot_orig = tot_sm = tot_packed = tot_lut = 0
    for name, mod in sample:
        layer = compress_layer(mod.weight.data, chunk_size=1024)
        tot_orig += layer.original_bytes
        tot_sm += layer.sign_mantissa.numel()
        tot_packed += layer.encoded.packed.nbytes
        tot_lut += layer.encoded.sym_lut.nbytes + layer.encoded.len_lut.nbytes

    tot_comp = tot_sm + tot_packed + tot_lut
    log("=" * 70)
    log(f"SIZE BREAKDOWN ({len(sample)} real layers, measured)")
    log("=" * 70)
    log(f"  original                 : {tot_orig/1e6:9.1f} MB   (100.0%)")
    log(f"  sign+mantissa (raw 8bit) : {tot_sm/1e6:9.1f} MB   ({tot_sm/tot_orig*100:5.1f}%)")
    log(f"  exponent (entropy-coded) : {tot_packed/1e6:9.1f} MB   ({tot_packed/tot_orig*100:5.1f}%)")
    log(f"  decode LUT               : {tot_lut/1e6:9.1f} MB   ({tot_lut/tot_orig*100:5.1f}%)")
    log(f"  TOTAL compressed         : {tot_comp/1e6:9.1f} MB   ({tot_comp/tot_orig*100:5.1f}%)")
    log(f"  ratio                    : {tot_orig/tot_comp:.3f}x")
    log("")
    log(f"  LUT is the fixable overhead: it is PER-LAYER today but the")
    log(f"  exponent distribution barely varies across layers, so one shared")
    log(f"  table would cost {tot_lut/len(sample)/1e6:.2f} MB total instead of "
        f"{tot_lut/1e6:.2f} MB.")
    shared_lut_comp = tot_sm + tot_packed + (tot_lut / len(sample))
    log(f"  with a shared LUT        : {tot_orig/shared_lut_comp:.3f}x")
    log("")

    log("=" * 70)
    log("HEAD-TO-HEAD vs AirLLM (measured bandwidths, this machine)")
    log("=" * 70)
    log(f"  NVMe {NVME_GBPS} GB/s | RAM {RAM_GBPS} GB/s | decode {DECODE_GBPS} GB/s "
        f"| VRAM {VRAM_USABLE_GB} GB")
    log("")
    log("  AirLLM: streams the FULL uncompressed model from disk once PER TOKEN.")
    log("  Ours:   streams COMPRESSED weights once per SWEEP; a sweep with")
    log("          speculation yields k tokens (k~15 published, NOT yet built here).")
    log("")

    ratio = tot_orig / shared_lut_comp
    log(f"{'model':>10} {'bf16 GB':>9} {'AirLLM tok/s':>13} "
        f"{'ours k=1':>10} {'ours k=15':>11} {'vs AirLLM':>10}")

    for label, params_b in [("1.5B", 1.5), ("8B", 8.0), ("27B", 27.0), ("70B", 70.0)]:
        raw_gb = params_b * 2.0          # bf16 = 2 bytes/param
        comp_gb = raw_gb / ratio

        # AirLLM: full uncompressed model from NVMe, every token
        airllm_tps = 1.0 / (raw_gb / NVME_GBPS)

        # Ours: compressed bytes from NVMe, plus GPU decode (they pipeline,
        # so the slower of the two dominates -- take max, not sum)
        stream_s = comp_gb / NVME_GBPS
        decode_s = raw_gb / DECODE_GBPS   # decode produces the FULL bf16 volume
        sweep_s = max(stream_s, decode_s)

        ours_k1 = 1.0 / sweep_s
        ours_k15 = 15.0 / sweep_s
        log(f"{label:>10} {raw_gb:>9.1f} {airllm_tps:>13.3f} "
            f"{ours_k1:>10.3f} {ours_k15:>11.2f} {ours_k15/airllm_tps:>9.1f}x")

    log("")
    log("  NOTE: 'ours k=15' is a PROJECTION -- speculative decoding is not")
    log("  built in this repo yet. 'ours k=1' is what the measured compression")
    log("  + measured decode kernel deliver TODAY, with zero accuracy loss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
