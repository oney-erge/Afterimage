#!/usr/bin/env python3
"""Throughput of the vectorized (v2) kernel, same methodology as
run_decode_throughput.py, so the two are directly comparable."""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import triton

from afterimage.runtime.gpu_decode_v2 import _huffman_decode_kernel_v2
from afterimage.runtime.huffman_chunked import encode_chunked


def log(m: str) -> None:
    print(m, flush=True)


def bench_one(n_weights: int, chunk_size: int, block_chunks: int,
              n_warmup: int = 3, n_iters: int = 10) -> dict:
    torch.manual_seed(0)
    hot = torch.randperm(256)[:6]
    probs = torch.zeros(256)
    probs[hot] = torch.rand(6)
    probs /= probs.sum()
    exponents = torch.multinomial(probs, n_weights, replacement=True)

    enc = encode_chunked(exponents, chunk_size=chunk_size, max_bits=16)

    packed_t = torch.from_numpy(enc.packed).cuda()
    sym_lut_t = torch.from_numpy(enc.sym_lut.astype(np.int32)).cuda()
    len_lut_t = torch.from_numpy(enc.len_lut.astype(np.int32)).cuda()
    offsets_t = torch.from_numpy(enc.chunk_offsets.astype(np.int32)).cuda()
    out_t = torch.zeros(enc.n_chunks * enc.chunk_size, dtype=torch.int32, device="cuda")
    grid = (triton.cdiv(enc.n_chunks, block_chunks),)

    def launch():
        _huffman_decode_kernel_v2[grid](
            packed_t, sym_lut_t, len_lut_t, offsets_t, out_t,
            enc.n_chunks,
            chunk_size=enc.chunk_size, max_bits=enc.max_bits, BLOCK_CHUNKS=block_chunks,
        )

    for _ in range(n_warmup):
        launch()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        launch()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / n_iters
    reconstructed_bf16_bytes = enc.n_symbols * 2  # see run_decode_throughput.py's note on this
    gbps = reconstructed_bf16_bytes / (elapsed_ms / 1000) / 1e9

    return {
        "n_weights": n_weights, "chunk_size": chunk_size, "block_chunks": block_chunks,
        "n_chunks": enc.n_chunks, "elapsed_ms": elapsed_ms, "reconstructed_gbps": gbps,
    }


def main():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log("")
    log(f"{'n_weights':>12} {'chunk':>7} {'block':>6} {'ms':>8} {'bf16 GB/s':>10}")

    configs = []
    for n in [1_000_000, 27_525_120]:
        for chunk_size in [512, 1024]:
            for block_chunks in [32, 128, 256, 512]:
                configs.append((n, chunk_size, block_chunks))

    results = []
    for n, cs, bc in configs:
        r = bench_one(n, cs, bc)
        results.append(r)
        log(f"{r['n_weights']:>12,} {r['chunk_size']:>7} {r['block_chunks']:>6} "
            f"{r['elapsed_ms']:>8.3f} {r['reconstructed_gbps']:>10.2f}")

    log("")
    best = max(results, key=lambda r: r["reconstructed_gbps"])
    log(f"BEST: {best['reconstructed_gbps']:.2f} GB/s "
        f"(n={best['n_weights']:,}, chunk={best['chunk_size']}, block={best['block_chunks']})")
    log(f"  vs v1 baseline (2.04 GB/s): {best['reconstructed_gbps']/2.04:.2f}x")
    log("")
    log("Target thresholds:")
    log(f"  vs host RAM tier (~20 GB/s): {'CLEARS' if best['reconstructed_gbps'] > 20 else 'DOES NOT CLEAR'}")
    log(f"  vs NVMe tier (~2 GB/s):      {'CLEARS' if best['reconstructed_gbps'] > 2 else 'DOES NOT CLEAR'}")


if __name__ == "__main__":
    main()
