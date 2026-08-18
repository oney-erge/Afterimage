#!/usr/bin/env python3
"""THE Phase A measurement (LOSSLESS_ENGINE.md #7): is GPU decode fast enough
to hide behind the bus it replaces?

Target: exceed effective bandwidth from the source tier, so decode never
becomes the bottleneck. Measured on this rig: ~20 GB/s from host RAM over
PCIe, ~2 GB/s from NVMe (docs/EXECUTION_PLAN.md Stage A.4).

Measures DECODE throughput only (bytes of *decompressed output* per second),
isolated from disk I/O, H2D transfer, and Python/kernel-launch overhead via
CUDA event timing with proper warmup and synchronization -- not wall-clock
Python timing, which would conflate kernel launch latency with actual GPU
execution time and understate throughput at these tensor sizes.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from afterimage.runtime.gpu_decode import decode_gpu
from afterimage.runtime.huffman_chunked import encode_chunked


def log(m: str) -> None:
    print(m, flush=True)


def bench_one(n_weights: int, chunk_size: int, n_warmup: int = 3, n_iters: int = 10) -> dict:
    torch.manual_seed(0)
    # realistic bf16-like exponent distribution: concentrated on a handful
    # of values, matching the measured ~2.6-bit entropy from the real
    # entropy audit (docs/LOSSLESS_ENGINE.md #2), not uniform random noise
    hot = torch.randperm(256)[:6]
    probs = torch.zeros(256)
    probs[hot] = torch.rand(6)
    probs /= probs.sum()
    exponents = torch.multinomial(probs, n_weights, replacement=True)

    enc = encode_chunked(exponents, chunk_size=chunk_size, max_bits=16)

    from afterimage.runtime.gpu_decode import _huffman_decode_kernel
    import numpy as np

    packed_t = torch.from_numpy(enc.packed).cuda()
    sym_lut_t = torch.from_numpy(enc.sym_lut.astype(np.int32)).cuda()
    len_lut_t = torch.from_numpy(enc.len_lut.astype(np.int32)).cuda()
    offsets_t = torch.from_numpy(enc.chunk_offsets.astype(np.int32)).cuda()
    out_t = torch.zeros(enc.n_chunks * enc.chunk_size, dtype=torch.int32, device="cuda")
    grid = (enc.n_chunks,)

    def launch():
        _huffman_decode_kernel[grid](
            packed_t, sym_lut_t, len_lut_t, offsets_t, out_t,
            chunk_size=enc.chunk_size, max_bits=enc.max_bits,
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

    # The kernel writes each decoded symbol as int32 (4 bytes) -- a
    # convenience for the LUT/store path, NOT the real output size. What
    # Phase A actually cares about is bytes of the ORIGINAL bf16 weight
    # reconstructed per second, which is 2 bytes/weight (sign+exponent+
    # mantissa together), not 4. Using the int32 figure here silently
    # overstated throughput by 2x in the first run of this script -- caught
    # by hand-checking the reported number against what it should mean, not
    # by any test, since nothing asserted the metric's definition. Kept as
    # `raw_kernel_output_bytes` for transparency about what the kernel
    # literally wrote, with `reconstructed_bf16_bytes` as the number that
    # actually answers the Phase A question.
    raw_kernel_output_bytes = enc.n_symbols * 4
    reconstructed_bf16_bytes = enc.n_symbols * 2
    input_bytes = enc.packed.nbytes

    gbps_reconstructed = reconstructed_bf16_bytes / (elapsed_ms / 1000) / 1e9
    gbps_raw_kernel = raw_kernel_output_bytes / (elapsed_ms / 1000) / 1e9
    gbps_input = input_bytes / (elapsed_ms / 1000) / 1e9

    return {
        "n_weights": n_weights,
        "chunk_size": chunk_size,
        "n_chunks": enc.n_chunks,
        "max_bits": enc.max_bits,
        "input_bytes": input_bytes,
        "reconstructed_bf16_bytes": reconstructed_bf16_bytes,
        "compression_ratio": reconstructed_bf16_bytes / input_bytes,
        "elapsed_ms": elapsed_ms,
        "reconstructed_gbps": gbps_reconstructed,
        "raw_kernel_gbps": gbps_raw_kernel,
        "input_gbps": gbps_input,
    }


def main():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log("")
    log(f"{'n_weights':>12} {'chunk':>7} {'chunks':>8} {'ms':>8} "
        f"{'bf16 GB/s':>10} {'in GB/s':>10} {'compress':>9}")

    # sizes spanning one small layer to one large layer of a real model
    configs = [
        (1_000_000, 256),
        (1_000_000, 512),
        (1_000_000, 1024),
        (10_000_000, 512),
        (10_000_000, 1024),
        (27_525_120, 512),   # exact size of Qwen2.5-1.5B's down_proj layer (8960*1536)
        (27_525_120, 1024),
        (27_525_120, 2048),
    ]

    results = []
    for n, cs in configs:
        r = bench_one(n, cs)
        results.append(r)
        log(f"{r['n_weights']:>12,} {r['chunk_size']:>7} {r['n_chunks']:>8,} "
            f"{r['elapsed_ms']:>8.3f} {r['reconstructed_gbps']:>10.2f} {r['input_gbps']:>10.2f} "
            f"{r['compression_ratio']:>8.2f}x")

    log("")
    best = max(results, key=lambda r: r["reconstructed_gbps"])
    log(f"BEST: {best['reconstructed_gbps']:.2f} GB/s reconstructed-bf16 "
        f"(n={best['n_weights']:,}, chunk={best['chunk_size']})")
    log(f"  (raw kernel output, int32 units: {best['raw_kernel_gbps']:.2f} GB/s -- "
        f"NOT the number that answers the Phase A question)")
    log("")
    log("Target thresholds (docs/EXECUTION_PLAN.md measured bandwidths):")
    log(f"  vs host RAM tier (~20 GB/s): {'CLEARS' if best['reconstructed_gbps'] > 20 else 'DOES NOT CLEAR'}")
    log(f"  vs NVMe tier (~2 GB/s):      {'CLEARS' if best['reconstructed_gbps'] > 2 else 'DOES NOT CLEAR'}")


if __name__ == "__main__":
    main()
