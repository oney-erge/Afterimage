"""Vectorized version of gpu_decode.py -- BLOCK_CHUNKS independent chunks
decoded together per Triton program, instead of one chunk per program.

Diagnosis from measuring gpu_decode.py's v1 kernel: it achieved only
2.04 GB/s (docs/LOSSLESS_ENGINE.md Phase A), a 10x shortfall against the
~20 GB/s target, despite ~1000x spare compute existing in the offloaded
regime. v1 launches one Triton program per chunk using shape-[1] ("scalar")
tensors throughout -- each program does chunk_size sequential steps with no
internal vectorization, so the GPU's 32-wide SIMD execution is used at
roughly 1/32 of its width. The decode dependency within one chunk is
inherently sequential (Huffman symbol N's bit position depends on symbol
N-1's length) and cannot be removed without a different codec, but nothing
requires only ONE chunk to occupy a warp: BLOCK_CHUNKS independent chunks
can share a program and advance in lockstep, one decoded symbol per chunk
per step, using real vector width instead of scalar-per-program state.
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from .huffman_chunked import ChunkedEncoded


@triton.jit
def _huffman_decode_kernel_v2(
    packed_ptr, sym_lut_ptr, len_lut_ptr, offsets_ptr, out_ptr,
    n_chunks,
    chunk_size: tl.constexpr,
    max_bits: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk_ids = pid * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
    valid = chunk_ids < n_chunks
    # Real per-chunk byte offset, loaded from the prefix-sum array, rather
    # than chunk_id * uniform_stride. huffman_chunked.py dropped uniform
    # padding because it cost 35-88% of the exponent stream on real layers;
    # one extra indexed load here is what buys that back.
    base = tl.load(offsets_ptr + chunk_ids, mask=valid, other=0)

    buf = tl.zeros([BLOCK_CHUNKS], dtype=tl.uint32)
    nbits = tl.zeros([BLOCK_CHUNKS], dtype=tl.int32)
    byte_pos = tl.zeros([BLOCK_CHUNKS], dtype=tl.int32)

    mask = (1 << max_bits) - 1

    for i in range(chunk_size):
        for _ in range(2):
            need = (nbits < max_bits) & valid
            b = tl.load(packed_ptr + base + byte_pos, mask=need, other=0).to(tl.uint32)
            shift = (24 - nbits).to(tl.uint32)
            new_buf = buf | (b << shift)
            buf = tl.where(need, new_buf, buf)
            nbits = tl.where(need, nbits + 8, nbits)
            byte_pos = tl.where(need, byte_pos + 1, byte_pos)

        window = ((buf >> (32 - max_bits)) & mask).to(tl.int32)
        sym = tl.load(sym_lut_ptr + window, mask=valid, other=0)
        clen = tl.load(len_lut_ptr + window, mask=valid, other=0).to(tl.int32)

        out_offset = chunk_ids * chunk_size + i
        tl.store(out_ptr + out_offset, sym.to(tl.uint8), mask=valid)

        buf = buf << clen.to(tl.uint32)
        nbits = nbits - clen


def decode_gpu_v2(enc: ChunkedEncoded, block_chunks: int = 32, device: str = "cuda") -> torch.Tensor:
    """block_chunks must be a power of 2 -- Triton's tl.arange requires it,
    and this is checked here (not left to fail deep inside the Triton
    compiler with a less legible error) since it is a real constraint
    callers need to know about, not an internal implementation detail.

    Default of 32 matches the GPU's native warp width and was empirically
    the fastest configuration measured (16.87 GB/s vs. e.g. 128's
    13.83 GB/s on the same data -- see docs/LOSSLESS_ENGINE.md Phase A),
    consistent with the diagnosis that v1's scalar-per-program design left
    most of the SIMD width idle: 32 is the size where each program exactly
    fills one warp, and larger blocks that span multiple warps measured
    worse, likely from increased register pressure or reduced occupancy.
    """
    if block_chunks & (block_chunks - 1) != 0:
        raise ValueError(f"block_chunks must be a power of 2, got {block_chunks}")

    packed_t = torch.from_numpy(enc.packed).to(device=device, dtype=torch.uint8)
    sym_lut_t = torch.from_numpy(enc.sym_lut.astype(np.int32)).to(device=device)
    len_lut_t = torch.from_numpy(enc.len_lut.astype(np.int32)).to(device=device)
    offsets_t = torch.from_numpy(enc.chunk_offsets.astype(np.int32)).to(device=device)
    # uint8, not int32. Symbols here are float EXPONENT fields, which are
    # 8 bits wide by construction -- storing them 4 bytes each wasted 4x
    # the scratch memory. On a 778M-weight embedding that is 3.1 GB of
    # scratch instead of 778 MB, which is what actually blew a 4 GB VRAM
    # cap (the cap surfaced the waste; it was there all along).
    assert enc.sym_lut.max() <= 255, (
        "decoder output is uint8; symbol alphabet must fit in a byte")
    out_t = torch.zeros(enc.n_chunks * enc.chunk_size, dtype=torch.uint8, device=device)

    grid = (triton.cdiv(enc.n_chunks, block_chunks),)
    _huffman_decode_kernel_v2[grid](
        packed_t, sym_lut_t, len_lut_t, offsets_t, out_t,
        enc.n_chunks,
        chunk_size=enc.chunk_size,
        max_bits=enc.max_bits,
        BLOCK_CHUNKS=block_chunks,
    )
    torch.cuda.synchronize()
    return out_t[: enc.n_symbols]
