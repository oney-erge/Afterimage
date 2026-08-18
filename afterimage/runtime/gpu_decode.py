"""Triton GPU port of the ChunkedBitReader algorithm in huffman_chunked.py
(LOSSLESS_ENGINE.md Phase A -- the kill-switch test).

One Triton program per chunk. Every program performs exactly `chunk_size`
decode iterations (a compile-time constant, per huffman_chunked.py's design
note on why fixed symbol-count chunks matter for this). The bit-reader
refill is unrolled to a fixed 2 steps rather than a data-dependent loop --
provably sufficient, since max_bits <= 16 and each refill adds 8 bits, so at
most 2 refills are ever needed to go from 0 valid bits to >= max_bits.

Uses `tl.where` for the refill's conditional update rather than an `if`
branch, matching Triton's native style for per-lane conditional state
(this kernel runs one scalar "lane" per chunk, but the same idiom applies).
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from .huffman_chunked import ChunkedEncoded


@triton.jit
def _huffman_decode_kernel(
    packed_ptr, sym_lut_ptr, len_lut_ptr, offsets_ptr, out_ptr,
    chunk_size: tl.constexpr,
    max_bits: tl.constexpr,
):
    chunk_id = tl.program_id(0)
    base = tl.load(offsets_ptr + chunk_id)

    buf = tl.zeros([1], dtype=tl.uint32)
    nbits = tl.zeros([1], dtype=tl.int32)
    byte_pos = tl.zeros([1], dtype=tl.int32)

    mask = (1 << max_bits) - 1

    for i in range(chunk_size):
        for _ in range(2):
            need = nbits < max_bits
            b = tl.load(packed_ptr + base + byte_pos, mask=need, other=0).to(tl.uint32)
            shift = (24 - nbits).to(tl.uint32)
            new_buf = buf | (b << shift)
            buf = tl.where(need, new_buf, buf)
            nbits = tl.where(need, nbits + 8, nbits)
            byte_pos = tl.where(need, byte_pos + 1, byte_pos)

        window = ((buf >> (32 - max_bits)) & mask).to(tl.int32)
        sym = tl.load(sym_lut_ptr + window)
        clen = tl.load(len_lut_ptr + window).to(tl.int32)

        out_offset = chunk_id * chunk_size + i + tl.zeros([1], dtype=tl.int32)
        tl.store(out_ptr + out_offset, sym)

        buf = buf << clen.to(tl.uint32)
        nbits = nbits - clen


def decode_gpu(enc: ChunkedEncoded, device: str = "cuda") -> torch.Tensor:
    packed_t = torch.from_numpy(enc.packed).to(device=device, dtype=torch.uint8)
    sym_lut_t = torch.from_numpy(enc.sym_lut.astype(np.int32)).to(device=device)
    len_lut_t = torch.from_numpy(enc.len_lut.astype(np.int32)).to(device=device)
    offsets_t = torch.from_numpy(enc.chunk_offsets.astype(np.int32)).to(device=device)
    out_t = torch.zeros(enc.n_chunks * enc.chunk_size, dtype=torch.int32, device=device)

    grid = (enc.n_chunks,)
    _huffman_decode_kernel[grid](
        packed_t, sym_lut_t, len_lut_t, offsets_t, out_t,
        chunk_size=enc.chunk_size,
        max_bits=enc.max_bits,
    )
    torch.cuda.synchronize()
    return out_t[: enc.n_symbols]
