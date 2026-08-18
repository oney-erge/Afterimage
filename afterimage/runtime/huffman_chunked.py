"""Chunked Huffman encoding for parallel GPU decode (LOSSLESS_ENGINE.md Phase A).

The codec in huffman.py packs a whole tensor into ONE bitstream, which is
fine for a single-threaded decoder but cannot be decoded in parallel --
symbol N's position in the bitstream depends on the bit-lengths of all N-1
symbols before it. Splitting the tensor into fixed-size chunks and encoding
each chunk to its own byte-aligned bitstream removes that dependency
*across* chunks: many chunks can be decoded simultaneously, each on its own
GPU thread/program, using one SHARED Huffman table built from the whole
tensor's statistics (so compression ratio is unaffected by chunking; only
parallelism changes).

Chunk size is a fixed SYMBOL count, not a byte count, so every GPU program
performs exactly the same number of decode iterations -- a compile-time
constant for the kernel, which is what makes the Triton port in
gpu_decode.py tractable (no data-dependent loop bound).

`ChunkedBitReader` below is a pure-Python implementation of the EXACT
algorithm the GPU kernel implements: a 32-bit shift register refilled one
byte at a time. It exists specifically to validate the bit-manipulation
algorithm in isolation before porting it to Triton, so a mismatch during
GPU testing can be attributed to the port, not the algorithm.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from .huffman import build_decode_lut, build_lengths, canonical_codes, pack_bits


@dataclasses.dataclass
class ChunkedEncoded:
    packed: np.ndarray          # uint8, all chunks concatenated
    chunk_offsets: np.ndarray   # int32, byte offset of each chunk's start
    chunk_nbytes: np.ndarray    # int32, byte length of each chunk
    sym_lut: np.ndarray
    len_lut: np.ndarray
    max_bits: int
    chunk_size: int
    n_symbols: int
    shape: tuple[int, ...]

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_offsets)


def encode_chunked(exponents: torch.Tensor, chunk_size: int = 512,
                    max_bits: int = 16) -> ChunkedEncoded:
    flat = exponents.flatten().cpu().numpy().astype(np.int64)
    n = flat.shape[0]

    counts = np.bincount(flat, minlength=256)
    freqs = {int(s): int(c) for s, c in enumerate(counts) if c > 0}
    lengths = build_lengths(freqs, max_bits=max_bits)
    table, codes = canonical_codes(lengths)
    sym_lut, len_lut = build_decode_lut(table, codes)

    # table.max_bits is the ACTUAL bit-width the LUT was built at, which can
    # be smaller than the `max_bits` ceiling passed in above (that parameter
    # only caps package-merge length-limiting; a small or skewed alphabet
    # can need far fewer bits than the ceiling allows). Every downstream
    # consumer must index the LUT at table.max_bits, not the ceiling -- using
    # the ceiling here previously produced an out-of-bounds LUT index
    # whenever the two differed (caught by
    # tests/test_huffman_chunked.py::test_single_chunk_smaller_than_data_
    # still_correct, where the actual table needed only 5 bits against a
    # ceiling of 16, so the true LUT had 32 entries but decode computed a
    # 16-bit window and indexed entry 30402).
    max_bits = table.max_bits

    n_chunks = (n + chunk_size - 1) // chunk_size
    chunk_bytes: list[bytes] = []
    for c in range(n_chunks):
        seg = flat[c * chunk_size: (c + 1) * chunk_size]
        chunk_bytes.append(pack_bits(seg, codes, lengths))

    chunk_nbytes = np.array([len(b) for b in chunk_bytes], dtype=np.int32)
    # Real prefix-sum offsets, NOT a uniform padded stride. An earlier
    # version padded every chunk out to the layer's longest chunk so the
    # kernel could find a chunk with one multiply instead of an offset
    # lookup. Measured on real Qwen2.5-1.5B layers, that padding cost
    # 35-88% of the entire exponent stream (a 2.36M-weight layer stored
    # 5.375 bits/weight against an entropy floor of 2.822) -- because one
    # unlucky chunk full of rare symbols sets the stride for all of them.
    # An extra indexed load per program is far cheaper than that.
    chunk_offsets = np.zeros(n_chunks, dtype=np.int32)
    if n_chunks:
        chunk_offsets[1:] = np.cumsum(chunk_nbytes)[:-1]
    # A few trailing slack bytes: the bit-reader refills to keep max_bits
    # available, so decoding the LAST symbol of the LAST chunk legitimately
    # reads 1-2 bytes past that chunk's own data. With uniform padding that
    # was harmless; with tightly-packed chunks the final chunk would read
    # past the end of the whole buffer. Those over-read bits are never
    # consumed (a canonical prefix code is fully determined by its leading
    # bits, and the LUT replicates each entry across every window sharing
    # that prefix), so their VALUE is irrelevant -- they only have to be
    # addressable. 8 bytes covers the worst case of max_bits=16 plus a
    # partial byte.
    packed = np.concatenate([
        np.frombuffer(b"".join(chunk_bytes), dtype=np.uint8),
        np.zeros(8, dtype=np.uint8),
    ])

    return ChunkedEncoded(
        packed=packed, chunk_offsets=chunk_offsets, chunk_nbytes=chunk_nbytes,
        sym_lut=sym_lut, len_lut=len_lut, max_bits=max_bits,
        chunk_size=chunk_size, n_symbols=n, shape=tuple(exponents.shape),
    )


class ChunkedBitReader:
    """Pure-Python reference for the exact bit-reader algorithm the GPU
    kernel implements: a 32-bit MSB-aligned shift register, refilled one
    byte at a time whenever fewer than max_bits valid bits remain.

    32-bit width is deliberate headroom, not arbitrary: max_bits is capped
    at 16 by huffman.py's LUT construction, refills happen in 8-bit steps,
    so the buffer never needs to hold more than max_bits + 7 <= 23 valid
    bits at once -- safely under 32, with margin, so no shift ever operates
    on a full-width value (which would be undefined behavior in the C-like
    semantics Triton's integer ops follow).
    """

    def __init__(self, data: bytes | np.ndarray):
        self.data = data
        self.byte_pos = 0
        self.buf = 0
        self.nbits = 0
        self.n = len(data)

    def _refill(self, max_bits: int) -> None:
        while self.nbits < max_bits and self.byte_pos < self.n:
            byte = int(self.data[self.byte_pos])
            self.byte_pos += 1
            self.buf |= byte << (24 - self.nbits)  # place at correct position within the 32-bit window
            self.nbits += 8

    def decode_one(self, sym_lut: np.ndarray, len_lut: np.ndarray, max_bits: int) -> int:
        self._refill(max_bits)
        window = (self.buf >> (32 - max_bits)) & ((1 << max_bits) - 1)
        sym = int(sym_lut[window])
        consumed = int(len_lut[window])
        self.buf = (self.buf << consumed) & 0xFFFFFFFF
        self.nbits -= consumed
        return sym


def decode_chunk_reference(chunk_bytes: np.ndarray, sym_lut: np.ndarray, len_lut: np.ndarray,
                            max_bits: int, chunk_size: int) -> np.ndarray:
    reader = ChunkedBitReader(chunk_bytes)
    out = np.zeros(chunk_size, dtype=np.int32)
    for i in range(chunk_size):
        out[i] = reader.decode_one(sym_lut, len_lut, max_bits)
    return out


def decode_chunked_cpu_reference(enc: ChunkedEncoded) -> np.ndarray:
    """Decodes every chunk using ChunkedBitReader (the GPU-equivalent
    algorithm), concatenates, and truncates to the true symbol count (the
    last chunk may be padded with meaningless trailing symbols beyond
    n_symbols, since chunk_size need not divide n_symbols evenly)."""
    out = np.zeros(enc.n_chunks * enc.chunk_size, dtype=np.int32)
    for c in range(enc.n_chunks):
        start = int(enc.chunk_offsets[c])
        end = start + int(enc.chunk_nbytes[c])
        out[c * enc.chunk_size: (c + 1) * enc.chunk_size] = decode_chunk_reference(
            enc.packed[start:end], enc.sym_lut, enc.len_lut, enc.max_bits, enc.chunk_size)
    return out[: enc.n_symbols]
