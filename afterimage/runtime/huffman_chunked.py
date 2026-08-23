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

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @numba.njit(cache=True)
    def _pack_chunked_kernel(flat, code_lut, len_lut, chunk_size, out, chunk_nbytes):
        """One 64-bit accumulator, filled MSB-first and flushed a byte at a
        time -- the compiled replacement for calling pack_bits() once per
        chunk (see the profiling note on encode_chunked below). nbits stays
        under 8 between symbols and each code is at most 16 bits, so
        nbits + ln never approaches 64 and the shift is always safe.

        Chunks are packed in order because each one's start position depends
        on every earlier chunk's real (variable) byte length -- unlike GPU
        decode, this can't be parallelized across chunks without a separate
        prefix-sum pass first, and a single core already clears 300+ M
        symbols/s, well past what one compression run needs."""
        n = flat.shape[0]
        n_chunks = (n + chunk_size - 1) // chunk_size
        pos = 0
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, n)
            acc = np.uint64(0)
            nbits = 0
            cstart = pos
            for i in range(start, end):
                s = flat[i]
                code = np.uint64(code_lut[s])
                ln = len_lut[s]
                acc |= code << np.uint64(64 - nbits - ln)
                nbits += ln
                while nbits >= 8:
                    out[pos] = np.uint8(acc >> np.uint64(56))
                    pos += 1
                    acc = acc << np.uint64(8)
                    nbits -= 8
            if nbits > 0:
                out[pos] = np.uint8(acc >> np.uint64(56))
                pos += 1
            chunk_nbytes[c] = pos - cstart
        return pos


def _pack_all_chunks_numba(flat: np.ndarray, codes: dict[int, int], lengths: dict[int, int],
                           chunk_size: int, max_bits: int) -> tuple[np.ndarray, np.ndarray]:
    n = flat.shape[0]
    n_chunks = (n + chunk_size - 1) // chunk_size
    code_lut = np.zeros(256, dtype=np.uint64)
    len_lut = np.zeros(256, dtype=np.int64)
    for s, c in codes.items():
        code_lut[s] = c
        len_lut[s] = lengths[s]

    # Worst case every symbol costs max_bits bits, plus up to 7 wasted
    # padding bits per chunk from each chunk's own byte alignment; +8 is
    # general slack. Sliced down to the real length the kernel reports.
    upper_bound = (n * max_bits + 7) // 8 + n_chunks + 8
    out = np.zeros(upper_bound, dtype=np.uint8)
    chunk_nbytes = np.zeros(n_chunks, dtype=np.int32)
    total = _pack_chunked_kernel(flat, code_lut, len_lut, chunk_size, out, chunk_nbytes)
    return out[:total].copy(), chunk_nbytes


def _pack_all_chunks_python(flat: np.ndarray, codes: dict[int, int], lengths: dict[int, int],
                            chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = flat.shape[0]
    n_chunks = (n + chunk_size - 1) // chunk_size
    chunk_bytes = [pack_bits(flat[c * chunk_size:(c + 1) * chunk_size], codes, lengths)
                   for c in range(n_chunks)]
    chunk_nbytes = np.array([len(b) for b in chunk_bytes], dtype=np.int32)
    packed = np.frombuffer(b"".join(chunk_bytes), dtype=np.uint8)
    return packed, chunk_nbytes


def _pack_all_chunks(flat: np.ndarray, codes: dict[int, int], lengths: dict[int, int],
                     chunk_size: int, max_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Packs every chunk's bitstream and returns (packed_bytes, chunk_nbytes),
    with no trailing slack tail -- encode_chunked appends that.

    Numba path measured 19.2x over the per-chunk pack_bits() loop on a
    2048x2048 bf16 tensor (17.7 M/s -> 339.5 M/s symbols/s), byte-identical
    output verified against it. The per-chunk loop calling pack_bits()
    rebuilds its own code/length LUTs from a dict and runs ~15 numpy ops on
    a chunk_size-element array EVERY chunk -- 14.5M calls for a 14B model --
    so interpreter and numpy per-call overhead dominates actual work. A
    whole-tensor numpy vectorization (hoist the LUTs, expand every symbol's
    bits at once) was tried first and measured SLOWER (0.63x) than the
    current loop: expanding one int64 per bit blows past cache. That rules
    out "vectorize it harder" and is why this needs a compiled loop, not
    better numpy.
    """
    if _HAS_NUMBA:
        return _pack_all_chunks_numba(flat, codes, lengths, chunk_size, max_bits)
    return _pack_all_chunks_python(flat, codes, lengths, chunk_size)


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
    # bincount and the code lookup accept uint8 symbols directly. Casting a
    # large exponent field to int64 made an unnecessary 8x full-field copy
    # during offline compression (5.37 GB for a 27B-class output head).
    flat = exponents.flatten().cpu().numpy()
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
    packed_body, chunk_nbytes = _pack_all_chunks(flat, codes, lengths, chunk_size, max_bits)
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
    packed = np.concatenate([packed_body, np.zeros(8, dtype=np.uint8)])

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
