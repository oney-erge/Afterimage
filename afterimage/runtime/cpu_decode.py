"""Multicore CPU Huffman decode.

STATUS: this is NOT on the GPU inference path, and deliberately so.

It was built to test docs/archive/PROPOSAL.md's own H2 (unrelated to the
current H2 hazard-cost stopping hypothesis in docs/RESEARCH_METHODS.md) --
splitting entropy decode across CPU and GPU, on the reasoning that decode
(~13 s/token) and disk I/O
(~14 s/token) were co-bottlenecks while 16 CPU cores sat idle. The isolated
throughput gate PASSED clearly: the numba path below decodes a real 14B
tensor at 1.33 GB/s across 16 threads, matching this engine's own in-situ
GPU decode rate.

Wired into the engine, it made things WORSE -- flat at a 25% CPU share,
then monotonically worse: 0.88x at 50%, 0.72x at 75%, 0.52x at 100%.
`gpu_decode_s` fell exactly as predicted, but `io_s` rose in lockstep
(93 -> 176 s) and wall time never improved. The intended overlap never
happened: CPU decode ran inside the same background thread that does the
disk reads it was supposed to overlap with, so it extended that thread's
own critical path and plausibly starved the read() syscalls of scheduling
time. Full numbers and analysis: docs/RESULTS_LOG.md.

**The engine-side dispatch was removed rather than left as a
default-off footgun.** What remains here is kept for two reasons, not as
dead code:

  1. **CPU-only fallback.** `afterimage doctor` can report a machine with
     no usable GPU, and the Triton decode kernels require CUDA. This is
     the only decoder that runs without a GPU at all, so it is what a
     CPU-only path would be built on.
  2. **A reusable, verified negative result.** The throughput numbers are
     real and the decoder is bit-exact (tests/test_cpu_decode.py). If the
     overlap is ever retried, it needs a genuinely independent decode
     thread -- not one nested inside the I/O reader -- and this module is
     the starting point, with the failure mode already documented.

Both decoders below are bit-exact against
huffman_chunked.decode_chunked_cpu_reference.

Why there are two implementations
----------------------------------
`decode_chunks_vectorized` (numpy, vectorized across chunks, mirroring
gpu_decode_v2's lockstep trick) was the first attempt and is far too slow
to be useful: 0.037 GB/s peak, and it gets WORSE past 4 threads. It is
retained only because its bit-exactness tests also validate the shared
correctness argument about reading past a chunk boundary, and because
"the obvious numpy approach does not work here" is worth recording rather
than rediscovering. `decode_chunks_numba` is the one that clears the gate,
and the only one worth using.
"""
from __future__ import annotations

import concurrent.futures

import numpy as np

from .huffman_chunked import ChunkedEncoded

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def _decode_numba(packed, chunk_offsets, chunk_nbytes, sym_lut, len_lut,
                      max_bits, chunk_size, n_chunks, out):
        """Compiled, per-chunk-bounded decode, parallelized across chunks by
        numba's own thread pool. Unlike decode_chunks_vectorized, this stays
        strictly within each chunk's own [offset, offset+nbytes) byte range
        (exactly mirroring ChunkedBitReader), because a tight compiled loop
        does not pay the redundant-byte-regather cost that made staying in
        bounds cheap to skip in the numpy version -- so it needs no separate
        cross-chunk-boundary correctness argument."""
        mask = (1 << max_bits) - 1
        for c in numba.prange(n_chunks):
            pos = chunk_offsets[c]
            end = pos + chunk_nbytes[c]
            buf = 0
            nbits = 0
            for i in range(chunk_size):
                while nbits < max_bits and pos < end:
                    byte = packed[pos]
                    pos += 1
                    buf |= byte << (24 - nbits)
                    nbits += 8
                window = (buf >> (32 - max_bits)) & mask
                sym = sym_lut[window]
                length = len_lut[window]
                out[c, i] = sym
                buf = (buf << length) & 0xFFFFFFFF
                nbits -= length


def decode_chunks_numba(enc: ChunkedEncoded, chunk_lo: int = 0,
                        chunk_hi: int | None = None) -> np.ndarray:
    """The compiled fallback docs/archive/PROPOSAL.md's own H2 anticipated
    needing if the numpy-vectorized path ("released into a C extension" per
    that doc) came up short. Requires the optional `numba` dependency;
    raises clearly if it's absent rather than silently falling back to a
    slower path."""
    if not _HAS_NUMBA:
        raise RuntimeError("decode_chunks_numba requires the 'numba' package")
    chunk_hi = enc.n_chunks if chunk_hi is None else chunk_hi
    n = chunk_hi - chunk_lo
    out = np.empty((n, enc.chunk_size), dtype=np.uint8)
    _decode_numba(enc.packed, enc.chunk_offsets[chunk_lo:chunk_hi].astype(np.int64),
                 enc.chunk_nbytes[chunk_lo:chunk_hi].astype(np.int64),
                 enc.sym_lut, enc.len_lut.astype(np.int64),
                 enc.max_bits, enc.chunk_size, n, out)
    return out.reshape(-1)


def decode_chunks_vectorized(enc: ChunkedEncoded, chunk_lo: int, chunk_hi: int) -> np.ndarray:
    """Decodes chunks [chunk_lo, chunk_hi) of enc, all in lockstep, and
    returns a flat (chunk_hi - chunk_lo) * enc.chunk_size uint8 array
    (exponents are 8 bits) -- callers truncate to n_symbols themselves,
    matching decode_chunked_cpu_reference's existing convention.
    """
    n = chunk_hi - chunk_lo
    if n <= 0:
        return np.zeros(0, dtype=np.uint8)

    chunk_size = enc.chunk_size
    max_bits = enc.max_bits
    packed = enc.packed
    packed_len = packed.shape[0]
    last_idx = packed_len - 1

    offsets = enc.chunk_offsets[chunk_lo:chunk_hi].astype(np.int64)
    bit_pos = np.zeros(n, dtype=np.int64)  # bits consumed so far, per chunk, relative to its own start

    sym_lut = enc.sym_lut
    len_lut = enc.len_lut.astype(np.int64)

    out = np.empty((n, chunk_size), dtype=np.uint8)
    shift_bits = np.uint64(64 - max_bits)

    for i in range(chunk_size):
        byte0 = offsets + (bit_pos >> 3)
        shift = (bit_pos & 7).astype(np.uint64)

        idx0 = np.minimum(byte0, last_idx)
        idx1 = np.minimum(byte0 + 1, last_idx)
        idx2 = np.minimum(byte0 + 2, last_idx)
        idx3 = np.minimum(byte0 + 3, last_idx)
        idx4 = np.minimum(byte0 + 4, last_idx)

        b0 = packed[idx0].astype(np.uint64)
        b1 = packed[idx1].astype(np.uint64)
        b2 = packed[idx2].astype(np.uint64)
        b3 = packed[idx3].astype(np.uint64)
        b4 = packed[idx4].astype(np.uint64)

        # 5 bytes, MSB-aligned in the top 40 bits of a 64-bit word.
        window = ((b0 << np.uint64(56)) | (b1 << np.uint64(48)) | (b2 << np.uint64(40))
                 | (b3 << np.uint64(32)) | (b4 << np.uint64(24)))
        shifted = window << shift  # numpy uint64 shift truncates past bit 63, as intended
        top_bits = (shifted >> shift_bits).astype(np.int64)

        out[:, i] = sym_lut[top_bits]
        bit_pos += len_lut[top_bits]

    return out.reshape(-1)


def decode_chunks_threaded(enc: ChunkedEncoded, n_threads: int,
                           chunk_lo: int = 0, chunk_hi: int | None = None) -> np.ndarray:
    """Splits [chunk_lo, chunk_hi) into n_threads contiguous ranges and
    decodes them concurrently. Threads, not processes: each range's inner
    loop is numpy array ops, which release the GIL during the actual C-level
    computation, so this gets real parallelism without IPC/pickling cost --
    the same reasoning that makes numpy-heavy code a good fit for
    ThreadPoolExecutor when multiprocessing would only add overhead.
    """
    chunk_hi = enc.n_chunks if chunk_hi is None else chunk_hi
    total = chunk_hi - chunk_lo
    if n_threads <= 1 or total < n_threads:
        return decode_chunks_vectorized(enc, chunk_lo, chunk_hi)

    bounds = np.linspace(chunk_lo, chunk_hi, n_threads + 1).astype(np.int64)
    ranges = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_threads)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        parts = list(ex.map(lambda r: decode_chunks_vectorized(enc, *r), ranges))
    return np.concatenate(parts)
