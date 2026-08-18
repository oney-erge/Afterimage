"""Canonical Huffman codec for the exponent field of float weights
(LOSSLESS_ENGINE.md Phase A).

Canonical Huffman rather than a plain Huffman tree, for a concrete reason:
canonical codes are fully described by a per-symbol *code length* array (no
tree pointers to store or walk), and codes of the same length are guaranteed
consecutive integers -- both properties this module leans on to build a flat
lookup table a GPU kernel can index directly, instead of a decoder that has
to branch down a tree one bit at a time (the exact thing the literature
flags as unsuitable for GPU: "sequential, bit-by-bit traversal of a Huffman
tree").

Length-limited to `max_bits` so the resulting LUT has a bounded, known size
(2**max_bits entries) -- required for a GPU kernel that indexes it in O(1).
If the natural Huffman lengths would exceed max_bits, they are limited via
the standard package-merge algorithm (Larmore & Hirschberg 1990), which
finds the optimal code under a maximum-length constraint rather than merely
truncating and hoping it still sums to a valid code (naive truncation does
not, in general, produce a valid prefix code).
"""
from __future__ import annotations

import dataclasses
import heapq

import numpy as np
import torch


@dataclasses.dataclass
class HuffmanTable:
    symbols: list[int]          # symbol values, index-aligned with lengths
    lengths: list[int]          # code length in bits per symbol, canonical order
    max_bits: int

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)


def _natural_lengths(freqs: dict[int, int]) -> dict[int, int]:
    """Standard Huffman tree construction; returns code length per symbol.
    Symbols with zero frequency are omitted."""
    heap = [[f, i, [sym]] for i, (sym, f) in enumerate(freqs.items()) if f > 0]
    heapq.heapify(heap)
    if len(heap) == 1:
        # single-symbol alphabet: give it a 1-bit code, canonical convention
        return {heap[0][2][0]: 1}

    length = {sym: 0 for sym in freqs if freqs[sym] > 0}
    counter = len(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for sym in lo[2]:
            length[sym] += 1
        for sym in hi[2]:
            length[sym] += 1
        heapq.heappush(heap, [lo[0] + hi[0], counter, lo[2] + hi[2]])
        counter += 1
    return length


def _package_merge_limit(freqs: dict[int, int], max_bits: int) -> dict[int, int]:
    """Length-limited optimal code lengths via package-merge. Only invoked
    when natural Huffman lengths exceed max_bits -- expensive relative to
    plain Huffman, but only runs once per tensor at encode time, off the
    critical path entirely."""
    items = [(f, s) for s, f in freqs.items() if f > 0]
    items.sort()
    n = len(items)
    if n <= 1:
        return {items[0][1]: 1} if items else {}

    # package-merge over `max_bits` levels: level k holds "packages" that are
    # 2^k leaves wide; merging packages level-by-level and taking the 2(n-1)
    # cheapest at the final level yields optimal lengths <= max_bits.
    lists: list[list[tuple[int, list[int]]]] = []
    for k in range(max_bits):
        leaves = [(f, [s]) for f, s in items]
        if k == 0:
            lists.append(leaves)
            continue
        prev = lists[-1]
        packages = []
        for i in range(0, len(prev) - 1, 2):
            packages.append((prev[i][0] + prev[i + 1][0], prev[i][1] + prev[i + 1][1]))
        merged = sorted(leaves + packages)
        lists.append(merged)

    chosen = sorted(lists[-1])[: 2 * (n - 1)]
    length = {s: 0 for _, s in items}
    for _, syms in chosen:
        for s in syms:
            length[s] += 1
    return length


def build_lengths(freqs: dict[int, int], max_bits: int = 16) -> dict[int, int]:
    lengths = _natural_lengths(freqs)
    if not lengths or max(lengths.values()) <= max_bits:
        return lengths
    return _package_merge_limit(freqs, max_bits)


def canonical_codes(lengths: dict[int, int]) -> tuple[HuffmanTable, dict[int, int]]:
    """Assigns canonical codes: symbols sorted by (length, symbol value),
    codes assigned as consecutive integers, incrementing and left-shifting
    by one bit whenever length increases. Returns the table plus a
    symbol -> integer code dict for the CPU-side encoder."""
    order = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    symbols = [s for s, _ in order]
    lens = [l for _, l in order]

    code = 0
    prev_len = lens[0] if lens else 0
    codes: dict[int, int] = {}
    for sym, ln in zip(symbols, lens):
        code <<= (ln - prev_len)
        codes[sym] = code
        code += 1
        prev_len = ln

    table = HuffmanTable(symbols=symbols, lengths=lens, max_bits=max(lens) if lens else 0)
    return table, codes


def build_decode_lut(table: HuffmanTable, codes: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Flat LUT of size 2**max_bits: for every possible max_bits-wide window
    of upcoming bits, precomputes which symbol it decodes to and how many
    bits that symbol's actual code consumed. A short code's entry is
    replicated across every window sharing its prefix -- the standard
    canonical-Huffman table-decode trick, and the reason lookups are O(1)
    regardless of code length.
    """
    size = 1 << table.max_bits
    sym_lut = np.zeros(size, dtype=np.int32)
    len_lut = np.zeros(size, dtype=np.int8)

    for sym, ln in zip(table.symbols, table.lengths):
        code = codes[sym]
        # left-align this code's bits at the top of the max_bits window, then
        # every combination of the remaining (max_bits - ln) low bits shares
        # this same decode result
        base = code << (table.max_bits - ln)
        span = 1 << (table.max_bits - ln)
        sym_lut[base: base + span] = sym
        len_lut[base: base + span] = ln

    return sym_lut, len_lut


def pack_bits(symbol_stream: np.ndarray, codes: dict[int, int], lengths: dict[int, int]) -> bytes:
    """MSB-first bitstream packing, vectorized.

    An earlier version looped over every symbol in pure Python. That is
    O(n) interpreter overhead on a stream with one symbol PER WEIGHT --
    measured at 467 s to compress a 1.5B model, which extrapolates to over
    an hour for a 14B one and made the whole comparison impractical. This
    version expands every code to its bits with numpy index arithmetic and
    packs once, moving the loop into C.

    The bit expansion: each symbol i contributes lengths[i] bits. np.repeat
    gives, for every output bit, which symbol it came from; subtracting that
    symbol's start offset gives the bit's index WITHIN its code; and the
    code's MSB-first bit j is (code >> (L-1-j)) & 1.
    """
    if len(symbol_stream) == 0:
        return b""

    max_sym = int(symbol_stream.max())
    code_lut = np.zeros(max_sym + 1, dtype=np.int64)
    len_lut = np.zeros(max_sym + 1, dtype=np.int64)
    for s, c in codes.items():
        if s <= max_sym:
            code_lut[s] = c
            len_lut[s] = lengths[s]

    syms = symbol_stream.astype(np.int64)
    sym_codes = code_lut[syms]
    sym_lens = len_lut[syms]

    total = int(sym_lens.sum())
    starts = np.zeros(len(syms), dtype=np.int64)
    np.cumsum(sym_lens[:-1], out=starts[1:])

    sym_idx = np.repeat(np.arange(len(syms), dtype=np.int64), sym_lens)
    bit_in_sym = np.arange(total, dtype=np.int64) - starts[sym_idx]
    shift = sym_lens[sym_idx] - 1 - bit_in_sym
    bits = ((sym_codes[sym_idx] >> shift) & 1).astype(np.uint8)

    return np.packbits(bits).tobytes()


def decode_reference(packed: bytes, table: HuffmanTable, codes: dict[int, int],
                      n_symbols: int) -> np.ndarray:
    """Bit-by-bit CPU reference decoder -- deliberately NOT the LUT-based
    approach, so it can serve as an independent correctness oracle for both
    the CPU LUT decode and the GPU kernel."""
    length_of = {codes[s]: ln for s, ln in zip(table.symbols, table.lengths)}
    rev = {ln: {} for ln in set(table.lengths)}
    for s, ln in zip(table.symbols, table.lengths):
        rev[ln][codes[s]] = s

    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
    out = np.zeros(n_symbols, dtype=np.int32)
    pos = 0
    for i in range(n_symbols):
        code = 0
        for ln in range(1, table.max_bits + 1):
            code = (code << 1) | int(bits[pos])
            pos += 1
            if code in rev.get(ln, {}):
                out[i] = rev[ln][code]
                break
        else:
            raise ValueError("no matching code found -- corrupt stream or bad table")
    return out


def decode_lut_cpu(packed: bytes, sym_lut: np.ndarray, len_lut: np.ndarray,
                    max_bits: int, n_symbols: int) -> np.ndarray:
    """CPU version of the LUT-driven decode -- the same algorithm the GPU
    kernel implements, run here first so the algorithm itself (independent
    of any GPU-specific bugs) is validated against decode_reference."""
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8)).astype(np.int64)
    total_bits = len(bits)

    out = np.zeros(n_symbols, dtype=np.int32)
    pos = 0
    for i in range(n_symbols):
        window = np.zeros(max_bits, dtype=np.int64)
        avail = min(max_bits, total_bits - pos)
        window[:avail] = bits[pos: pos + avail]
        idx = 0
        for b in window:
            idx = (idx << 1) | int(b)
        out[i] = sym_lut[idx]
        pos += int(len_lut[idx])
    return out


@dataclasses.dataclass
class EncodedTensor:
    packed: bytes
    table: HuffmanTable
    codes: dict[int, int]
    sym_lut: np.ndarray
    len_lut: np.ndarray
    n_symbols: int
    shape: tuple[int, ...]


def encode_exponents(exponents: torch.Tensor, max_bits: int = 16) -> EncodedTensor:
    """End-to-end: build the code from the tensor's own exponent
    distribution, pack it, and build the GPU-ready LUT, all in one call."""
    flat = exponents.flatten().cpu().numpy().astype(np.int64)
    n = flat.shape[0]
    counts = np.bincount(flat)
    freqs = {int(s): int(c) for s, c in enumerate(counts) if c > 0}

    lengths = build_lengths(freqs, max_bits=max_bits)
    table, codes = canonical_codes(lengths)
    packed = pack_bits(flat, codes, lengths)
    sym_lut, len_lut = build_decode_lut(table, codes)

    return EncodedTensor(
        packed=packed, table=table, codes=codes,
        sym_lut=sym_lut, len_lut=len_lut,
        n_symbols=n, shape=tuple(exponents.shape),
    )
