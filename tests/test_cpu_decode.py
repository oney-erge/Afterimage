import numpy as np
import pytest
import torch

from afterimage.runtime.cpu_decode import (
    _HAS_NUMBA, decode_chunks_numba, decode_chunks_threaded, decode_chunks_vectorized,
)
from afterimage.runtime.huffman_chunked import decode_chunked_cpu_reference, encode_chunked


def _real_shaped_exponents(seed: int, n: int, chunk_size: int, max_bits: int):
    """Real transformer weights have skewed exponent distributions (that's
    the whole reason they compress); a uniform random test would exercise
    a degenerate, unrealistically-flat Huffman table. Sampling from a
    Gaussian and taking a bf16-like exponent field approximates the real
    skew without needing an actual model."""
    torch.manual_seed(seed)
    w = (torch.randn(n) * 0.02).to(torch.bfloat16)
    bits = w.view(torch.int16).to(torch.int32) & 0xFFFF
    exponent = (bits >> 7) & 0xFF
    return encode_chunked(exponent, chunk_size=chunk_size, max_bits=max_bits)


def test_vectorized_decode_matches_reference_bit_exact():
    enc = _real_shaped_exponents(seed=0, n=500_000, chunk_size=256, max_bits=16)
    expected = decode_chunked_cpu_reference(enc)
    got = decode_chunks_vectorized(enc, 0, enc.n_chunks)[: enc.n_symbols]
    assert np.array_equal(got, expected)


def test_vectorized_decode_matches_reference_across_chunk_sizes_and_max_bits():
    for chunk_size in (32, 128, 1024):
        for max_bits in (8, 12, 16):
            enc = _real_shaped_exponents(seed=chunk_size + max_bits, n=80_000,
                                         chunk_size=chunk_size, max_bits=max_bits)
            expected = decode_chunked_cpu_reference(enc)
            got = decode_chunks_vectorized(enc, 0, enc.n_chunks)[: enc.n_symbols]
            assert np.array_equal(got, expected), (chunk_size, max_bits)


def test_partial_chunk_range_matches_the_same_slice_of_the_full_decode():
    """The correctness argument in cpu_decode.py's module docstring, checked
    empirically: decoding chunks [10, 20) alone must equal decoding
    everything and slicing out that same range -- reading into a
    neighbouring chunk's real bytes must never change a decoded symbol."""
    enc = _real_shaped_exponents(seed=7, n=300_000, chunk_size=256, max_bits=16)
    full = decode_chunks_vectorized(enc, 0, enc.n_chunks)
    partial = decode_chunks_vectorized(enc, 10, 20)
    cs = enc.chunk_size
    assert np.array_equal(partial, full[10 * cs: 20 * cs])


def test_threaded_decode_matches_single_threaded():
    enc = _real_shaped_exponents(seed=3, n=400_000, chunk_size=128, max_bits=16)
    single = decode_chunks_vectorized(enc, 0, enc.n_chunks)
    for n_threads in (1, 2, 4, 8):
        threaded = decode_chunks_threaded(enc, n_threads)
        assert np.array_equal(threaded, single), n_threads


def test_threaded_decode_with_more_threads_than_chunks_does_not_crash():
    enc = _real_shaped_exponents(seed=11, n=2_000, chunk_size=1024, max_bits=16)
    assert enc.n_chunks <= 4
    got = decode_chunks_threaded(enc, n_threads=16)
    expected = decode_chunked_cpu_reference(enc)
    assert np.array_equal(got[: enc.n_symbols], expected)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
def test_numba_decode_matches_reference_bit_exact():
    enc = _real_shaped_exponents(seed=13, n=500_000, chunk_size=256, max_bits=16)
    expected = decode_chunked_cpu_reference(enc)
    got = decode_chunks_numba(enc)[: enc.n_symbols]
    assert np.array_equal(got, expected)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
def test_numba_decode_matches_reference_across_chunk_sizes_and_max_bits():
    for chunk_size in (32, 128, 1024):
        for max_bits in (8, 12, 16):
            enc = _real_shaped_exponents(seed=100 + chunk_size + max_bits, n=80_000,
                                         chunk_size=chunk_size, max_bits=max_bits)
            expected = decode_chunked_cpu_reference(enc)
            got = decode_chunks_numba(enc)[: enc.n_symbols]
            assert np.array_equal(got, expected), (chunk_size, max_bits)
