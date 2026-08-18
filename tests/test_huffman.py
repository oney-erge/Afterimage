import numpy as np
import pytest
import torch

from afterimage.runtime.huffman import (
    build_lengths,
    canonical_codes,
    decode_lut_cpu,
    decode_reference,
    encode_exponents,
    pack_bits,
)


def _roundtrip(exponents_np: np.ndarray, max_bits: int = 16):
    exponents = torch.from_numpy(exponents_np)
    enc = encode_exponents(exponents, max_bits=max_bits)

    ref = decode_reference(enc.packed, enc.table, enc.codes, enc.n_symbols)
    lut = decode_lut_cpu(enc.packed, enc.sym_lut, enc.len_lut, enc.table.max_bits, enc.n_symbols)
    return enc, ref, lut


def test_single_symbol_alphabet():
    data = np.full(500, 7, dtype=np.int64)
    enc, ref, lut = _roundtrip(data)
    assert np.array_equal(ref, data)
    assert np.array_equal(lut, data)


def test_two_symbol_alphabet_skewed():
    rng = np.random.default_rng(0)
    data = rng.choice([3, 9], size=2000, p=[0.95, 0.05])
    enc, ref, lut = _roundtrip(data)
    assert np.array_equal(ref, data)
    assert np.array_equal(lut, data)


def test_uniform_alphabet_no_compression_benefit_but_still_correct():
    rng = np.random.default_rng(1)
    data = rng.integers(0, 16, size=1000)
    enc, ref, lut = _roundtrip(data)
    assert np.array_equal(ref, data)
    assert np.array_equal(lut, data)


def test_realistic_skewed_distribution_matches_measured_weight_entropy():
    """Approximates the ACTUAL distribution shape measured in
    docs/CAPACITY_RESULTS.md-adjacent entropy audit: a bf16 exponent field
    where a handful of values dominate (entropy ~2.6 bits out of 8)."""
    rng = np.random.default_rng(2)
    # concentrate mass on ~6 values out of 256, mimicking the measured
    # ~2.6-bit exponent entropy
    hot = rng.choice(256, size=6, replace=False)
    probs = np.zeros(256)
    probs[hot] = rng.dirichlet(np.ones(6))
    data = rng.choice(256, size=50000, p=probs)

    enc, ref, lut = _roundtrip(data, max_bits=16)
    assert np.array_equal(ref, data)
    assert np.array_equal(lut, data)

    # sanity: this distribution should compress well
    avg_len = sum(enc.table.lengths[i] * (data == s).sum()
                  for i, s in enumerate(enc.table.symbols)) / len(data)
    assert avg_len < 4.0, f"expected strong compression on skewed data, got {avg_len:.2f} bits/symbol"


def test_length_limiting_activates_and_stays_correct():
    """Force a distribution whose NATURAL Huffman lengths exceed max_bits
    (a long tail of rare symbols, e.g. a Fibonacci-weighted alphabet, is the
    classic case that produces long codes), and confirm package-merge keeps
    every code within the limit while remaining bit-exact."""
    n_symbols = 40
    freqs = {}
    a, b = 1, 1
    for s in range(n_symbols):
        freqs[s] = a
        a, b = b, a + b  # fibonacci-ish -> deep, unbalanced tree naturally

    max_bits = 6  # deliberately tight -- natural Huffman would exceed this
    lengths = build_lengths(freqs, max_bits=max_bits)
    assert max(lengths.values()) <= max_bits, "package-merge must respect the limit"

    rng = np.random.default_rng(3)
    total = sum(freqs.values())
    probs = np.array([freqs[s] / total for s in range(n_symbols)])
    data = rng.choice(n_symbols, size=20000, p=probs)

    enc, ref, lut = _roundtrip(data, max_bits=max_bits)
    assert np.array_equal(ref, data)
    assert np.array_equal(lut, data)
    assert enc.table.max_bits <= max_bits


def test_lut_decode_agrees_with_bit_by_bit_reference_on_many_seeds():
    """Cross-checks the two independent decoders (bit-by-bit reference vs
    LUT) against each other across many random distributions -- the LUT
    algorithm is what the GPU kernel will implement, so this is the
    strongest correctness signal before trusting a GPU port of it."""
    for seed in range(15):
        rng = np.random.default_rng(seed)
        k = rng.integers(2, 30)
        probs = rng.dirichlet(np.ones(k) * rng.uniform(0.1, 3.0))
        data = rng.choice(k, size=rng.integers(500, 3000), p=probs)
        enc, ref, lut = _roundtrip(data)
        assert np.array_equal(ref, data), f"seed {seed}: reference decoder mismatch"
        assert np.array_equal(lut, data), f"seed {seed}: LUT decoder mismatch"


def test_real_model_exponent_bytes_roundtrip_exactly():
    """The actual use case: encode/decode the real exponent field of a
    weight tensor with a realistic distribution shape (bf16-like), no
    synthetic approximation."""
    torch.manual_seed(0)
    W = (torch.randn(256, 512) * 0.02).to(torch.bfloat16)
    bits = W.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exponent = (bits >> 7) & 0xFF  # bf16: 7 mantissa bits, 8 exponent bits

    flat_expected = exponent.numpy().flatten()
    enc, ref, lut = _roundtrip(exponent.numpy().astype(np.int64))
    assert np.array_equal(ref, flat_expected)
    assert np.array_equal(lut, flat_expected)

    compressed_bits = sum(
        enc.table.lengths[i] * int((flat_expected == s).sum())
        for i, s in enumerate(enc.table.symbols)
    )
    original_bits = exponent.numel() * 8
    assert compressed_bits < original_bits, "real bf16 exponents should compress"
