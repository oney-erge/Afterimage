import numpy as np
import torch

from afterimage.runtime.huffman import decode_reference
from afterimage.runtime.huffman_chunked import (
    ChunkedBitReader,
    decode_chunked_cpu_reference,
    encode_chunked,
)


def test_chunked_roundtrip_matches_original_exactly():
    torch.manual_seed(0)
    exponents = torch.randint(100, 140, (10000,))
    enc = encode_chunked(exponents, chunk_size=256, max_bits=16)
    decoded = decode_chunked_cpu_reference(enc)
    assert np.array_equal(decoded, exponents.numpy())


def test_chunked_roundtrip_accepts_uint8_without_int64_staging():
    torch.manual_seed(10)
    exponents = torch.randint(0, 256, (4099,), dtype=torch.uint8)
    enc = encode_chunked(exponents, chunk_size=127, max_bits=16)
    decoded = decode_chunked_cpu_reference(enc)

    assert np.array_equal(decoded, exponents.numpy())


def test_chunked_roundtrip_when_chunk_size_does_not_divide_evenly():
    torch.manual_seed(1)
    exponents = torch.randint(100, 140, (10007,))  # deliberately not a multiple of chunk_size
    enc = encode_chunked(exponents, chunk_size=300, max_bits=16)
    decoded = decode_chunked_cpu_reference(enc)
    assert decoded.shape[0] == exponents.shape[0]
    assert np.array_equal(decoded, exponents.numpy())


def test_chunked_matches_single_stream_huffman_bit_for_bit_symbols():
    """Cross-validates against the UNCHUNKED reference decoder in
    huffman.py -- two independently written decoders, same input, same
    output, is the strongest correctness signal available before trusting
    either."""
    torch.manual_seed(2)
    exponents = torch.randint(0, 256, (5000,))

    enc = encode_chunked(exponents, chunk_size=128, max_bits=16)
    chunked_out = decode_chunked_cpu_reference(enc)

    from afterimage.runtime.huffman import canonical_codes, build_lengths
    flat = exponents.numpy().astype(np.int64)
    counts = np.bincount(flat, minlength=256)
    freqs = {int(s): int(c) for s, c in enumerate(counts) if c > 0}
    lengths = build_lengths(freqs, max_bits=16)
    table, codes = canonical_codes(lengths)
    from afterimage.runtime.huffman import pack_bits
    packed = pack_bits(flat, codes, lengths)
    single_stream_out = decode_reference(packed, table, codes, len(flat))

    assert np.array_equal(chunked_out, single_stream_out)
    assert np.array_equal(chunked_out, flat)


def test_bit_reader_never_shifts_by_32_or_more():
    """Guards the documented safety margin: nbits must never exceed 24
    (max_bits=16 + one 8-bit refill), so `self.buf << consumed` never
    approaches undefined-shift territory -- checked directly against the
    reader's internal state, not just final output."""
    torch.manual_seed(3)
    exponents = torch.randint(0, 256, (3000,))
    enc = encode_chunked(exponents, chunk_size=64, max_bits=16)

    for c in range(min(enc.n_chunks, 10)):
        start = int(enc.chunk_offsets[c])
        end = start + int(enc.chunk_nbytes[c])
        reader = ChunkedBitReader(enc.packed[start:end])
        for _ in range(enc.chunk_size):
            reader.decode_one(enc.sym_lut, enc.len_lut, enc.max_bits)
            assert reader.nbits <= 24, f"nbits={reader.nbits} exceeds safety margin"
            assert 0 <= reader.buf <= 0xFFFFFFFF, "buffer escaped 32-bit range"


def test_chunked_roundtrip_on_real_bf16_exponents():
    torch.manual_seed(4)
    W = (torch.randn(128, 4096) * 0.02).to(torch.bfloat16)
    bits = W.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exponent = ((bits >> 7) & 0xFF).flatten()

    enc = encode_chunked(exponent, chunk_size=512, max_bits=16)
    decoded = decode_chunked_cpu_reference(enc)
    assert np.array_equal(decoded, exponent.numpy())


def test_single_chunk_smaller_than_data_still_correct():
    """Very small chunk_size relative to data -- many chunks, stresses the
    offset/padding arithmetic specifically."""
    torch.manual_seed(5)
    exponents = torch.randint(50, 80, (2000,))
    enc = encode_chunked(exponents, chunk_size=16, max_bits=16)
    assert enc.n_chunks == 125
    decoded = decode_chunked_cpu_reference(enc)
    assert np.array_equal(decoded, exponents.numpy())
