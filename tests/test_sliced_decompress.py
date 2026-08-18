"""Sliced decode must be bit-identical to unsliced -- it exists to bound
peak VRAM, and would be worthless if it changed a single weight."""
import numpy as np
import pytest
import torch

try:
    import triton  # noqa: F401
    _HAS = torch.cuda.is_available()
except ImportError:
    _HAS = False

from afterimage.runtime.compressed_store import compress_layer, decompress_layer_cpu_reference


def test_slice_encoded_preserves_symbols_cpu():
    """Chunk-range slicing must decode the same symbols the full stream
    does -- verified on CPU so it holds independently of the GPU kernel."""
    from afterimage.runtime.compressed_store import _slice_encoded
    from afterimage.runtime.huffman_chunked import decode_chunked_cpu_reference, encode_chunked

    torch.manual_seed(0)
    exps = torch.randint(100, 140, (10000,))
    enc = encode_chunked(exps, chunk_size=256, max_bits=16)
    full = decode_chunked_cpu_reference(enc)

    cps = 5
    pieces = []
    for c0 in range(0, enc.n_chunks, cps):
        c1 = min(c0 + cps, enc.n_chunks)
        pieces.append(decode_chunked_cpu_reference(_slice_encoded(enc, c0, c1)))
    joined = np.concatenate(pieces)[: enc.n_symbols]

    assert np.array_equal(joined, full), "sliced decode diverged from full decode"


@pytest.mark.skipif(not _HAS, reason="needs CUDA + triton")
def test_sliced_gpu_decompress_matches_unsliced_bit_exact():
    from afterimage.runtime.compressed_store import decompress_layer_gpu

    torch.manual_seed(1)
    W = (torch.randn(4096, 512) * 0.02).to(torch.bfloat16)   # 2.09M weights
    layer = compress_layer(W, chunk_size=1024)

    big = decompress_layer_gpu(layer, max_slice_elems=1 << 30)   # one shot
    small = decompress_layer_gpu(layer, max_slice_elems=1 << 17) # ~131k -> many slices

    assert torch.equal(big.view(torch.int16), small.view(torch.int16))
    assert torch.equal(small.cpu().view(torch.int16), W.view(torch.int16))


def test_sliced_cpu_reference_still_bit_exact():
    torch.manual_seed(2)
    W = (torch.randn(2048, 256) * 0.02).to(torch.bfloat16)
    layer = compress_layer(W, chunk_size=512)
    assert torch.equal(decompress_layer_cpu_reference(layer).view(torch.int16),
                       W.view(torch.int16))
