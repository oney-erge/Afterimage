"""Correctness tests for the vectorized (v2) GPU decoder -- mirrors
test_gpu_decode.py's coverage, since v2 must be exactly as correct as v1
while being the fast path (see LOSSLESS_ENGINE.md Phase A performance note).
"""
import numpy as np
import pytest
import torch

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAS_TRITON),
    reason="GPU decode requires CUDA + triton (see IMPLEMENTATION_STATUS.md)",
)


def _check(exponents: torch.Tensor, chunk_size: int, block_chunks: int, max_bits: int = 16):
    from afterimage.runtime.gpu_decode_v2 import decode_gpu_v2
    from afterimage.runtime.huffman_chunked import decode_chunked_cpu_reference, encode_chunked

    enc = encode_chunked(exponents, chunk_size=chunk_size, max_bits=max_bits)
    cpu_ref = decode_chunked_cpu_reference(enc)
    assert np.array_equal(cpu_ref, exponents.numpy()), "CPU reference itself is wrong"

    gpu_out = decode_gpu_v2(enc, block_chunks=block_chunks).cpu().numpy()
    assert np.array_equal(gpu_out, exponents.numpy()), (
        f"v2 GPU decode mismatch at block_chunks={block_chunks}: "
        f"{int((gpu_out != exponents.numpy()).sum())} / {exponents.numel()} symbols wrong"
    )


@pytest.mark.parametrize("block_chunks", [1, 4, 32, 128, 256])
def test_various_block_sizes(block_chunks):
    torch.manual_seed(0)
    exponents = torch.randint(100, 140, (20000,))
    _check(exponents, chunk_size=64, block_chunks=block_chunks)


def test_n_chunks_not_multiple_of_block_chunks():
    """The masked tail case: n_chunks doesn't divide evenly by block_chunks,
    so the last program has some lanes past the end -- must not read/write
    out of bounds or corrupt valid lanes.

    block_chunks itself must stay a power of 2: tl.arange requires it (a
    real Triton language constraint, not a kernel bug -- confirmed by
    running this test with block_chunks=37 first, which failed at
    `tl.arange(0, 37)` inside Triton's own semantic checker, not inside any
    of this project's code). The uneven-tail case this test targets only
    needs n_chunks to not divide evenly by block_chunks, which 7777 chunks
    of 100 exponents against a block of 32 already gives (7777 exponents /
    100 per chunk = 78 chunks, not a multiple of 32)."""
    torch.manual_seed(1)
    exponents = torch.randint(0, 256, (7777,))
    _check(exponents, chunk_size=100, block_chunks=32)


def test_real_bf16_exponent_distribution():
    torch.manual_seed(2)
    W = (torch.randn(512, 4096) * 0.02).to(torch.bfloat16)
    bits = W.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exponent = ((bits >> 7) & 0xFF).flatten()
    _check(exponent, chunk_size=512, block_chunks=128)


@pytest.mark.parametrize("seed", range(8))
def test_matches_v1_on_random_distributions(seed):
    """Cross-validates v2 against v1 -- two independently structured kernels
    (scalar-per-program vs vectorized-per-program) agreeing is a much
    stronger signal than either alone agreeing with the CPU reference."""
    from afterimage.runtime.gpu_decode import decode_gpu
    from afterimage.runtime.gpu_decode_v2 import decode_gpu_v2
    from afterimage.runtime.huffman_chunked import encode_chunked

    rng = np.random.default_rng(seed)
    k = rng.integers(2, 200)
    probs = rng.dirichlet(np.ones(k) * rng.uniform(0.1, 3.0))
    data = rng.choice(k, size=rng.integers(2000, 8000), p=probs)
    exponents = torch.from_numpy(data.astype(np.int64))

    enc = encode_chunked(exponents, chunk_size=256, max_bits=16)
    v1_out = decode_gpu(enc).cpu().numpy()
    v2_out = decode_gpu_v2(enc, block_chunks=64).cpu().numpy()

    assert np.array_equal(v1_out, exponents.numpy())
    assert np.array_equal(v2_out, exponents.numpy())
    assert np.array_equal(v1_out, v2_out)
