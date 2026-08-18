"""GPU decode correctness tests (LOSSLESS_ENGINE.md Phase A).

Requires CUDA + triton; skips cleanly on the CPU-only Windows development
box (see IMPLEMENTATION_STATUS.md) and runs for real in WSL2/CUDA, which is
where this module was actually developed and where its correctness matters.
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


def _check(exponents: torch.Tensor, chunk_size: int, max_bits: int = 16):
    from afterimage.runtime.gpu_decode import decode_gpu
    from afterimage.runtime.huffman_chunked import decode_chunked_cpu_reference, encode_chunked

    enc = encode_chunked(exponents, chunk_size=chunk_size, max_bits=max_bits)
    cpu_ref = decode_chunked_cpu_reference(enc)
    assert np.array_equal(cpu_ref, exponents.numpy()), "CPU reference itself is wrong"

    gpu_out = decode_gpu(enc).cpu().numpy()
    assert np.array_equal(gpu_out, exponents.numpy()), (
        f"GPU decode mismatch: {int((gpu_out != exponents.numpy()).sum())} / "
        f"{exponents.numel()} symbols wrong"
    )


def test_small_skewed_distribution():
    torch.manual_seed(0)
    exponents = torch.randint(100, 140, (2000,))
    _check(exponents, chunk_size=64)


def test_full_256_symbol_alphabet_uniform():
    """Worst case for compression (uniform = high entropy = long-ish codes)
    but still must decode correctly."""
    torch.manual_seed(1)
    exponents = torch.randint(0, 256, (8000,))
    _check(exponents, chunk_size=256)


def test_real_bf16_exponent_distribution():
    torch.manual_seed(2)
    W = (torch.randn(512, 4096) * 0.02).to(torch.bfloat16)
    bits = W.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exponent = ((bits >> 7) & 0xFF).flatten()
    _check(exponent, chunk_size=512)


def test_various_chunk_sizes():
    torch.manual_seed(3)
    exponents = torch.randint(50, 90, (10000,))
    for cs in [32, 128, 512, 1024]:
        _check(exponents, chunk_size=cs)


def test_chunk_size_not_dividing_data_evenly():
    torch.manual_seed(4)
    exponents = torch.randint(0, 200, (7777,))
    _check(exponents, chunk_size=333)


def test_single_symbol_degenerate_case():
    exponents = torch.full((5000,), 42, dtype=torch.int64)
    _check(exponents, chunk_size=256)


def test_two_symbol_extreme_skew():
    torch.manual_seed(5)
    exponents = torch.from_numpy(
        np.random.default_rng(5).choice([10, 200], size=6000, p=[0.99, 0.01])
    )
    _check(exponents, chunk_size=256)


@pytest.mark.parametrize("seed", range(10))
def test_many_random_distributions(seed):
    """Broad sweep across random alphabet sizes and skew, matching the CPU
    reference sweep in test_huffman_chunked.py but now on the actual GPU."""
    rng = np.random.default_rng(seed)
    k = rng.integers(2, 200)
    probs = rng.dirichlet(np.ones(k) * rng.uniform(0.1, 3.0))
    data = rng.choice(k, size=rng.integers(2000, 8000), p=probs)
    exponents = torch.from_numpy(data.astype(np.int64))
    _check(exponents, chunk_size=int(rng.choice([64, 128, 256, 512])))
