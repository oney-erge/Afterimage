import pytest
import torch

from afterimage.runtime.compressed_store import (
    compress_layer, decompress_layer_cpu_reference, decompress_layer_gpu, decompress_rows_gpu,
)
from afterimage.runtime.cpu_decode import _HAS_NUMBA


def test_full_weight_reconstruction_bit_exact_cpu():
    """The actual product claim: compress a real bf16 tensor, decompress it,
    get back the IDENTICAL bit pattern -- not close, identical."""
    torch.manual_seed(0)
    W = (torch.randn(128, 512) * 0.02).to(torch.bfloat16)
    layer = compress_layer(W, chunk_size=256)
    recon = decompress_layer_cpu_reference(layer)

    assert recon.dtype == torch.bfloat16
    assert recon.shape == W.shape
    assert torch.equal(recon, W), "reconstruction is not bit-exact"

    orig_bits = W.view(torch.int16)
    recon_bits = recon.view(torch.int16)
    assert torch.equal(orig_bits, recon_bits), "bit patterns differ even if values compare equal"


def test_bounded_work_chunks_preserve_every_bf16_bit():
    """Force many extraction chunks without allocating a multi-GB fixture."""
    torch.manual_seed(40)
    W = (torch.randn(37, 59) * 0.02).to(torch.bfloat16)
    layer = compress_layer(
        W, chunk_size=31, work_chunk_elems=17)
    recon = decompress_layer_cpu_reference(layer)

    assert torch.equal(recon.view(torch.int16), W.view(torch.int16))


def test_reconstruction_bit_exact_on_realistic_weight_scale():
    torch.manual_seed(1)
    for scale in [0.001, 0.02, 0.5, 3.0]:
        W = (torch.randn(64, 256) * scale).to(torch.bfloat16)
        layer = compress_layer(W)
        recon = decompress_layer_cpu_reference(layer)
        assert torch.equal(recon, W), f"failed at scale={scale}"


def test_compression_reduces_size_at_real_layer_scale():
    """The LUT itself has a fixed cost: 2**max_bits entries (up to 320KB at
    the max_bits=16 ceiling -- sym_lut is int32, len_lut is int8, so
    65536 * (4+1) = 327680 bytes). At real transformer layer scale
    (millions of weights) that is negligible overhead. An earlier version
    of this test used a small 256x1024 (262144-weight, 512KB original)
    tensor, where the LUT overhead alone exceeded the entire original size
    and made compression look like it LOST -- a real amortization effect,
    not a bug, but the wrong scale to test the actual claim at."""
    torch.manual_seed(2)
    W = (torch.randn(8960, 1536) * 0.02).to(torch.bfloat16)  # real down_proj scale
    layer = compress_layer(W)
    assert layer.compressed_bytes < layer.original_bytes
    assert layer.compressed_bytes / layer.original_bytes < 0.75


def test_lut_fixed_cost_dominates_below_a_scale_threshold():
    """Documents the amortization boundary explicitly rather than letting it
    surprise someone in production: compression genuinely does not help
    (and can hurt) on small enough tensors, because the LUT is a fixed cost
    independent of tensor size."""
    torch.manual_seed(9)
    tiny = (torch.randn(16, 16) * 0.02).to(torch.bfloat16)  # 256 weights, 512 bytes original
    layer = compress_layer(tiny)
    assert layer.compressed_bytes > layer.original_bytes, (
        "expected the LUT's fixed cost to dominate at this scale -- if this "
        "now passes, either the LUT got much smaller or this test's premise "
        "needs revisiting"
    )


def test_zero_and_negative_and_extreme_values():
    """Edge cases that could break sign/exponent/mantissa bit extraction:
    exact zero (exponent all-zero, a real IEEE special case), negative
    values, and values near the representable extremes."""
    W = torch.tensor([0.0, -0.0, 1.0, -1.0, 1e-30, -1e-30, 1e30, -1e30, 3.14159],
                      dtype=torch.bfloat16).reshape(3, 3)
    layer = compress_layer(W, chunk_size=9)
    recon = decompress_layer_cpu_reference(layer)
    assert torch.equal(recon.view(torch.int16), W.view(torch.int16))


def test_shape_preserved_through_roundtrip():
    torch.manual_seed(3)
    for shape in [(1, 100), (100, 1), (7, 13), (8960, 1536)]:
        W = (torch.randn(*shape) * 0.02).to(torch.bfloat16)
        layer = compress_layer(W, chunk_size=512)
        recon = decompress_layer_cpu_reference(layer)
        assert recon.shape == shape
        assert torch.equal(recon, W)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
def test_decompress_layer_gpu_dispatches_to_cpu_decoder_when_no_cuda():
    """decompress_layer_gpu's name is legacy; on device='cpu' it must route
    through cpu_decode.py's numba decoder (there is no CUDA to fall back to)
    and still be bit-exact -- this is the path a machine with no GPU at all
    (e.g. macOS) actually runs at inference time."""
    torch.manual_seed(20)
    W = (torch.randn(128, 512) * 0.02).to(torch.bfloat16)
    layer = compress_layer(W, chunk_size=256)

    out = decompress_layer_gpu(layer, device="cpu")
    ref = decompress_layer_cpu_reference(layer)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    assert torch.equal(out.view(torch.int16), W.view(torch.int16))


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
def test_sliced_cpu_decompress_matches_unsliced():
    """Same guarantee test_sliced_decompress.py checks for the GPU kernel,
    for the CPU dispatch path: forcing many slices must not change a single
    decoded weight."""
    torch.manual_seed(21)
    W = (torch.randn(4096, 512) * 0.02).to(torch.bfloat16)
    layer = compress_layer(W, chunk_size=1024)

    big = decompress_layer_gpu(layer, device="cpu", max_slice_elems=1 << 30)
    small = decompress_layer_gpu(layer, device="cpu", max_slice_elems=1 << 17)

    assert torch.equal(big.view(torch.int16), small.view(torch.int16))
    assert torch.equal(small.view(torch.int16), W.view(torch.int16))


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
def test_decompress_rows_cpu_matches_full_layer_slice():
    torch.manual_seed(22)
    W = (torch.randn(200, 256) * 0.02).to(torch.bfloat16)
    layer = compress_layer(W, chunk_size=64)

    full = decompress_layer_gpu(layer, device="cpu")
    rows = decompress_rows_gpu(layer, 30, 90, device="cpu")

    assert torch.equal(rows.view(torch.int16), full[30:90].view(torch.int16))
