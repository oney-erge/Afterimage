import math

import pytest
import torch

from afterimage.probe.entropy import analyze_tensor, compressed_bytes



pytestmark = pytest.mark.archive  # Phase-0 subspace-activation-cache branch, killed per its own gate
def test_rejects_non_float16_dtypes():
    with pytest.raises(TypeError):
        analyze_tensor(torch.randn(10, 10))  # float32


def test_constant_tensor_has_near_zero_entropy():
    """Every weight identical -> one exponent, one sign, one mantissa ->
    essentially free to code."""
    W = torch.full((256, 256), 1.5, dtype=torch.float16)
    rep = analyze_tensor(W)
    assert rep.exponent_entropy < 0.01
    assert rep.mantissa_entropy < 0.01
    assert rep.compression_ratio > 100


def test_normal_weights_show_low_exponent_entropy():
    """The mechanism the whole approach rests on: trained-like weights
    concentrate in few exponents, far below the field's allocated width."""
    torch.manual_seed(0)
    W = (torch.randn(512, 512) * 0.02).to(torch.bfloat16)
    rep = analyze_tensor(W)
    assert rep.exponent_entropy < rep.exponent_bits, (
        f"exponent entropy {rep.exponent_entropy:.2f} should be below its "
        f"{rep.exponent_bits}-bit allocation"
    )
    assert rep.n_distinct_exponents < 2 ** rep.exponent_bits


def test_mantissa_is_nearly_incompressible():
    """Confirms where the gain does NOT come from -- important, because a
    plan that assumes the mantissa compresses would overestimate badly."""
    torch.manual_seed(1)
    W = (torch.randn(512, 512) * 0.02).to(torch.bfloat16)
    rep = analyze_tensor(W)
    assert rep.mantissa_entropy > rep.mantissa_bits * 0.9, (
        f"mantissa entropy {rep.mantissa_entropy:.2f} of {rep.mantissa_bits} "
        f"bits -- expected near-uniform"
    )


def test_bfloat16_beats_float16_only_when_the_source_is_fp32():
    """bf16 compresses better than fp16 *from an fp32 source*, because fp16's
    10-bit mantissa then carries real information that bf16's 7-bit one
    discarded."""
    torch.manual_seed(2)
    base = torch.randn(512, 512) * 0.02
    rep_bf = analyze_tensor(base.to(torch.bfloat16))
    rep_fp = analyze_tensor(base.to(torch.float16))
    assert rep_bf.size_fraction < rep_fp.size_fraction - 0.1, (
        f"from fp32: bf16 {rep_bf.size_fraction:.3f} should clearly beat "
        f"fp16 {rep_fp.size_fraction:.3f}"
    )


def test_native_bfloat16_checkpoint_compresses_the_same_in_either_dtype():
    """The real-model case, which contradicts the naive prediction and is
    what the audit of Qwen2.5-1.5B actually measured (66.1% vs 66.2%).

    Serving a native-bf16 checkpoint as fp16 zero-pads the mantissa by 3
    bits, adding no information; entropy coding recovers exactly that
    padding, so fp16's extra mantissa waste offsets its smaller exponent
    waste and both land in the same place."""
    torch.manual_seed(7)
    native = (torch.randn(512, 512) * 0.02).to(torch.bfloat16)
    rep_bf = analyze_tensor(native)
    rep_fp = analyze_tensor(native.to(torch.float16))
    assert abs(rep_bf.size_fraction - rep_fp.size_fraction) < 0.02, (
        f"native-bf16 checkpoint should compress alike in both dtypes: "
        f"bf16 {rep_bf.size_fraction:.3f} vs fp16 {rep_fp.size_fraction:.3f}"
    )


def test_compressed_bytes_is_smaller_than_raw_and_self_consistent():
    torch.manual_seed(3)
    W = (torch.randn(256, 256) * 0.02).to(torch.bfloat16)
    raw = W.numel() * 2
    comp = compressed_bytes(W)
    assert 0 < comp < raw
    rep = analyze_tensor(W)
    exact = rep.n_weights * rep.compressed_bits_per_weight / 8
    # compressed_bytes truncates to a whole number of bytes, so it lands
    # within one byte below the exact entropy figure
    assert exact - 1 <= comp <= exact


def test_entropy_never_exceeds_field_width():
    """Sanity: a k-bit field cannot carry more than k bits of entropy."""
    torch.manual_seed(4)
    W = (torch.randn(1024, 128) * 0.05).to(torch.bfloat16)
    rep = analyze_tensor(W)
    assert rep.exponent_entropy <= rep.exponent_bits + 1e-6
    assert rep.mantissa_entropy <= rep.mantissa_bits + 1e-6
    assert rep.sign_entropy <= 1.0 + 1e-6
