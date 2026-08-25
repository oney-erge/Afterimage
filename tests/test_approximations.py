import torch

from afterimage.probe.approximations import (
    activation_weighted_svd,
    full_bytes,
    lowrank_plus_quantized_residual,
    pca_projection,
    quantize_uniform,
    quantize_uniform_bytes,
    relative_output_error,
)



import pytest

pytestmark = pytest.mark.archive  # Phase-0 subspace-activation-cache branch, killed per its own gate
def _setup(d_out=40, d_in=64, n=300, seed=0):
    torch.manual_seed(seed)
    W = torch.randn(d_out, d_in)
    X = torch.randn(n, d_in)
    return W, X


def test_full_rank_pca_projection_is_near_exact():
    W, X = _setup()
    W_hat = pca_projection(W, X, r=min(W.shape[1], X.shape[0]))
    assert relative_output_error(W, W_hat, X) < 1e-4


def test_more_rank_never_hurts_pca():
    W, X = _setup()
    errs = [relative_output_error(W, pca_projection(W, X, r), X) for r in [2, 8, 32]]
    assert errs[0] >= errs[1] >= errs[2]


def test_more_bits_never_hurts_quantization():
    W, X = _setup()
    errs = [relative_output_error(W, quantize_uniform(W, b), X) for b in [2, 4, 8]]
    assert errs[0] > errs[1] > errs[2]


def test_quantization_bytes_scale_with_bits():
    W, _ = _setup()
    assert quantize_uniform_bytes(W, 2) < quantize_uniform_bytes(W, 4) < quantize_uniform_bytes(W, 8)
    assert quantize_uniform_bytes(W, 8) < full_bytes(W)


def test_activation_weighted_svd_beats_pca_when_channels_have_unequal_scale():
    """The specific failure mode activation weighting is meant to fix: when
    some input channels are far larger than others (the massive-activation
    regime real LLMs are in), unweighted PCA spends its budget on the wrong
    directions."""
    torch.manual_seed(3)
    d_out, d_in, n, r = 32, 48, 400, 6

    X = torch.randn(n, d_in)
    X[:, :4] *= 60.0  # a few channels dominate, as in a real transformer
    W = torch.randn(d_out, d_in)

    err_pca = relative_output_error(W, pca_projection(W, X, r), X)
    err_awsvd = relative_output_error(W, activation_weighted_svd(W, X, r), X)

    assert err_awsvd < err_pca, (
        f"activation-weighted SVD ({err_awsvd:.4f}) should beat plain PCA "
        f"({err_pca:.4f}) under unequal channel scales"
    )


def test_hybrid_beats_both_of_its_parts_at_comparable_memory():
    """The core claim of the proposed fix: low-rank + quantized residual is
    better than either the low-rank part alone or the quantizer alone."""
    torch.manual_seed(4)
    d_out, d_in, n = 64, 96, 500
    X = torch.randn(n, d_in)
    X[:, :5] *= 30.0
    W = torch.randn(d_out, d_in)

    r, bits = 8, 4
    err_lowrank_only = relative_output_error(W, activation_weighted_svd(W, X, r), X)
    err_quant_only = relative_output_error(W, quantize_uniform(W, bits), X)
    err_hybrid = relative_output_error(
        W, lowrank_plus_quantized_residual(W, X, r, bits), X)

    assert err_hybrid < err_lowrank_only, (
        f"hybrid {err_hybrid:.4f} should beat low-rank alone {err_lowrank_only:.4f}")
    assert err_hybrid < err_quant_only, (
        f"hybrid {err_hybrid:.4f} should beat quantization alone {err_quant_only:.4f}")


def test_residual_is_easier_to_quantize_than_the_original_matrix():
    """The mechanism behind the hybrid: removing dominant low-rank structure
    leaves a residual with a lighter tail, which a uniform quantizer handles
    better than the original heavy-tailed matrix."""
    torch.manual_seed(5)
    d_out, d_in, n, r = 64, 96, 400, 12
    X = torch.randn(n, d_in)
    # give W genuine low-rank structure plus noise, as real weights have
    W = torch.randn(d_out, r) @ torch.randn(r, d_in) * 3.0 + torch.randn(d_out, d_in) * 0.1

    lowrank = activation_weighted_svd(W, X, r)
    residual = W - lowrank

    kurt_W = ((W - W.mean()) ** 4).mean() / ((W - W.mean()) ** 2).mean() ** 2
    kurt_res = ((residual - residual.mean()) ** 4).mean() / ((residual - residual.mean()) ** 2).mean() ** 2

    assert residual.abs().max() < W.abs().max(), "residual should have smaller dynamic range"
    assert kurt_res <= kurt_W * 1.5, f"residual kurtosis {kurt_res:.2f} vs W {kurt_W:.2f}"
