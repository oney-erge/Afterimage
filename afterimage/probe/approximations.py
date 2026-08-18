"""Weight-approximation schemes, compared like-for-like.

Every function here returns an approximate weight matrix W_hat of the SAME
shape as W, so they can be swapped into a real model and measured with one
harness. Each has a matching `*_bytes` function so compression is computed
from the actual stored representation, not asserted.

Why this module exists: PHASE0_RESULTS.md measured the original Afterimage
scheme (PCA subspace projection) at 60-96% output error, and the diagnosis
showed a SINGLE layer already carries 67-89% of that -- compounding across
layers is not the cause. That points at the per-layer approximation itself,
and there are two specific, testable reasons it could be bad:

  1. WRONG OBJECTIVE. PCA on activations picks the subspace minimizing
     ||x - UU^T x||. But the quantity that matters is ||W(x - UU^T x)||.
     Directions where x varies a lot but W is insensitive get spent budget;
     directions where x varies little but W amplifies get dropped. These are
     different problems and PCA solves the wrong one.

  2. DISCARDED RESIDUAL. Projection throws the orthogonal component away
     entirely -- it contributes exactly zero to the output. Quantization by
     contrast spends its bit budget uniformly and captures *everything*
     coarsely. That is the fundamental reason a 4-bit quantizer at 4x
     compression beats a rank-128 projection at 10x compression by a wide
     margin, and it suggests the fix: keep the low-rank part AND store the
     residual coarsely instead of dropping it.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fp16_bytes(*shapes: tuple[int, ...]) -> int:
    return sum(int(torch.tensor(s).prod().item()) * 2 for s in shapes)


def full_bytes(W: torch.Tensor) -> int:
    """fp16 reference storage."""
    return W.shape[0] * W.shape[1] * 2


# ---------------------------------------------------------------------------
# 1. PCA subspace projection -- the ORIGINAL Afterimage scheme
# ---------------------------------------------------------------------------


def pca_projection(W: torch.Tensor, X: torch.Tensor, r: int) -> torch.Tensor:
    """W_hat = W U U^T, with U = top-r principal directions of the activations.

    This is what PHASE0_RESULTS.md measured. Stored as U (d_in x r) and
    M = W U (d_out x r), so a hit costs one (r x d_in) projection plus one
    (d_out x r) expansion and never touches W.
    """
    Xf = X.float()
    Xc = Xf - Xf.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
    U = Vt[:r].T.to(W.dtype)  # (d_in, r)
    return W @ U @ U.T


def pca_projection_bytes(W: torch.Tensor, r: int) -> int:
    d_out, d_in = W.shape
    return (d_in * r + d_out * r) * 2


# ---------------------------------------------------------------------------
# 2. Activation-weighted SVD -- fixes objective (1)
# ---------------------------------------------------------------------------


def activation_weighted_svd(W: torch.Tensor, X: torch.Tensor, r: int,
                             eps: float = 1e-6) -> torch.Tensor:
    """Best rank-r approximation of W *in the metric the activations induce*.

    Minimizes E||(W - W_hat)x||^2 rather than the activation reconstruction
    error. With S = diag(RMS of each input channel), the objective is
    ||(W - W_hat) S||_F^2, whose optimum is the truncated SVD of (W S) mapped
    back through S^-1:

        W S = U D V^T   ->   W_hat = U_r D_r V_r^T S^-1

    Diagonal S (per-input-channel scaling) rather than a full covariance
    Cholesky: it is what ASVD uses, it is far cheaper, and it captures the
    dominant effect, which is that LLM activation channels differ in scale by
    orders of magnitude (the massive-activation phenomenon). A channel that is
    1000x larger dominates the output and must not be traded away for one that
    is tiny, which is exactly the mistake unweighted PCA makes.
    """
    Xf = X.float()
    scale = Xf.pow(2).mean(dim=0).sqrt().clamp_min(eps)  # (d_in,)
    Wf = W.float()

    WS = Wf * scale.unsqueeze(0)          # (d_out, d_in)
    U, D, Vt = torch.linalg.svd(WS, full_matrices=False)
    WS_r = (U[:, :r] * D[:r]) @ Vt[:r]    # rank-r approx of WS
    W_hat = WS_r / scale.unsqueeze(0)     # undo the weighting
    return W_hat.to(W.dtype)


def activation_weighted_svd_bytes(W: torch.Tensor, r: int) -> int:
    """A = U_r D_r (d_out x r), B = V_r^T S^-1 (r x d_in). Same cost as PCA."""
    d_out, d_in = W.shape
    return (d_in * r + d_out * r) * 2


# ---------------------------------------------------------------------------
# 3. Uniform quantization -- the competitor
# ---------------------------------------------------------------------------


def quantize_uniform(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-row uniform quantization -- the standard baseline
    that Q4_K_M-class GGUF quantization approximates.

    Per-row (per output channel) scales rather than one global scale: weight
    rows differ substantially in dynamic range, and a single global scale
    wastes most of the code space on the few largest rows.
    """
    Wf = W.float()
    qmax = 2 ** (bits - 1) - 1
    scale = Wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.clamp(torch.round(Wf / scale), -qmax, qmax)
    return (q * scale).to(W.dtype)


def quantize_uniform_bytes(W: torch.Tensor, bits: int) -> int:
    d_out, d_in = W.shape
    return (d_out * d_in * bits) // 8 + d_out * 2  # codes + fp16 row scales


def quantize_grouped(W: torch.Tensor, bits: int, group_size: int = 64) -> torch.Tensor:
    """Group-wise quantization: an independent scale per contiguous block of
    `group_size` weights, instead of one scale for a whole row.

    This is what real GGUF k-quants (Q4_K_M etc.) actually do, and it matters
    enormously: one scale across a full 8960-element row is set by that row's
    single largest weight, so every other weight in the row loses code space
    to an outlier thousands of positions away. Shrinking the scope to 64
    confines each outlier's damage to its own group.

    Reporting per-row numbers as "the quantization baseline" would understate
    the real competitor and flatter anything compared against it -- which is
    why this exists.
    """
    Wf = W.float()
    d_out, d_in = Wf.shape
    pad = (-d_in) % group_size
    if pad:
        Wf = torch.nn.functional.pad(Wf, (0, pad))
    g = Wf.reshape(d_out, -1, group_size)

    qmax = 2 ** (bits - 1) - 1
    scale = g.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.clamp(torch.round(g / scale), -qmax, qmax)
    out = (q * scale).reshape(d_out, -1)
    if pad:
        out = out[:, :d_in]
    return out.to(W.dtype)


def quantize_grouped_bytes(W: torch.Tensor, bits: int, group_size: int = 64) -> int:
    d_out, d_in = W.shape
    n_groups = d_out * ((d_in + group_size - 1) // group_size)
    return (d_out * d_in * bits) // 8 + n_groups * 2  # codes + one fp16 scale per group


def quantize_grouped_with_outliers(W: torch.Tensor, bits: int, group_size: int = 64,
                                    outlier_frac: float = 0.001) -> torch.Tensor:
    """Group-wise quantization, but the largest-magnitude `outlier_frac` of
    weights are kept in fp16 and excluded from quantization (SpQR / AWQ-style).

    LLM weight distributions are heavy-tailed; a vanishing fraction of entries
    carries disproportionate output influence. Spending ~0.1% of the matrix at
    full precision removes the tail that was forcing every group's scale wide,
    which improves the *other* 99.9% at almost no memory cost.
    """
    Wf = W.float()
    k = max(1, int(outlier_frac * Wf.numel()))
    flat = Wf.abs().flatten()
    thresh = torch.topk(flat, k, largest=True).values.min()
    mask = Wf.abs() >= thresh

    kept = torch.where(mask, Wf, torch.zeros_like(Wf))
    rest = torch.where(mask, torch.zeros_like(Wf), Wf)
    return (quantize_grouped(rest, bits, group_size).float() + kept).to(W.dtype)


def quantize_grouped_with_outliers_bytes(W: torch.Tensor, bits: int, group_size: int = 64,
                                          outlier_frac: float = 0.001) -> int:
    d_out, d_in = W.shape
    k = max(1, int(outlier_frac * d_out * d_in))
    # each outlier costs an fp16 value plus a 32-bit index
    return quantize_grouped_bytes(W, bits, group_size) + k * (2 + 4)


# ---------------------------------------------------------------------------
# 4. Low-rank + quantized residual -- fixes objective (2)
# ---------------------------------------------------------------------------


def lowrank_plus_quantized_residual(W: torch.Tensor, X: torch.Tensor, r: int,
                                     bits: int, weighted: bool = True) -> torch.Tensor:
    """W_hat = AB + Q_bits(W - AB).

    The residual is *stored coarsely*, not discarded. This should beat both
    of its parts for a principled reason: after the dominant low-rank
    structure is removed, what remains is far closer to i.i.d. Gaussian than
    W itself, and Gaussian-ish data is precisely what a uniform quantizer
    handles well (no heavy tail to waste code space on). So the low-rank term
    captures the structure quantization handles badly, and the quantizer
    captures the diffuse remainder low-rank handles badly.
    """
    lowrank = (activation_weighted_svd(W, X, r) if weighted
               else pca_projection(W, X, r))
    residual = W - lowrank
    return lowrank + quantize_uniform(residual, bits)


def lowrank_plus_quantized_residual_bytes(W: torch.Tensor, r: int, bits: int) -> int:
    d_out, d_in = W.shape
    return (d_in * r + d_out * r) * 2 + (d_out * d_in * bits) // 8 + d_out * 2


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def relative_output_error(W: torch.Tensor, W_hat: torch.Tensor,
                           X: torch.Tensor) -> float:
    """mean_i ||W_hat x_i - W x_i|| / ||W x_i||.

    The same metric PHASE0_RESULTS.md reports, so numbers are directly
    comparable to the original measurement.
    """
    Xf = X.float()
    y = Xf @ W.float().T
    y_hat = Xf @ W_hat.float().T
    num = torch.linalg.vector_norm(y_hat - y, dim=1)
    den = torch.linalg.vector_norm(y, dim=1).clamp_min(1e-12)
    return float((num / den).mean().item())
