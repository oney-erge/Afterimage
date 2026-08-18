"""Rank-vs-error curves (IMPLEMENTATION_PLAN.md Phase 0.1).

For each linear layer, two curves are computed from the same captured
activations:

  - variance_rank_curve: the standard PCA picture, fraction of activation
    variance captured by the top-r principal directions.
  - functional_error_curve: fraction of OUTPUT error (||W x_perp|| relative
    to ||W x||) if only the top-r directions are kept.

HYPOTHESIS.md #3.1 predicts these two curves diverge -- that variance
concentrates in directions (massive activations / rogue dimensions) which
carry little functional weight, so the variance curve looks far more
favourable than the functional one. Reporting both, and their gap, is the
point of this module regardless of which way the numbers land.
"""
from __future__ import annotations

import torch


def variance_rank_curve(X: torch.Tensor, max_rank: int | None = None) -> tuple[list[int], list[float]]:
    Xc = X - X.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    total = var.sum().clamp_min(1e-12)
    cum = torch.cumsum(var, dim=0) / total
    ranks = list(range(1, len(S) + 1))
    if max_rank is not None:
        ranks = ranks[:max_rank]
        cum = cum[:max_rank]
    return ranks, cum.tolist()


def effective_rank(X: torch.Tensor) -> float:
    """exp(Shannon entropy of the normalized singular value distribution) --
    the entropy-based effective-rank definition cited in HYPOTHESIS.md #3.2
    and LITERATURE.md. Higher = energy spread more uniformly across
    directions = harder to compress with a small basis."""
    Xc = X - X.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(Xc, full_matrices=False)
    p = S / S.sum().clamp_min(1e-12)
    p = p.clamp_min(1e-12)
    entropy = -(p * p.log()).sum()
    return torch.exp(entropy).item()


def _fit_basis(X: torch.Tensor, r: int) -> torch.Tensor:
    Xc = X - X.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
    return Vt[:r].T  # (d_in, r)


def functional_error_curve(X: torch.Tensor, W: torch.Tensor,
                            ranks: list[int]) -> list[float]:
    """For each r in ranks: fit a rank-r PCA basis on X (batch, open-loop --
    this is the raw n-width picture, not the closed-loop replay), then report
    mean_i ||W x_perp_i|| / ||W x_i|| over the rows of X.

    The SVD of X is computed ONCE and sliced for every rank, not recomputed
    per rank (an earlier version called torch.linalg.svd once per rank via
    _fit_basis in this loop -- len(ranks) redundant full SVDs of the same
    matrix). On a real model this stopped being a rounding error: a wide MLP
    down_proj layer (d_in in the thousands) with several hundred captured
    activation rows made 7 full economy SVDs per layer, and cuSOLVER's SVD
    performance on WSL2/CUDA for these shapes was slow enough that this was
    the actual bottleneck in the real Phase 0 run, not the model forward
    pass. Caught by profiling a real run that was taking far longer than the
    toy-model tests predicted."""
    Xc = X - X.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)

    y_full = X @ W.T  # (N, d_out)
    y_norms = torch.linalg.vector_norm(y_full, dim=1).clamp_min(1e-12)

    out = []
    for r in ranks:
        Q = Vt[:r].T  # (d_in, r), sliced from the single shared SVD
        C = X @ Q  # (N, r)
        X_hat = C @ Q.T
        x_perp = X - X_hat
        y_perp = x_perp @ W.T
        errs = torch.linalg.vector_norm(y_perp, dim=1) / y_norms
        out.append(errs.mean().item())
    return out


def layer_rank_report(X: torch.Tensor, W: torch.Tensor, ranks: list[int]) -> dict:
    """One SVD of X, shared across the variance curve, the functional-error
    curve, and the effective-rank number -- variance_rank_curve(),
    effective_rank(), and functional_error_curve() each compute their own
    SVD of X independently, which is 3 redundant full SVDs of the same
    matrix per layer when a caller (as scripts/run_probe_real.py does) wants
    all three. On a real model, with dozens of layers and several
    workloads, that 3x becomes a real wall-clock cost, not a rounding
    error -- use this instead of calling all three separately."""
    Xc = X - X.mean(dim=0, keepdim=True)
    _, S, Vt = torch.linalg.svd(Xc, full_matrices=False)

    var = S ** 2
    total = var.sum().clamp_min(1e-12)
    cum = (torch.cumsum(var, dim=0) / total).tolist()

    p = (S / S.sum().clamp_min(1e-12)).clamp_min(1e-12)
    eff_rank = torch.exp(-(p * p.log()).sum()).item()

    y_full = X @ W.T
    y_norms = torch.linalg.vector_norm(y_full, dim=1).clamp_min(1e-12)
    func_curve = []
    for r in ranks:
        Q = Vt[:r].T
        x_perp = X - (X @ Q) @ Q.T
        errs = torch.linalg.vector_norm(x_perp @ W.T, dim=1) / y_norms
        func_curve.append(errs.mean().item())

    var_at_ranks = [cum[r - 1] for r in ranks]
    return {
        "ranks": ranks,
        "variance_captured": var_at_ranks,
        "functional_error": func_curve,
        "gap": [v - (1 - f) for v, f in zip(var_at_ranks, func_curve)],
        "effective_rank": eff_rank,
        "d_in": X.shape[1],
        "n_samples": X.shape[0],
    }


def rogue_dimension_gap(X: torch.Tensor, W: torch.Tensor, ranks: list[int]) -> dict:
    """The headline Phase-0 output: variance curve, functional curve, and
    their gap at each rank, plus the entropy-based effective rank."""
    var_ranks, var_curve = variance_rank_curve(X, max_rank=max(ranks))
    func_curve = functional_error_curve(X, W, ranks)
    var_at_ranks = [var_curve[r - 1] for r in ranks]
    return {
        "ranks": ranks,
        "variance_captured": var_at_ranks,
        "functional_error": func_curve,
        "gap": [v - (1 - f) for v, f in zip(var_at_ranks, func_curve)],
        "effective_rank": effective_rank(X),
        "d_in": X.shape[1],
    }
