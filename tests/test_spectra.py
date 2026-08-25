import torch

from afterimage.probe.spectra import (
    effective_rank,
    functional_error_curve,
    rogue_dimension_gap,
    variance_rank_curve,
)
from afterimage.testing.toy_model import full_rank_inputs, narrow_session_inputs



import pytest

pytestmark = pytest.mark.archive  # Phase-0 subspace-activation-cache branch, killed per its own gate
def test_variance_and_functional_curves_agree_for_clean_low_rank_signal():
    """When there is no rogue-dimension effect (the linear map cares about
    the same directions that carry the variance), variance and functional
    curves should agree: both should be near-saturated by the true rank."""
    torch.manual_seed(0)
    d, r = 40, 6
    X = narrow_session_inputs(n_tokens=500, d_model=d, effective_rank=r, seed=1)
    W = torch.randn(30, d)

    ranks = [2, 4, 6, 8, 12]
    _, var_curve = variance_rank_curve(X, max_rank=max(ranks))
    var_at_r = [var_curve[k - 1] for k in ranks]
    func_curve = functional_error_curve(X, W, ranks)

    assert var_at_r[2] > 0.99, f"variance should saturate by true rank: {var_at_r}"
    assert func_curve[2] < 0.02, f"functional error should vanish by true rank: {func_curve}"


def test_rogue_dimensions_create_a_gap_between_variance_and_functional_curves():
    """Direct validation of HYPOTHESIS.md #3.1: construct activations where a
    few dimensions carry huge variance but a linear map is functionally
    blind to them (zero weight columns), and confirm variance and
    functional curves diverge -- the variance-based criterion looks great
    while the functional criterion says the opposite."""
    torch.manual_seed(2)
    n, d = 800, 50
    n_rogue = 5

    signal = torch.randn(n, d - n_rogue) * 0.1
    rogue = torch.randn(n, n_rogue) * 20.0  # ~200x the variance of the signal dims
    X = torch.cat([rogue, signal], dim=1)

    W = torch.zeros(10, d)
    W[:, n_rogue:] = torch.randn(10, d - n_rogue)  # W ignores the rogue dims entirely

    ranks = [1, 2, 3, 4, 5]
    result = rogue_dimension_gap(X, W, ranks)

    assert result["variance_captured"][-1] > 0.9, (
        f"rogue dims should dominate variance: {result['variance_captured']}"
    )
    assert result["functional_error"][-1] > 0.5, (
        f"but a rank-5 basis built on variance should still miss most of the "
        f"function: {result['functional_error']}"
    )
    assert result["gap"][-1] > 0.4, f"expected a large variance/function gap: {result['gap']}"


def test_effective_rank_distinguishes_narrow_from_full_rank_workloads():
    torch.manual_seed(3)
    d = 60
    narrow = narrow_session_inputs(n_tokens=400, d_model=d, effective_rank=5, seed=4)
    full = full_rank_inputs(n_tokens=400, d_model=d, seed=5)

    er_narrow = effective_rank(narrow)
    er_full = effective_rank(full)

    assert er_narrow < er_full * 0.5, f"narrow={er_narrow} full={er_full}"
    assert er_narrow < 10, f"narrow-session effective rank should be near the true rank: {er_narrow}"
