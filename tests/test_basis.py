import torch

from afterimage.runtime.basis import OnlineBasis



import pytest

pytestmark = pytest.mark.archive  # Phase-0 subspace-activation-cache branch, killed per its own gate
def test_project_zero_rank_returns_full_residual():
    b = OnlineBasis(dim=16, max_rank=8)
    x = torch.randn(16)
    c, x_perp = b.project(x)
    assert c is None
    assert torch.allclose(x_perp, x)


def test_add_reduces_residual_to_near_zero_for_repeated_direction():
    torch.manual_seed(0)
    b = OnlineBasis(dim=32, max_rank=8)
    u = torch.randn(32)
    u = u / u.norm()
    _, x_perp = b.project(u)
    b.add(x_perp)
    _, x_perp2 = b.project(u)
    assert x_perp2.norm().item() < 1e-5


def test_orthogonality_preserved_after_many_adds_with_eviction():
    torch.manual_seed(1)
    dim, max_rank = 64, 12
    b = OnlineBasis(dim=dim, max_rank=max_rank)
    for _ in range(500):
        v = torch.randn(dim)
        _, x_perp = b.project(v)
        b.add(x_perp)
    assert b.rank == max_rank
    assert b.total_evictions > 0
    err = b.orthogonality_error()
    assert err < 1e-3, f"orthogonality drifted: {err}"


def test_rebuild_restores_orthogonality():
    torch.manual_seed(2)
    dim, max_rank = 20, 6
    b = OnlineBasis(dim=dim, max_rank=max_rank)
    for _ in range(200):
        v = torch.randn(dim)
        _, x_perp = b.project(v)
        b.add(x_perp)
    b.rebuild()
    assert b.orthogonality_error() < 1e-6


def test_negligible_residual_is_not_installed():
    torch.manual_seed(3)
    b = OnlineBasis(dim=10, max_rank=5)
    u = torch.randn(10)
    u = u / u.norm()
    _, x_perp = b.project(u)
    b.add(x_perp)
    assert b.rank == 1
    # feeding the exact same direction again should be a no-op add
    _, x_perp2 = b.project(u)
    added, evicted = b.add(x_perp2)
    assert not added
    assert evicted is None
    assert b.rank == 1


def test_subspace_activations_saturate_basis_at_true_rank():
    """If every activation lives in a fixed rank-r subspace, the basis should
    stop growing once it reaches r, and every subsequent activation should
    project to near-zero residual -- this is the property the whole cache
    depends on (HYPOTHESIS.md #3(a))."""
    torch.manual_seed(4)
    dim, r, max_rank = 40, 7, 32
    basis_gen = torch.randn(dim, r)
    q, _ = torch.linalg.qr(basis_gen)

    b = OnlineBasis(dim=dim, max_rank=max_rank)
    for _ in range(300):
        coeffs = torch.randn(r)
        x = q @ coeffs
        c, x_perp = b.project(x)
        b.add(x_perp, x_norm=x.norm().item())

    assert b.rank <= r + 1, f"basis grew to {b.rank}, expected <= {r + 1}"

    residuals = []
    for _ in range(50):
        coeffs = torch.randn(r)
        x = q @ coeffs
        _, x_perp = b.project(x)
        residuals.append(x_perp.norm().item() / x.norm().item())
    assert max(residuals) < 1e-4, f"residual ratio too high: {max(residuals)}"


def test_without_relative_floor_float32_noise_pollutes_the_basis():
    """Documents WHY x_norm-relative filtering matters: without it, repeated
    projection noise at float32 precision (~1e-7 * ||x||) is large enough to
    clear a small absolute floor and gets installed as spurious directions,
    so the basis never saturates at the true subspace rank."""
    torch.manual_seed(4)
    dim, r, max_rank = 40, 7, 32
    basis_gen = torch.randn(dim, r)
    q, _ = torch.linalg.qr(basis_gen)

    b = OnlineBasis(dim=dim, max_rank=max_rank, min_norm_ratio=0.0)
    for _ in range(300):
        x = q @ torch.randn(r)
        _, x_perp = b.project(x)
        b.add(x_perp)  # no x_norm -> falls back to the tiny absolute floor

    assert b.rank > r + 1, "expected float32 noise to pollute the basis without relative filtering"
