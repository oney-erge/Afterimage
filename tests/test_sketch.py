import torch

from afterimage.runtime.gate import GlobalController, JLGate
from afterimage.runtime.sketch import AfterimageLayer
from afterimage.runtime.tiers import Tier, TieredStore



import pytest

pytestmark = pytest.mark.archive  # Phase-0 subspace-activation-cache branch, killed per its own gate
def make_layer(tmp_path, d_in, d_out, max_rank, seed=0):
    torch.manual_seed(seed)
    W = torch.randn(d_out, d_in) / (d_in ** 0.5)
    store = TieredStore(tmp_path / "nvme")
    store.write_nvme("W0", W)
    gate = JLGate(d_in, d_out, m=32, seed=seed + 1)
    ctrl = GlobalController(lam=1e-4)
    layer = AfterimageLayer("W0", d_in, d_out, max_rank, store, gate, ctrl)
    return layer, W, store


def test_hits_reproduce_exact_output_for_subspace_activations(tmp_path):
    """The central correctness claim (HYPOTHESIS.md #3(a), IMPLEMENTATION_PLAN
    #10.1): once the basis has learned a rank-r activation subspace, cache
    hits must reproduce W @ x to machine precision -- not approximately, but
    exactly, because for activations confined to span(U), M @ (U^T x) is an
    algebraic identity, not an approximation."""
    torch.manual_seed(10)
    d_in, d_out, r, max_rank = 60, 90, 10, 32
    layer, W, store = make_layer(tmp_path, d_in, d_out, max_rank, seed=10)

    basis_gen = torch.randn(d_in, r)
    q, _ = torch.linalg.qr(basis_gen)

    # Warm-up: force enough misses to let the basis learn the subspace.
    for _ in range(400):
        x = q @ torch.randn(r)
        layer.forward(x)

    store.reset_stats()
    max_rel_err = 0.0
    hits_seen = 0
    for _ in range(200):
        x = q @ torch.randn(r)
        y_cached = layer.forward(x)
        y_exact = W @ x
        rel_err = (y_cached - y_exact).norm().item() / y_exact.norm().item()
        max_rel_err = max(max_rel_err, rel_err)

    assert layer.hit_rate > 0.9, f"hit rate too low after warm-up: {layer.hit_rate}"
    assert max_rel_err < 1e-4, f"cached output diverged from exact: {max_rel_err}"
    assert store.stats[Tier.NVME].read_count < 20, "warmed-up subspace should rarely miss"


def test_out_of_subspace_activation_always_computed_exactly(tmp_path):
    """A miss must return the EXACT W @ x, not the cached approximation --
    correctness is never sacrificed, only I/O is."""
    d_in, d_out, max_rank = 40, 60, 16
    layer, W, store = make_layer(tmp_path, d_in, d_out, max_rank, seed=20)

    torch.manual_seed(21)
    x = torch.randn(d_in)
    y = layer.forward(x)  # first call is always a miss (gate uncalibrated)
    y_exact = W @ x
    assert torch.allclose(y, y_exact, atol=1e-5)
    assert layer.misses == 1


def test_hit_rate_rises_over_a_session_on_narrow_workload(tmp_path):
    """Reproduces the qualitative claim of the whole project (README.md):
    hit rate should climb as a session progresses on a workload confined to
    a narrow subspace, and plateau once the basis saturates."""
    torch.manual_seed(30)
    d_in, d_out, r, max_rank = 50, 70, 8, 24
    layer, W, store = make_layer(tmp_path, d_in, d_out, max_rank, seed=30)
    basis_gen = torch.randn(d_in, r)
    q, _ = torch.linalg.qr(basis_gen)

    window = 50
    rates = []
    for block in range(6):
        hits_before = layer.hits
        misses_before = layer.misses
        for _ in range(window):
            x = q @ torch.randn(r)
            layer.forward(x)
        h = layer.hits - hits_before
        m = layer.misses - misses_before
        rates.append(h / (h + m))

    assert rates[0] < rates[-1], f"hit rate did not rise: {rates}"
    assert rates[-1] > 0.85, f"hit rate did not plateau high: {rates}"
