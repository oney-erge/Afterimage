import torch

from afterimage.runtime.gate import GlobalController, JLGate
from afterimage.runtime.sketch import AfterimageLayer
from afterimage.runtime.tiers import Tier, TieredStore



import pytest

pytestmark = pytest.mark.archive  # Phase-0 subspace-activation-cache branch, killed -- see docs/archive/README.md
def test_batched_verification_fetches_weight_once_per_sweep(tmp_path):
    """The amortization claim itself: verifying a whole draft chain (a batch
    of B candidate positions) that ALL miss must still fetch the weight
    matrix exactly once, not B times -- this is the entire mechanical reason
    speculative verification saves I/O (LITERATURE.md #5)."""
    torch.manual_seed(0)
    d_in, d_out, max_rank, B = 30, 40, 16, 8
    W = torch.randn(d_out, d_in) / (d_in ** 0.5)
    store = TieredStore(tmp_path / "nvme")
    store.write_nvme("W0", W)
    gate = JLGate(d_in, d_out, m=16, seed=1)
    ctrl = GlobalController(lam=1e-4)
    layer = AfterimageLayer("W0", d_in, d_out, max_rank, store, gate, ctrl)

    X = torch.randn(B, d_in)  # unrelated random rows -> every row misses
    store.reset_stats()
    Y = layer.forward_batch(X)

    assert store.stats[Tier.NVME].read_count == 1, "weight fetched more than once for a fully-miss batch"
    assert torch.allclose(Y, X @ W.T, atol=1e-5)
    assert layer.misses == B


def test_batched_verification_reproduces_exact_output_after_warmup(tmp_path):
    torch.manual_seed(1)
    d_in, d_out, r, max_rank, B = 40, 60, 8, 24, 6
    W = torch.randn(d_out, d_in) / (d_in ** 0.5)
    store = TieredStore(tmp_path / "nvme")
    store.write_nvme("W0", W)
    gate = JLGate(d_in, d_out, m=24, seed=2)
    ctrl = GlobalController(lam=1e-5)
    layer = AfterimageLayer("W0", d_in, d_out, max_rank, store, gate, ctrl)

    q, _ = torch.linalg.qr(torch.randn(d_in, r))
    for _ in range(300):
        x = q @ torch.randn(r)
        layer.forward(x)

    X = q @ torch.randn(r, B)
    X = X.T  # (B, d_in)
    Y = layer.forward_batch(X)
    Y_exact = X @ W.T
    rel_err = (Y - Y_exact).norm().item() / Y_exact.norm().item()
    assert rel_err < 1e-3, f"batched hits diverged from exact output: {rel_err}"
    assert layer.hit_rate > 0.85
