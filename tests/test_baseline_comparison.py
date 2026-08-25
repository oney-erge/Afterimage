import torch

from afterimage.baselines.b3_sequential import prepare_sequential_baseline, run_sequential_baseline
from afterimage.runtime.draft import build_substitute_draft
from afterimage.runtime.engine import build_offloaded_target, run_decode
from afterimage.runtime.tiers import TieredStore
from afterimage.testing.toy_lm import ToyLM



import pytest

pytestmark = pytest.mark.archive  # Phase-0 subspace-activation-cache branch, killed per its own gate
def test_gb_per_token_improves_across_the_test_matrix(tmp_path):
    """A cut-down version of the test matrix in IMPLEMENTATION_PLAN.md #9:
    row A (sequential, no speculation, no cache) vs row F-equivalent
    (speculative batched verification + a permissive Afterimage cache).
    GB/accepted-token is the project's primary metric (IMPLEMENTATION_PLAN
    #3.1) precisely because it is comparable across these very different
    execution strategies."""
    torch.manual_seed(0)
    target = ToyLM(vocab_size=20, d_model=16, d_ffn=32, n_layers=2, seed=0)
    target.eval()

    store_a = TieredStore(tmp_path / "nvme_a")
    metas = prepare_sequential_baseline(target, store_a)
    prefix = torch.tensor([0])
    with torch.no_grad():
        _, stats_a = run_sequential_baseline(target, store_a, metas, prefix, n_tokens=30,
                                              temperature=0.7, seed=1)

    store_f = TieredStore(tmp_path / "nvme_f")
    offloaded = build_offloaded_target(target, store_f, max_rank=12, gate_m=16, lam=1e-4, seed=2)
    draft = build_substitute_draft(target, rank=4)
    draft.eval()
    with torch.no_grad():
        _, stats_f = run_decode(target, draft, store_f, offloaded, prefix, k=4, n_sweeps=15,
                                 temperature=0.7, seed=3)

    assert stats_a.gb_per_token > 0.0
    assert stats_f.gb_per_token > 0.0
    assert stats_f.gb_per_token < stats_a.gb_per_token, (
        f"expected the speculative+cached engine to move fewer bytes per "
        f"token: sequential={stats_a.gb_per_token:.2e} full={stats_f.gb_per_token:.2e}"
    )
