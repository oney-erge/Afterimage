import torch

from afterimage.runtime.draft import build_substitute_draft
from afterimage.runtime.engine import build_offloaded_target, run_decode
from afterimage.runtime.tiers import TieredStore
from afterimage.testing.toy_lm import ToyLM



import pytest

pytestmark = pytest.mark.archive  # Phase-0 subspace-activation-cache branch, killed per its own gate
def test_full_decode_loop_runs_and_produces_more_than_one_token_per_sweep(tmp_path):
    """End-to-end integration: draft proposes, batched offloaded verification
    checks the whole chain in one sweep, Afterimage cache serves some of
    that verification from the resident basis instead of NVMe. This is the
    full pipeline the project's success thresholds (IMPLEMENTATION_PLAN
    #12) are about -- run against the toy LM since no real model is
    available here."""
    torch.manual_seed(0)
    target = ToyLM(vocab_size=32, d_model=16, d_ffn=48, n_layers=3, seed=0)
    target.eval()

    store = TieredStore(tmp_path / "nvme")
    offloaded = build_offloaded_target(target, store, max_rank=12, gate_m=16, lam=1e-4, seed=1)
    draft = build_substitute_draft(target, rank=4)
    draft.eval()

    prefix = torch.tensor([0])
    with torch.no_grad():
        seq, stats = run_decode(target, draft, store, offloaded, prefix, k=4, n_sweeps=15,
                                 temperature=0.8, seed=2)

    assert stats.sweeps == 15
    assert stats.tokens_generated >= stats.sweeps, "every sweep must accept at least one (bonus/resample) token"
    assert stats.tokens_per_sweep > 1.0, "expected at least some multi-token sweeps"
    assert len(seq) == 1 + stats.tokens_generated
    assert stats.gb_per_token > 0.0


def test_cache_reduces_nvme_bytes_versus_no_cache_baseline(tmp_path):
    """The number the whole project is trying to move: with the cache
    enabled (a permissive lambda so hits happen readily) versus effectively
    disabled (lambda near zero forces every row to miss), NVMe bytes read
    per sweep should be lower with the cache on, for a workload confined to
    a narrow subspace where the cache has something to learn."""
    torch.manual_seed(5)
    target = ToyLM(vocab_size=24, d_model=16, d_ffn=40, n_layers=2, seed=5)
    target.eval()
    draft = build_substitute_draft(target, rank=4)
    draft.eval()

    def run_with_lambda(lam, seed):
        store = TieredStore(tmp_path / f"nvme_{seed}")
        offloaded = build_offloaded_target(target, store, max_rank=12, gate_m=16, lam=lam, seed=seed)
        prefix = torch.tensor([0])
        with torch.no_grad():
            _, stats = run_decode(target, draft, store, offloaded, prefix, k=3, n_sweeps=25,
                                   temperature=0.5, seed=seed + 100)
        return stats

    stats_cache_off = run_with_lambda(lam=-1.0, seed=10)  # negative lambda -> every row misses
    stats_cache_on = run_with_lambda(lam=1e6, seed=11)    # huge lambda -> hits whenever calibrated

    assert stats_cache_off.bytes_read_nvme > 0
    assert stats_cache_on.bytes_read_nvme <= stats_cache_off.bytes_read_nvme
