"""Runnable end-to-end demo of the decode engine: draft proposal, batched
offloaded verification through the Afterimage cache, speculative sampling,
compared against the sequential (AirLLM-equivalent) control baseline.

Run against the synthetic toy LM (no real tokenizer/model available in this
environment -- see IMPLEMENTATION_STATUS.md). This is a demonstration that
the full pipeline actually runs and moves fewer bytes per token, not a claim
about real-model performance.

Run: python scripts/run_engine_demo.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from afterimage.baselines.b3_sequential import prepare_sequential_baseline, run_sequential_baseline
from afterimage.runtime.draft import build_substitute_draft
from afterimage.runtime.engine import build_offloaded_target, run_decode
from afterimage.runtime.tiers import TieredStore
from afterimage.testing.toy_lm import ToyLM


def main():
    torch.manual_seed(0)
    print("=" * 70)
    print("Afterimage engine demo (SYNTHETIC toy LM, not a real model)")
    print("=" * 70)

    target = ToyLM(vocab_size=48, d_model=32, d_ffn=96, n_layers=4, seed=0)
    target.eval()
    n_tokens_target = 60
    prefix = torch.tensor([0])

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)

        print("\n[A] Sequential baseline (AirLLM-equivalent control)")
        store_a = TieredStore(tmp / "nvme_a")
        metas = prepare_sequential_baseline(target, store_a)
        with torch.no_grad():
            _, stats_a = run_sequential_baseline(target, store_a, metas, prefix,
                                                   n_tokens=n_tokens_target, temperature=0.7, seed=1)
        print(f"    tokens generated:  {stats_a.tokens_generated}")
        print(f"    NVMe bytes read:   {stats_a.bytes_read_nvme:,}")
        print(f"    GB / token:        {stats_a.gb_per_token:.6f}")

        print("\n[F] Full: substitute draft + batched verification + Afterimage cache")
        store_f = TieredStore(tmp / "nvme_f")
        offloaded = build_offloaded_target(target, store_f, max_rank=16, gate_m=20, lam=5e-4, seed=2)
        draft = build_substitute_draft(target, rank=6)
        draft.eval()
        n_sweeps = 20
        with torch.no_grad():
            _, stats_f = run_decode(target, draft, store_f, offloaded, prefix, k=4,
                                     n_sweeps=n_sweeps, temperature=0.7, seed=3)
        print(f"    sweeps:            {stats_f.sweeps}")
        print(f"    tokens generated:  {stats_f.tokens_generated}")
        print(f"    tokens / sweep:    {stats_f.tokens_per_sweep:.2f}")
        print(f"    NVMe bytes read:   {stats_f.bytes_read_nvme:,}")
        print(f"    GB / token:        {stats_f.gb_per_token:.6f}")

        hit_rates = [(f"block{i}.up", b.up.hit_rate) for i, b in enumerate(offloaded.blocks)]
        hit_rates += [(f"block{i}.down", b.down.hit_rate) for i, b in enumerate(offloaded.blocks)]
        print("\n    per-layer cache hit rate:")
        for name, hr in hit_rates:
            print(f"      {name:>12}: {hr:.2%}")

        print(f"\n--- GB/token improvement: {stats_a.gb_per_token / stats_f.gb_per_token:.2f}x ---")
        print("All of that improvement came from speculative batching (fetch once per")
        print("sweep, not once per token) -- hit rate is 0% because this toy LM's")
        print("nn.Embedding table is unstructured random, so there is no activation")
        print("subspace for the cache to learn. That is itself an honest result: it is")
        print("exactly the D-vs-B ablation IMPLEMENTATION_PLAN.md #9 calls for, showing")
        print("the cache contributes nothing on a workload with no exploitable structure.")

        print("\n[F'] Same run, but with a LOW-RANK embedding table")
        print("     (the optimistic case HYPOTHESIS.md #2 depends on -- see")
        print("     tests/test_sketch.py for the controlled version of this result)")
        low_rank_embed_basis, _ = torch.linalg.qr(torch.randn(target.embed.embedding_dim, 6))
        coeffs = torch.randn(target.vocab_size, 6)
        with torch.no_grad():
            target.embed.weight.copy_(coeffs @ low_rank_embed_basis.T)

        store_fp = TieredStore(tmp / "nvme_fprime")
        offloaded_p = build_offloaded_target(target, store_fp, max_rank=16, gate_m=20, lam=5e-4, seed=2)
        draft_p = build_substitute_draft(target, rank=6)
        draft_p.eval()
        with torch.no_grad():
            _, stats_fp = run_decode(target, draft_p, store_fp, offloaded_p, prefix, k=4,
                                      n_sweeps=n_sweeps, temperature=0.7, seed=3)
        print(f"    GB / token:        {stats_fp.gb_per_token:.6f}")
        hit_rates_p = [(f"block{i}.up", b.up.hit_rate) for i, b in enumerate(offloaded_p.blocks)]
        for name, hr in hit_rates_p:
            print(f"      {name:>12}: {hr:.2%}")
        print(f"    GB/token improvement over [A]: {stats_a.gb_per_token / stats_fp.gb_per_token:.2f}x")
        print("\n    Note the hit rate collapses to ~0 after block0: a low-rank EMBEDDING")
        print("    table does not stay low-rank once GELU and LayerNorm act on it. This")
        print("    is a small, toy-scale instance of exactly the concern HYPOTHESIS.md")
        print("    #6.2 raises from the real-model literature (residual streams measuring")
        print("    ~90% effective rank) -- low input rank does not by itself guarantee")
        print("    low rank at depth, which is precisely why Phase 0 must be measured on")
        print("    a real model, per-layer, rather than assumed from the embedding alone.")


if __name__ == "__main__":
    main()
