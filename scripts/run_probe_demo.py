"""Runnable demo of the Phase 0 probe (IMPLEMENTATION_PLAN.md #2) against the
synthetic toy model. This is NOT Phase 0 itself -- Phase 0 means running this
methodology against Gemma-3-27B or Qwen3-32B, which this environment cannot
do (no CUDA, no `transformers`, no downloaded weights; see
IMPLEMENTATION_STATUS.md). What this script demonstrates for real: the
measurement code runs, produces the variance/functional gap the hypothesis
predicts, and the closed-loop harness correctly propagates approximation
error across layers instead of measuring an open-loop fiction.

Run: python scripts/run_probe_demo.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from afterimage.probe.closed_loop import calibrate_bases, closed_loop_error, open_loop_error
from afterimage.probe.spectra import rogue_dimension_gap
from afterimage.testing.toy_model import ToyTransformer, narrow_session_inputs, topic_switch_inputs


def main():
    torch.manual_seed(0)
    print("=" * 70)
    print("Afterimage Phase 0 probe demo (SYNTHETIC toy model, not a real LLM)")
    print("=" * 70)

    d_model, d_ffn, n_layers = 48, 128, 4
    model = ToyTransformer(d_model=d_model, d_ffn=d_ffn, n_layers=n_layers, seed=0)
    model.eval()

    print("\n--- Rogue-dimension gap demonstration (HYPOTHESIS.md #3.1) ---")
    n, n_rogue = 600, 4
    signal = torch.randn(n, d_model - n_rogue) * 0.1
    rogue = torch.randn(n, n_rogue) * 15.0
    X = torch.cat([rogue, signal], dim=1)
    W = torch.zeros(20, d_model)
    W[:, n_rogue:] = torch.randn(20, d_model - n_rogue)
    result = rogue_dimension_gap(X, W, ranks=[1, 2, 3, 4, 6, 8])
    print(f"{'rank':>6} {'var_captured':>14} {'functional_err':>16} {'gap':>8}")
    for r, v, f, g in zip(result["ranks"], result["variance_captured"],
                           result["functional_error"], result["gap"]):
        print(f"{r:>6} {v:>14.3f} {f:>16.3f} {g:>8.3f}")
    print(f"effective_rank (entropy-based): {result['effective_rank']:.2f} / d={result['d_in']}")

    print("\n--- Closed-loop vs open-loop, narrow-session workload ---")
    target_layers = [f"blocks.{i}.up" for i in range(n_layers)]
    calib_x = narrow_session_inputs(n_tokens=400, d_model=d_model, effective_rank=6, seed=1)
    eval_x = narrow_session_inputs(n_tokens=150, d_model=d_model, effective_rank=6, seed=2)
    print(f"{'rank':>6} {'open_loop(mean layer)':>22} {'closed_loop(end-to-end)':>24}")
    for r in [2, 4, 8, 16, d_model]:
        bases = calibrate_bases(model, calib_x, target_layers, rank=r)
        ol = open_loop_error(model, bases, eval_x, target_layers)
        ol_mean = sum(ol.values()) / len(ol)
        cl = closed_loop_error(model, bases, eval_x, target_layers)
        print(f"{r:>6} {ol_mean:>22.4f} {cl:>24.4f}")

    print("\n--- Same measurement, adversarial topic-switching workload ---")
    switch_calib = topic_switch_inputs(n_tokens=400, d_model=d_model, effective_rank=6,
                                        n_topics=4, switch_every=25, seed=3)
    switch_eval = topic_switch_inputs(n_tokens=150, d_model=d_model, effective_rank=6,
                                       n_topics=4, switch_every=25, seed=4)
    print(f"{'rank':>6} {'closed_loop(end-to-end)':>24}")
    for r in [2, 4, 8, 16, d_model]:
        bases = calibrate_bases(model, switch_calib, target_layers, rank=r)
        cl = closed_loop_error(model, bases, switch_eval, target_layers)
        print(f"{r:>6} {cl:>24.4f}")

    print("\nNote: HYPOTHESIS.md #3.3 predicts a single global basis should do "
          "noticeably worse here than on the narrow-session workload above, "
          "and that a clustered (per-topic) basis should recover most of the "
          "gap -- clustering is not implemented in this codebase yet.")


if __name__ == "__main__":
    main()
