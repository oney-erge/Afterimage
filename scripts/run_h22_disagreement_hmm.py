#!/usr/bin/env python3
"""H22 (Persistent Disagreement State): is draft/target disagreement
temporally predictable beyond the current position's own confidence?

Consumes the per-position traces scripts/run_h21_multi_source_oracle.py
(or any script writing the same {"traces": [{"case_id", "rows": [...]}]}
shape) already collected -- this needs no new GPU work of its own, only
the discretized signal (target_rank_under_primary, bucketed via
afterimage.runtime.speculation_oracle.disagreement_bucket) from an
existing trace file.

Fits a DiscreteHMM with --n-states hidden regimes (H22's own framing:
aligned / weak disagreement / broad ambiguity / severe divergence) on a
TRAIN split of traces, held out by whole trace (not by position within a
trace, which would leak information across the train/held-out boundary
through temporal continuity). Reports the actual G4 gate: does the fitted
HMM's one-step-ahead predictive log-likelihood on held-out traces beat a
memoryless baseline (afterimage.runtime.speculation_oracle.
memoryless_baseline_nll -- the empirical next-observation distribution
given only the CURRENT observation, "current draft confidence alone" in
the hypothesis's own words)? If not: kill the POMDP/MPC branch of the
speculation-tree research line (docs/SPECULATION_TREE_RESEARCH.md) rather
than forcing belief-space planning onto data that does not support it.

Usage:
    python scripts/run_h22_disagreement_hmm.py \\
        --traces results/h21_multi_source_oracle.json \\
        --out results/h22_disagreement_hmm.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from afterimage.runtime.speculation_oracle import (
    DiscreteHMM,
    disagreement_bucket,
    evaluate_predictive_nll,
    memoryless_baseline_nll,
)


def build_observation_sequences(traces: list[dict], observation_field: str,
                                rank_buckets: tuple[int, ...]) -> list[np.ndarray]:
    """One discretized observation sequence per trace -- positions within
    a trace are temporally ordered and real; different traces (different
    prompts) share no state, so they must never be concatenated into one
    sequence."""
    sequences = []
    for trace in traces:
        buckets = [disagreement_bucket(row[observation_field], rank_buckets)
                  for row in trace["rows"]]
        if buckets:
            sequences.append(np.array(buckets, dtype=np.int64))
    return sequences


def split_by_trace(sequences: list[np.ndarray], held_out_fraction: float,
                   seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sequences))
    n_held_out = max(1, round(len(sequences) * held_out_fraction)) if sequences else 0
    held_out_idx = set(order[:n_held_out].tolist())
    train = [sequences[i] for i in range(len(sequences)) if i not in held_out_idx]
    held_out = [sequences[i] for i in range(len(sequences)) if i in held_out_idx]
    return train, held_out


def fit_best_of(sequences: list[np.ndarray], n_states: int, n_symbols: int,
                restarts: int, max_iterations: int, seed: int) -> tuple[DiscreteHMM, float]:
    best_ll, best_hmm = -np.inf, None
    for offset in range(restarts):
        hmm = DiscreteHMM.random_init(n_states, n_symbols, seed=seed + offset)
        history = hmm.fit(sequences, max_iterations=max_iterations)
        if history and history[-1] > best_ll:
            best_ll, best_hmm = history[-1], hmm
    return best_hmm, best_ll


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", required=True,
                        help="path to an H21-shaped result JSON with a "
                             "top-level \"traces\" list")
    parser.add_argument("--observation-field", default="target_rank_under_primary",
                        choices=["target_rank_under_primary", "target_rank_under_scout"])
    parser.add_argument("--rank-buckets", default="1,2,4,8",
                        help="ascending rank thresholds; disagreement_bucket's "
                             "own default matches the hypothesis's own "
                             "rank_buckets: [1, 2, 4, 8, inf]")
    parser.add_argument("--n-states", type=int, default=4)
    parser.add_argument("--held-out-fraction", type=float, default=0.3)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        rank_buckets = tuple(int(part.strip()) for part in args.rank_buckets.split(",")
                             if part.strip())
    except ValueError:
        parser.error("--rank-buckets must be a comma-separated list of integers")
    if not rank_buckets or list(rank_buckets) != sorted(rank_buckets):
        parser.error("--rank-buckets must be non-empty and ascending")
    if args.n_states < 1:
        parser.error("--n-states must be positive")
    if not (0.0 < args.held_out_fraction < 1.0):
        parser.error("--held-out-fraction must be in (0, 1)")
    if args.restarts < 1:
        parser.error("--restarts must be positive")

    traces_path = pathlib.Path(args.traces).resolve()
    traces_payload = json.loads(traces_path.read_text(encoding="utf-8"))
    traces = traces_payload["traces"]
    if len(traces) < 4:
        parser.error(
            "--traces has only %d trace(s); need enough distinct prompts to "
            "hold some out by trace and still fit anything on the rest" %
            len(traces))

    out = pathlib.Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("refusing to overwrite immutable result: %s" % out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_symbols = len(rank_buckets) + 1
    sequences = build_observation_sequences(traces, args.observation_field, rank_buckets)
    train_sequences, held_out_sequences = split_by_trace(
        sequences, args.held_out_fraction, args.seed)
    print("traces: %d total, %d train, %d held out" %
         (len(sequences), len(train_sequences), len(held_out_sequences)))

    hmm, train_log_likelihood = fit_best_of(
        train_sequences, args.n_states, n_symbols, args.restarts,
        args.max_iterations, args.seed)

    hmm_nll = evaluate_predictive_nll(hmm, held_out_sequences)
    baseline_nll = memoryless_baseline_nll(train_sequences, held_out_sequences, n_symbols)
    both_defined = not (np.isnan(hmm_nll) or np.isnan(baseline_nll))
    beats_baseline = (hmm_nll < baseline_nll) if both_defined else None
    relative_improvement = (
        (baseline_nll - hmm_nll) / baseline_nll
        if both_defined and baseline_nll != 0.0 else None)

    result = {
        "schema_version": 1,
        "kind": "h22_disagreement_hmm",
        "hypothesis": "h22-persistent-disagreement-state",
        "exploratory": True,
        "evidence_level": "L1_mechanism_screen",
        "traces_source": str(traces_path),
        "observation_field": args.observation_field,
        "rank_buckets": list(rank_buckets),
        "n_states": args.n_states,
        "n_symbols": n_symbols,
        "held_out_fraction": args.held_out_fraction,
        "restarts": args.restarts,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "train_traces": len(train_sequences),
        "held_out_traces": len(held_out_sequences),
        "train_log_likelihood": train_log_likelihood,
        "hmm_held_out_predictive_nll": hmm_nll,
        "memoryless_baseline_held_out_nll": baseline_nll,
        "hmm_beats_memoryless_baseline": beats_baseline,
        "relative_nll_improvement": relative_improvement,
        "fitted_transition_matrix": np.exp(hmm.log_A).tolist(),
        "fitted_emission_matrix": np.exp(hmm.log_B).tolist(),
        "fitted_initial_distribution": np.exp(hmm.log_pi).tolist(),
        "gate": "G4: continue the belief-space planning line only if "
               "hmm_beats_memoryless_baseline is true with a real margin, "
               "not a marginal one -- see docs/SPECULATION_TREE_RESEARCH.md",
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote %s" % out)
    print("HMM held-out NLL: %.4f  |  baseline: %.4f  |  beats baseline: %s" %
         (hmm_nll, baseline_nll, beats_baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
