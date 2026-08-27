#!/usr/bin/env python3
"""H22 (Persistent Disagreement State), split into the two claims it
actually contains -- see docs/SPECULATION_TREE_RESEARCH.md's H22 section
for the full reasoning behind the split:

  H22a (Temporal Disagreement Persistence): does the SEQUENCE of past
  observed target-vs-Primary disagreement contain temporal structure a
  memoryless "current disagreement bucket alone" model cannot capture?
  Answered by comparing B5 (a fitted DiscreteHMM) against B1 (the
  existing memoryless order-1 baseline). This needs the target's own
  rank observations, which only exist AFTER a target sweep -- it
  establishes that state exists, not that it is usable online.

  H22b (Online Disagreement Prediction): can CHEAP information available
  BEFORE the next target sweep (Primary's own entropy/margin, Scout's
  entropy/margin, an approximate Primary/Scout divergence, recent
  observation history) predict the upcoming disagreement bucket?
  Answered by comparing B2/B3/B4 (multinomial logistic regression on
  increasingly rich cheap-feature sets) against B1. This is the result
  H22's own Critic line (H25 onward) actually needs, since a live system
  cannot know the target's real rank before it runs the sweep that
  produces it.

Consumes an h21_combined_analysis.json from scripts/run_h21_combine_
sources.py (schema_version 2) -- earlier single-source H21 outputs
without primary_entropy/scout_entropy/approx_js_divergence per row
cannot run H22b's baselines and are rejected with a clear error rather
than silently degrading.

Hidden states are NEVER pre-labeled here (state_0, state_1, ...): a
DiscreteHMM's state indices have no inherent identity across independent
fits (state 0 in one run can correspond to state 2 in another) --
interpretive labels like "aligned" or "severe divergence" belong in a
human's post-hoc reading of a specific fitted model's emission
distributions, not in this script's own field names.

Usage:
    python scripts/run_h22_disagreement_hmm.py \\
        --traces results/h21_analysis.json \\
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
    bootstrap_nll_difference_ci,
    constant_frequency_baseline_nll,
    disagreement_bucket,
    evaluate_predictive_nll,
    grouped_k_fold,
    logistic_regression_baseline_nll,
    memoryless_baseline_nll,
    minimum_positions_for_hmm,
    select_n_states_by_held_out_nll,
)

# Feature column layout for the cheap-feature baselines (B2/B3/B4). Every
# feature here must be computable from information available BEFORE the
# target sweep that produces the NEXT position's real rank -- that is
# what makes this a fair test of online predictability, unlike feeding
# the model the target's own answer.
_FEATURE_COLUMNS = ("primary_entropy", "primary_margin", "scout_entropy",
                    "scout_margin", "approx_js_divergence")
_B2_COLUMNS = slice(0, 2)   # primary_entropy, primary_margin
_B3_COLUMNS = slice(0, 5)   # + scout_entropy, scout_margin, approx_js_divergence
# B4 = B3 columns plus a one-hot previous-bucket block, appended separately.


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
        sequences.append(np.array(buckets, dtype=np.int64))
    return sequences


def build_feature_label_sequences(traces: list[dict], rank_buckets: tuple[int, ...],
                                  n_symbols: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per trace: (X, y) where X[t] is the cheap-feature row available
    BEFORE the sweep at position t+1, and y[t] is that next position's
    discretized target_rank_under_primary bucket -- the thing being
    predicted. X's columns are _FEATURE_COLUMNS followed by a one-hot
    encoding of the CURRENT (position t) bucket, which is legitimate
    recent-history information (B4's own "recent target feedback"), not
    the answer being predicted.

    A trace with fewer than 2 rows produces an empty (0-row) pair, not an
    error -- there is nothing to predict from a single observation, and
    the caller filters empty pairs out rather than this function raising.
    """
    pairs = []
    for trace in traces:
        rows = trace["rows"]
        if len(rows) < 2:
            pairs.append((np.zeros((0, len(_FEATURE_COLUMNS) + n_symbols)),
                         np.zeros(0, dtype=np.int64)))
            continue
        buckets = [disagreement_bucket(row["target_rank_under_primary"], rank_buckets)
                  for row in rows]
        X_rows, y_rows = [], []
        for t in range(len(rows) - 1):
            base = [rows[t][col] for col in _FEATURE_COLUMNS]
            history_onehot = [0.0] * n_symbols
            history_onehot[buckets[t]] = 1.0
            X_rows.append(base + history_onehot)
            y_rows.append(buckets[t + 1])
        pairs.append((np.array(X_rows), np.array(y_rows, dtype=np.int64)))
    return pairs


def fit_best_of(sequences: list[np.ndarray], n_states: int, n_symbols: int,
                restarts: int, max_iterations: int, seed: int) -> tuple[DiscreteHMM, float]:
    best_ll, best_hmm = -np.inf, None
    for offset in range(restarts):
        hmm = DiscreteHMM.random_init(n_states, n_symbols, seed=seed + offset)
        history = hmm.fit(sequences, max_iterations=max_iterations)
        if history and history[-1] > best_ll:
            best_ll, best_hmm = history[-1], hmm
    return best_hmm, best_ll


def _flatten_features(pairs: list[tuple[np.ndarray, np.ndarray]],
                      indices: list[int], columns) -> tuple[np.ndarray, np.ndarray]:
    Xs = [pairs[i][0][:, columns] for i in indices if pairs[i][0].shape[0] > 0]
    ys = [pairs[i][1] for i in indices if pairs[i][1].shape[0] > 0]
    if not Xs:
        return np.zeros((0, 0)), np.zeros(0, dtype=np.int64)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", required=True,
                        help="path to an h21_combined_analysis.json (schema_version "
                             "2) from scripts/run_h21_combine_sources.py")
    parser.add_argument("--rank-buckets", default="1,2,4,8",
                        help="ascending rank thresholds; disagreement_bucket's "
                             "own default matches the hypothesis's own "
                             "rank_buckets: [1, 2, 4, 8, inf]")
    parser.add_argument("--candidate-n-states", default="2,3,4,5,6",
                        help="hidden-state counts to consider; the winner is "
                             "chosen on a VALIDATION split, never on the final "
                             "CV test folds -- see select_n_states_by_held_out_nll")
    parser.add_argument("--state-selection-fraction", type=float, default=0.25,
                        help="fraction of traces set aside (by whole trace) for "
                             "state-count selection, split again internally into "
                             "a train/validation pair; the remainder is the pool "
                             "grouped cross-validation runs on")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--allow-insufficient-data", action="store_true",
        help="proceed even though minimum_positions_for_hmm's derived minimum is "
             "not met; the result is still written but flagged as underpowered, "
             "and should not be treated as a real G4 gate evaluation")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        rank_buckets = tuple(int(part.strip()) for part in args.rank_buckets.split(",")
                             if part.strip())
    except ValueError:
        parser.error("--rank-buckets must be a comma-separated list of integers")
    if not rank_buckets or list(rank_buckets) != sorted(rank_buckets):
        parser.error("--rank-buckets must be non-empty and ascending")
    try:
        candidate_n_states = tuple(int(part.strip()) for part in
                                   args.candidate_n_states.split(",") if part.strip())
    except ValueError:
        parser.error("--candidate-n-states must be a comma-separated list of integers")
    if any(n < 1 for n in candidate_n_states):
        parser.error("--candidate-n-states values must be positive")
    if not (0.0 < args.state_selection_fraction < 1.0):
        parser.error("--state-selection-fraction must be in (0, 1)")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    if args.restarts < 1:
        parser.error("--restarts must be positive")

    traces_path = pathlib.Path(args.traces).resolve()
    payload = json.loads(traces_path.read_text(encoding="utf-8"))
    if payload.get("schema_version", 1) < 2:
        parser.error(
            "--traces is schema_version < 2 (missing primary_entropy/scout_entropy/"
            "approx_js_divergence per row): H22b's cheap-feature baselines cannot "
            "run without them. Regenerate with scripts/run_h21_combine_sources.py.")
    traces = payload["traces"]
    n_symbols = len(rank_buckets) + 1

    out = pathlib.Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("refusing to overwrite immutable result: %s" % out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_traces = len(traces)
    if n_traces < 8:
        parser.error(
            "--traces has only %d trace(s); need enough distinct prompts to hold "
            "some out for state selection AND run grouped cross-validation on the "
            "rest" % n_traces)

    observation_sequences = build_observation_sequences(
        traces, "target_rank_under_primary", rank_buckets)
    feature_label_pairs = build_feature_label_sequences(traces, rank_buckets, n_symbols)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_traces)
    n_selection = max(2, round(n_traces * args.state_selection_fraction))
    selection_idx = order[:n_selection].tolist()
    cv_idx = order[n_selection:].tolist()
    if len(cv_idx) < args.cv_folds:
        parser.error(
            "only %d trace(s) remain for cross-validation after reserving %d for "
            "state selection, fewer than --cv-folds=%d -- collect more traces or "
            "lower --cv-folds/--state-selection-fraction" %
            (len(cv_idx), n_selection, args.cv_folds))

    # Split the state-selection pool itself into a train/validation pair,
    # by trace INDEX (not by sequence) so the same split can be applied
    # to both observation_sequences and feature_label_pairs, which must
    # stay aligned to the same underlying traces.
    sel_rng = np.random.default_rng(args.seed + 1)
    sel_order = sel_rng.permutation(len(selection_idx))
    n_sel_val = max(1, round(len(selection_idx) * 0.4))
    sel_val_positions = set(sel_order[:n_sel_val].tolist())
    sel_train_idx = [selection_idx[p] for p in range(len(selection_idx))
                     if p not in sel_val_positions]
    sel_val_idx = [selection_idx[p] for p in range(len(selection_idx))
                  if p in sel_val_positions]

    sel_train_seqs = [observation_sequences[i] for i in sel_train_idx]
    sel_val_seqs = [observation_sequences[i] for i in sel_val_idx]

    sel_train_positions = sum(len(s) for s in sel_train_seqs)
    sel_val_positions_total = sum(len(s) for s in sel_val_seqs)
    max_candidate_states = max(candidate_n_states)
    required_train = minimum_positions_for_hmm(max_candidate_states, n_symbols)
    required_val = minimum_positions_for_hmm(max_candidate_states, n_symbols)
    underpowered = (sel_train_positions < required_train
                    or sel_val_positions_total < required_val)
    if underpowered and not args.allow_insufficient_data:
        raise RuntimeError(
            "state-selection split has %d train / %d validation positions, below "
            "the derived minimum of %d/%d for the largest candidate state count "
            "(%d states, %d symbols) -- see minimum_positions_for_hmm. A "
            "selection this small was empirically shown (tests/"
            "test_speculation_oracle.py) to pick an overfit state count from "
            "noise. Collect more traces, reduce --candidate-n-states, or pass "
            "--allow-insufficient-data to proceed anyway with the result flagged." %
            (sel_train_positions, sel_val_positions_total, required_train,
             required_val, max_candidate_states, n_symbols))

    print("state selection: %d train traces (%d positions), %d validation traces "
         "(%d positions)" % (len(sel_train_idx), sel_train_positions,
                             len(sel_val_idx), sel_val_positions_total))
    state_selection = select_n_states_by_held_out_nll(
        sel_train_seqs, sel_val_seqs, n_symbols, candidate_n_states,
        restarts=args.restarts, max_iterations=args.max_iterations, seed=args.seed)
    selected_n_states = state_selection["selected_n_states"]
    if selected_n_states is None:
        raise RuntimeError("state selection failed to produce a finite validation "
                           "NLL for any candidate state count")
    print("selected n_states=%d" % selected_n_states)

    cv_positions_total = sum(len(observation_sequences[i]) for i in cv_idx)
    required_cv = minimum_positions_for_hmm(selected_n_states, n_symbols)
    cv_underpowered = cv_positions_total < required_cv
    if cv_underpowered and not args.allow_insufficient_data:
        raise RuntimeError(
            "cross-validation pool has %d total positions across %d traces, below "
            "the derived minimum of %d for a %d-state/%d-symbol HMM -- see "
            "minimum_positions_for_hmm. Pass --allow-insufficient-data to proceed "
            "anyway with the result flagged as underpowered." %
            (cv_positions_total, len(cv_idx), required_cv, selected_n_states, n_symbols))

    fold_splits = grouped_k_fold(len(cv_idx), args.cv_folds, seed=args.seed)
    fold_results = []
    for fold_i, (fold_train_pos, fold_test_pos) in enumerate(fold_splits):
        train_trace_idx = [cv_idx[p] for p in fold_train_pos]
        test_trace_idx = [cv_idx[p] for p in fold_test_pos]

        train_obs = [observation_sequences[i] for i in train_trace_idx]
        test_obs = [observation_sequences[i] for i in test_trace_idx]

        b0 = constant_frequency_baseline_nll(train_obs, test_obs, n_symbols)
        b1 = memoryless_baseline_nll(train_obs, test_obs, n_symbols)

        X_train_b2, y_train_b2 = _flatten_features(
            feature_label_pairs, train_trace_idx, _B2_COLUMNS)
        X_test_b2, y_test_b2 = _flatten_features(
            feature_label_pairs, test_trace_idx, _B2_COLUMNS)
        b2 = logistic_regression_baseline_nll(
            X_train_b2, y_train_b2, X_test_b2, y_test_b2, n_symbols, seed=args.seed)

        X_train_b3, y_train_b3 = _flatten_features(
            feature_label_pairs, train_trace_idx, _B3_COLUMNS)
        X_test_b3, y_test_b3 = _flatten_features(
            feature_label_pairs, test_trace_idx, _B3_COLUMNS)
        b3 = logistic_regression_baseline_nll(
            X_train_b3, y_train_b3, X_test_b3, y_test_b3, n_symbols, seed=args.seed)

        X_train_b4, y_train_b4 = _flatten_features(
            feature_label_pairs, train_trace_idx, slice(None))
        X_test_b4, y_test_b4 = _flatten_features(
            feature_label_pairs, test_trace_idx, slice(None))
        b4 = logistic_regression_baseline_nll(
            X_train_b4, y_train_b4, X_test_b4, y_test_b4, n_symbols, seed=args.seed)

        hmm, _ = fit_best_of(train_obs, selected_n_states, n_symbols,
                             args.restarts, args.max_iterations, args.seed)
        b5 = evaluate_predictive_nll(hmm, test_obs)

        fold_results.append({
            "fold": fold_i, "train_traces": len(train_trace_idx),
            "test_traces": len(test_trace_idx),
            "b0_constant_frequency_nll": b0,
            "b1_memoryless_current_bucket_nll": b1,
            "b2_primary_features_nll": b2,
            "b3_primary_scout_features_nll": b3,
            "b4_primary_scout_history_nll": b4,
            "b5_hmm_nll": b5,
        })
        print("fold %d/%d: B0=%.4f B1=%.4f B2=%.4f B3=%.4f B4=%.4f B5(HMM)=%.4f" %
             (fold_i + 1, args.cv_folds, b0, b1, b2, b3, b4, b5))

    def _paired(key_baseline: str, key_model: str) -> list[float]:
        return [fold[key_baseline] - fold[key_model] for fold in fold_results
               if not (np.isnan(fold[key_baseline]) or np.isnan(fold[key_model]))]

    gate_g4a = bootstrap_nll_difference_ci(
        _paired("b1_memoryless_current_bucket_nll", "b5_hmm_nll"),
        n_resamples=args.bootstrap_resamples, seed=args.seed)
    gate_g4b = bootstrap_nll_difference_ci(
        _paired("b1_memoryless_current_bucket_nll", "b4_primary_scout_history_nll"),
        n_resamples=args.bootstrap_resamples, seed=args.seed)
    hmm_vs_cheap_features = bootstrap_nll_difference_ci(
        _paired("b4_primary_scout_history_nll", "b5_hmm_nll"),
        n_resamples=args.bootstrap_resamples, seed=args.seed)

    result = {
        "schema_version": 2,
        "kind": "h22_disagreement_hmm",
        "hypothesis": "h22a-temporal-disagreement-persistence + "
                      "h22b-online-disagreement-prediction",
        "exploratory": True,
        "evidence_level": "L1_mechanism_screen",
        "traces_source": str(traces_path),
        "rank_buckets": list(rank_buckets),
        "n_symbols": n_symbols,
        "n_traces": n_traces,
        "state_selection": {
            "candidate_n_states": list(candidate_n_states),
            "train_traces": len(sel_train_idx), "train_positions": sel_train_positions,
            "validation_traces": len(sel_val_idx),
            "validation_positions": sel_val_positions_total,
            "underpowered": bool(underpowered),
            "candidates": state_selection["candidates"],
            "selected_n_states": selected_n_states,
        },
        "cross_validation": {
            "cv_folds": args.cv_folds, "cv_traces": len(cv_idx),
            "cv_positions": cv_positions_total,
            "underpowered": bool(cv_underpowered),
            "fold_results": fold_results,
        },
        "gates": {
            "G4a_temporal_persistence": {
                "description": "H22a: does a fitted HMM (B5) beat the memoryless "
                               "current-bucket baseline (B1)? Uses the target's own "
                               "past rank observations -- establishes state exists, "
                               "not that it is usable online.",
                "paired_nll_improvement": gate_g4a,
                "threshold_relative_improvement": 0.05,
                "passes": (gate_g4a["mean"] is not None
                          and gate_g4a["excludes_zero"]
                          and gate_g4a["mean"] / max(
                              statistics_mean_of(fold_results, "b1_memoryless_current_bucket_nll"),
                              1e-9) >= 0.05),
            },
            "G4b_online_predictability": {
                "description": "H22b: do CHEAP pre-sweep features (B4) beat the "
                               "same memoryless baseline (B1)? This is the result "
                               "the belief-space Critic line (H25 onward) actually "
                               "needs, since it cannot see the target's real rank "
                               "before the sweep that produces it.",
                "paired_nll_improvement": gate_g4b,
                "threshold_relative_improvement": 0.03,
                "passes": (gate_g4b["mean"] is not None
                          and gate_g4b["excludes_zero"]
                          and gate_g4b["mean"] / max(
                              statistics_mean_of(fold_results, "b1_memoryless_current_bucket_nll"),
                              1e-9) >= 0.03),
            },
        },
        "hmm_vs_cheap_features": {
            "description": "Does the HMM's latent-state modeling (B5) add anything "
                           "BEYOND what cheap online features alone (B4) already "
                           "capture? A near-zero or negative result here means the "
                           "HMM's apparent advantage in G4a is fully explained by "
                           "information B4 already has cheap access to online.",
            "paired_nll_improvement": hmm_vs_cheap_features,
        },
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("\nwrote %s" % out)
    print("G4a (temporal persistence, HMM vs B1): passes=%s" %
         result["gates"]["G4a_temporal_persistence"]["passes"])
    print("G4b (online predictability, B4 vs B1): passes=%s" %
         result["gates"]["G4b_online_predictability"]["passes"])
    return 0


def statistics_mean_of(fold_results: list[dict], key: str) -> float:
    values = [fold[key] for fold in fold_results if not np.isnan(fold[key])]
    return float(np.mean(values)) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
