"""build_feature_label_sequences and _flatten_features are pure CPU
functions with no HMM fitting involved -- what matters here is that
features stay temporally correct (X[t] predicts y[t]=bucket at t+1, never
the answer itself) and that short/empty traces degrade cleanly instead of
crashing the larger pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.run_h22_disagreement_hmm import (
    _B2_COLUMNS,
    _B3_COLUMNS,
    _flatten_features,
    build_feature_label_sequences,
    build_observation_sequences,
)

RANK_BUCKETS = (1, 2, 4, 8)
N_SYMBOLS = len(RANK_BUCKETS) + 1


def _row(rank, primary_entropy=0.5, primary_margin=1.0, scout_entropy=0.4,
        scout_margin=1.1, divergence=0.1):
    return {
        "target_rank_under_primary": rank,
        "primary_entropy": primary_entropy, "primary_margin": primary_margin,
        "scout_entropy": scout_entropy, "scout_margin": scout_margin,
        "approx_js_divergence": divergence,
    }


def test_feature_label_sequences_predict_the_next_position_not_the_current_one():
    """X[t]'s history-onehot block must encode bucket AT t, and y[t] must
    be the bucket AT t+1 -- never let X directly contain the label it is
    predicting."""
    trace = {"rows": [_row(1), _row(50), _row(1)]}  # buckets: 0, 4, 0
    pairs = build_feature_label_sequences([trace], RANK_BUCKETS, N_SYMBOLS)
    X, y = pairs[0]
    assert X.shape == (2, len(("e", "m", "e", "m", "d")) + N_SYMBOLS)
    assert y.tolist() == [4, 0]  # bucket at t=1, then bucket at t=2
    # history one-hot for row 0 (predicting y[0]=4) must mark bucket 0 (current)
    history_block = X[0, 5:]
    assert history_block.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    # history one-hot for row 1 (predicting y[1]=0) must mark bucket 4 (current)
    history_block_1 = X[1, 5:]
    assert history_block_1.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_feature_label_sequences_uses_the_correct_raw_feature_values():
    trace = {"rows": [_row(1, primary_entropy=0.7, primary_margin=2.0,
                           scout_entropy=0.6, scout_margin=1.9, divergence=0.15),
                      _row(1)]}
    X, y = build_feature_label_sequences([trace], RANK_BUCKETS, N_SYMBOLS)[0]
    assert X[0, :5].tolist() == pytest.approx([0.7, 2.0, 0.6, 1.9, 0.15])


def test_feature_label_sequences_handles_a_trace_too_short_to_predict_from():
    trace = {"rows": [_row(1)]}  # only 1 row -- nothing to predict
    pairs = build_feature_label_sequences([trace], RANK_BUCKETS, N_SYMBOLS)
    X, y = pairs[0]
    assert X.shape[0] == 0
    assert y.shape[0] == 0


def test_feature_label_sequences_handles_an_empty_trace():
    trace = {"rows": []}
    X, y = build_feature_label_sequences([trace], RANK_BUCKETS, N_SYMBOLS)[0]
    assert X.shape[0] == 0
    assert y.shape[0] == 0


def test_feature_label_sequences_are_aligned_with_observation_sequences():
    """The two sequence types must correspond to the SAME traces in the
    SAME order -- a caller applying one index-based split to both depends
    on this."""
    traces = [{"rows": [_row(1), _row(50)]}, {"rows": [_row(2), _row(1)]}]
    obs = build_observation_sequences(traces, "target_rank_under_primary", RANK_BUCKETS)
    pairs = build_feature_label_sequences(traces, RANK_BUCKETS, N_SYMBOLS)
    assert len(obs) == len(pairs) == 2
    for seq, (X, y) in zip(obs, pairs):
        assert len(y) == len(seq) - 1


# --------------------------------------------------------------- _flatten_features

def test_flatten_features_concatenates_selected_indices_only():
    pairs = [
        (np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]), np.array([0])),
        (np.array([[9.0, 9.0, 9.0, 9.0, 9.0]]), np.array([1])),  # excluded below
        (np.array([[6.0, 7.0, 8.0, 9.0, 10.0]]), np.array([2])),
    ]
    X, y = _flatten_features(pairs, indices=[0, 2], columns=_B2_COLUMNS)
    assert X.shape == (2, 2)
    assert X.tolist() == [[1.0, 2.0], [6.0, 7.0]]
    assert y.tolist() == [0, 2]


def test_flatten_features_b3_columns_include_five_features():
    pairs = [(np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0]]), np.array([0]))]
    X, y = _flatten_features(pairs, indices=[0], columns=_B3_COLUMNS)
    assert X.shape == (1, 5)


def test_flatten_features_skips_empty_sequences_without_crashing():
    pairs = [
        (np.zeros((0, 5)), np.zeros(0, dtype=np.int64)),  # too-short trace
        (np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]), np.array([1])),
    ]
    X, y = _flatten_features(pairs, indices=[0, 1], columns=_B2_COLUMNS)
    assert X.shape == (1, 2)
    assert y.tolist() == [1]


def test_flatten_features_returns_empty_arrays_when_nothing_selected():
    pairs = [(np.zeros((0, 5)), np.zeros(0, dtype=np.int64))]
    X, y = _flatten_features(pairs, indices=[0], columns=_B2_COLUMNS)
    assert X.shape[0] == 0
    assert y.shape[0] == 0
