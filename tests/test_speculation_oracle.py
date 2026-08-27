"""H21 (Multi-Source Oracle Headroom) and H22 (Persistent Disagreement
State) are both pure offline statistics -- no CUDA, no live speculative
execution, nothing this suite cannot fully verify on CPU. The DiscreteHMM
tests in particular exist because HMM implementations are easy to get
subtly wrong (log-space underflow, an EM update that does not actually
increase likelihood, a predictive distribution that silently uses the
wrong time index) in ways that would not show up as a crash -- only as
quietly wrong numbers. Verified empirically against synthetic data before
being written here (see the session record); the specific parameters
below were chosen for a fast, stable, non-flaky margin, not because they
looked good once.
"""
from __future__ import annotations

import numpy as np
import pytest

from afterimage.runtime.speculation_oracle import (
    DiscreteHMM,
    _logsumexp,
    compute_oracle_coverage_stats,
    disagreement_bucket,
    evaluate_predictive_nll,
    memoryless_baseline_nll,
    rank_of_token,
)


# --------------------------------------------------------------- rank_of_token

def test_rank_of_token_is_one_for_the_argmax():
    logits = np.array([1.0, 5.0, 2.0, 0.0])
    assert rank_of_token(logits, token_id=1) == 1


def test_rank_of_token_counts_strictly_greater_entries():
    logits = np.array([1.0, 5.0, 2.0, 0.0])
    assert rank_of_token(logits, token_id=2) == 2   # one value (5.0) exceeds it
    assert rank_of_token(logits, token_id=0) == 3   # two values (5.0, 2.0) exceed it
    assert rank_of_token(logits, token_id=3) == 4   # three values exceed it


def test_rank_of_token_ties_share_the_same_rank():
    logits = np.array([3.0, 3.0, 1.0])
    assert rank_of_token(logits, token_id=0) == rank_of_token(logits, token_id=1) == 1


# ----------------------------------------------------------- disagreement_bucket

def test_disagreement_bucket_boundaries():
    buckets = (1, 2, 4, 8)
    assert disagreement_bucket(1, buckets) == 0
    assert disagreement_bucket(2, buckets) == 1
    assert disagreement_bucket(3, buckets) == 2
    assert disagreement_bucket(4, buckets) == 2
    assert disagreement_bucket(5, buckets) == 3
    assert disagreement_bucket(8, buckets) == 3
    assert disagreement_bucket(9, buckets) == 4
    assert disagreement_bucket(10_000, buckets) == 4


# ------------------------------------------------------- compute_oracle_coverage_stats

def _oracle_row(primary_rank, scout_rank, primary_topk=None, scout_topk=None):
    return {
        "target_rank_under_primary": primary_rank,
        "target_rank_under_scout": scout_rank,
        "primary_topk": primary_topk or [],
        "scout_topk": scout_topk or [],
    }


def test_coverage_reports_the_fraction_within_k():
    rows = [_oracle_row(1, 1), _oracle_row(1, 1), _oracle_row(20, 20), _oracle_row(20, 20)]
    stats = compute_oracle_coverage_stats(rows, top_k_values=[8])
    assert stats["top_k"][8]["primary_coverage"] == pytest.approx(0.5)
    assert stats["top_k"][8]["scout_coverage"] == pytest.approx(0.5)


def test_union_coverage_can_exceed_either_source_alone():
    """The whole point of H21: Scout covering what Primary misses (and
    vice versa) raises union coverage above both individual sources."""
    rows = [
        _oracle_row(primary_rank=1, scout_rank=50),   # primary covers, scout doesn't
        _oracle_row(primary_rank=50, scout_rank=1),   # scout covers, primary doesn't
        _oracle_row(primary_rank=50, scout_rank=50),  # neither covers
    ]
    stats = compute_oracle_coverage_stats(rows, top_k_values=[8])
    point = stats["top_k"][8]
    assert point["primary_coverage"] == pytest.approx(1 / 3)
    assert point["scout_coverage"] == pytest.approx(1 / 3)
    assert point["union_coverage"] == pytest.approx(2 / 3)


def test_conditional_rescue_recall_is_the_useful_scout_signal():
    """Matches the hypothesis's own example: a scout that individually
    looks WORSE than the primary can still be worth having if it rescues
    a meaningful fraction of the primary's specific misses."""
    rows = [
        _oracle_row(1, 1),      # primary already covers -- not a rescue opportunity
        _oracle_row(1, 1),
        _oracle_row(50, 1),     # primary misses, scout rescues
        _oracle_row(50, 50),    # primary misses, scout also misses
    ]
    stats = compute_oracle_coverage_stats(rows, top_k_values=[8])
    point = stats["top_k"][8]
    assert point["primary_miss_count"] == 2
    assert point["conditional_rescue_recall"] == pytest.approx(0.5)


def test_conditional_rescue_recall_is_none_when_primary_never_misses():
    """Must read as "nothing to rescue", not as a misleading 0.0 that
    would look like the scout failed every opportunity when there were
    none."""
    rows = [_oracle_row(1, 1), _oracle_row(2, 1)]
    stats = compute_oracle_coverage_stats(rows, top_k_values=[8])
    assert stats["top_k"][8]["primary_miss_count"] == 0
    assert stats["top_k"][8]["conditional_rescue_recall"] is None


def test_jaccard_overlap_uses_the_actual_topk_sets():
    rows = [_oracle_row(1, 1, primary_topk=[1, 2, 3], scout_topk=[2, 3, 4])]
    stats = compute_oracle_coverage_stats(rows, top_k_values=[8])
    # intersection {2,3} / union {1,2,3,4} = 0.5
    assert stats["jaccard_overlap_at_collection_k"] == pytest.approx(0.5)


def test_empty_rows_does_not_crash():
    stats = compute_oracle_coverage_stats([], top_k_values=[8])
    assert stats["rows"] == 0


# ------------------------------------------------------------------- DiscreteHMM

def _sticky_two_state_hmm_sequences(seed, n_train, n_held_out, length,
                                    stay_prob=0.95, emit_prob=0.78):
    """A 2-state HMM with persistent (sticky) hidden state and NOISY,
    only-weakly-informative single-step emissions: exactly the regime
    where a memoryless order-1 observation table cannot fully recover
    what accumulating evidence across several steps (what the HMM's own
    filtering does) can. This is what H22's gate is actually asking
    whether Afterimage's real draft/target disagreement resembles.
    """
    rng = np.random.default_rng(seed)
    A_true = np.array([[stay_prob, 1 - stay_prob], [1 - stay_prob, stay_prob]])
    B_true = np.array([[emit_prob, 1 - emit_prob], [1 - emit_prob, emit_prob]])

    def simulate(n_seqs):
        seqs = []
        for _ in range(n_seqs):
            state = rng.integers(0, 2)
            obs = []
            for _ in range(length):
                obs.append(rng.choice(2, p=B_true[state]))
                state = rng.choice(2, p=A_true[state])
            seqs.append(np.array(obs))
        return seqs

    return simulate(n_train), simulate(n_held_out)


def _fit_best_of(sequences, n_states, n_symbols, restarts, max_iterations):
    best_ll, best_hmm = -np.inf, None
    for seed in range(restarts):
        hmm = DiscreteHMM.random_init(n_states, n_symbols, seed=seed)
        history = hmm.fit(sequences, max_iterations=max_iterations)
        if history[-1] > best_ll:
            best_ll, best_hmm = history[-1], hmm
    return best_hmm


def test_forward_and_backward_agree_on_total_log_likelihood():
    """A real, model-independent HMM invariant: P(obs) computed from
    alpha[T-1] alone must equal P(obs) computed from alpha[t]*beta[t] at
    ANY time slice t, not just the last one. If forward and backward
    disagree, at least one of them has a real bug."""
    train, _ = _sticky_two_state_hmm_sequences(seed=0, n_train=5, n_held_out=0, length=12)
    hmm = DiscreteHMM.random_init(2, 2, seed=1)
    obs = train[0]
    log_alpha = hmm._forward_log(obs)
    log_beta = hmm._backward_log(obs)
    ll_from_forward = hmm.log_likelihood(obs)
    for t in range(len(obs)):
        ll_at_t = float(_logsumexp(log_alpha[t] + log_beta[t], axis=0))
        assert ll_at_t == pytest.approx(ll_from_forward, abs=1e-6)


def test_baum_welch_never_decreases_likelihood():
    train, _ = _sticky_two_state_hmm_sequences(seed=0, n_train=10, n_held_out=0, length=20)
    hmm = DiscreteHMM.random_init(2, 2, seed=2)
    history = hmm.fit(train, max_iterations=30)
    assert len(history) >= 2
    for previous, current in zip(history, history[1:]):
        assert current >= previous - 1e-6, (
            "EM decreased log-likelihood: %r -> %r" % (previous, current))


def test_predictive_distribution_is_a_valid_probability_distribution():
    train, _ = _sticky_two_state_hmm_sequences(seed=0, n_train=5, n_held_out=0, length=10)
    hmm = DiscreteHMM.random_init(2, 2, seed=3)
    hmm.fit(train, max_iterations=10)
    posteriors = hmm.filtered_state_posterior(train[0])
    predicted = hmm.one_step_predictive_distribution(posteriors[0][None, :])[0]
    assert predicted.sum() == pytest.approx(1.0, abs=1e-8)
    assert (predicted >= 0).all()


def test_baum_welch_recovers_parameters_close_to_the_true_generator():
    """Not exact recovery (EM only guarantees a local optimum, and label
    order is arbitrary) -- but a well-separated 2-state generator with
    this much data should be recovered closely enough that either
    (state 0, state 1) or the swapped labeling matches the true stay/
    emit probabilities within a real margin."""
    train, _ = _sticky_two_state_hmm_sequences(seed=0, n_train=30, n_held_out=0, length=30)
    hmm = _fit_best_of(train, n_states=2, n_symbols=2, restarts=3, max_iterations=40)
    fitted_stay = np.diag(np.exp(hmm.log_A))
    # True stay_prob is 0.95 on both states; whichever labeling, both
    # diagonal entries should be well above chance (0.5) and close to it.
    assert all(stay > 0.8 for stay in fitted_stay), (
        "fitted transition matrix does not show the true sticky-state "
        "structure: %r" % fitted_stay)


def test_hmm_beats_memoryless_baseline_on_data_with_real_hidden_structure():
    """The actual G4 gate, exercised end to end: on data engineered to
    have genuine persistent hidden state with noisy single-step emissions
    (see _sticky_two_state_hmm_sequences' own docstring for why a
    memoryless table cannot fully capture this), the HMM's held-out
    one-step predictive NLL must be lower than the memoryless baseline's.
    If this ever failed, either the HMM fit or the predictive/baseline
    comparison itself would have a real bug -- this is not a marginal or
    seed-sensitive result (verified stable across seeds 0/1/2/3/5/9
    during development)."""
    train, held_out = _sticky_two_state_hmm_sequences(
        seed=0, n_train=30, n_held_out=15, length=30)
    hmm = _fit_best_of(train, n_states=2, n_symbols=2, restarts=3, max_iterations=40)
    hmm_nll = evaluate_predictive_nll(hmm, held_out)
    baseline_nll = memoryless_baseline_nll(train, held_out, n_symbols=2)
    assert hmm_nll < baseline_nll, (
        "HMM (%.4f) did not beat the memoryless baseline (%.4f) on data "
        "with real hidden structure" % (hmm_nll, baseline_nll))


def test_memoryless_baseline_nll_handles_an_unseen_transition_gracefully():
    """Additive smoothing must keep an unseen (obs_t, obs_{t+1}) pair in
    held-out data from producing an infinite penalty."""
    train = [np.array([0, 0, 0, 0])]  # never observes symbol 1 at all
    held_out = [np.array([0, 1])]     # held-out contains the unseen symbol
    nll = memoryless_baseline_nll(train, held_out, n_symbols=2)
    assert np.isfinite(nll)


def test_evaluate_predictive_nll_skips_sequences_too_short_to_predict_from():
    train, _ = _sticky_two_state_hmm_sequences(seed=0, n_train=5, n_held_out=0, length=10)
    hmm = DiscreteHMM.random_init(2, 2, seed=4)
    hmm.fit(train, max_iterations=5)
    nll = evaluate_predictive_nll(hmm, held_out_sequences=[np.array([0])])
    assert np.isnan(nll)  # no (t -> t+1) transitions to score at all
