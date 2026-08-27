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

import math

import numpy as np
import pytest

from afterimage.runtime.speculation_oracle import (
    DiscreteHMM,
    MultinomialLogisticRegression,
    _logsumexp,
    approximate_js_divergence_from_topk,
    bootstrap_nll_difference_ci,
    compute_equal_budget_metrics,
    compute_oracle_coverage_stats,
    compute_sustained_rescue_depth,
    constant_frequency_baseline_nll,
    disagreement_bucket,
    entropy_of_logits,
    evaluate_predictive_nll,
    logistic_regression_nll_from_model,
    grouped_k_fold,
    js_divergence,
    logistic_regression_baseline_nll,
    margin_of_logits,
    memoryless_baseline_nll,
    minimum_positions_for_hmm,
    rank_of_token,
    select_n_states_by_held_out_nll,
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


# ---------------------------------------------------- entropy/margin/divergence

def test_entropy_of_logits_is_lower_for_a_peaked_distribution():
    uniform = np.zeros(10)
    peaked = np.array([10.0] + [0.0] * 9)
    assert entropy_of_logits(peaked) < entropy_of_logits(uniform)


def test_margin_of_logits_is_the_top1_minus_top2_gap():
    assert margin_of_logits(np.array([5.0, 1.0, 0.0])) == pytest.approx(4.0)
    assert margin_of_logits(np.array([3.0, 3.0, 0.0])) == pytest.approx(0.0)


def test_js_divergence_is_zero_for_identical_distributions():
    logits = np.array([2.0, 1.0, 0.0, -1.0])
    assert js_divergence(logits, logits) == pytest.approx(0.0, abs=1e-9)


def test_js_divergence_is_positive_and_bounded_for_different_distributions():
    peaked = np.array([10.0, 0.0, 0.0, 0.0])
    uniform = np.zeros(4)
    d = js_divergence(peaked, uniform)
    assert 0.0 < d <= math.log(2)  # JS divergence in nats is bounded by ln(2)


# ------------------------------------------------ approximate_js_divergence_from_topk

def test_approximate_js_divergence_is_zero_for_identical_topk():
    d = approximate_js_divergence_from_topk([1, 2, 3], [0.5, 0.3, 0.2],
                                            [1, 2, 3], [0.5, 0.3, 0.2])
    assert d == pytest.approx(0.0, abs=1e-9)


def test_approximate_js_divergence_is_ln2_for_disjoint_supports():
    """Two sources whose stored top-k share no token at all look
    maximally divergent under this sparse approximation -- the honest
    consequence of not knowing whether they actually agreed on
    unrecorded tail mass."""
    d = approximate_js_divergence_from_topk([1, 2], [0.6, 0.4], [9, 8], [0.6, 0.4])
    assert d == pytest.approx(math.log(2), abs=1e-6)


def test_approximate_js_divergence_is_between_bounds_for_partial_overlap():
    d = approximate_js_divergence_from_topk([1, 2, 3], [0.7, 0.2, 0.1],
                                            [1, 2, 3], [0.1, 0.2, 0.7])
    assert 0.0 < d < math.log(2)


def test_approximate_js_divergence_handles_zero_recorded_mass():
    d = approximate_js_divergence_from_topk([], [], [], [])
    assert d == 0.0


# -------------------------------------------------------- compute_equal_budget_metrics

def _budget_row(target, primary_topk, scout_topk):
    return {"target_token": target, "primary_topk": primary_topk, "scout_topk": scout_topk}


def test_equal_budget_baseline_matches_primary_only_at_full_budget():
    """s=0 (no scout slots) must reproduce plain Primary-top-budget
    coverage exactly -- it is the control every other point compares
    against."""
    rows = [
        _budget_row(5, [5, 1, 2, 3], [9, 9, 9, 9]),
        _budget_row(5, [1, 2, 3, 4], [5, 9, 9, 9]),
    ]
    m = compute_equal_budget_metrics(rows, total_budget=4, scout_slots=[0])
    assert m["points"][0]["coverage"] == pytest.approx(m["baseline_coverage"])
    assert m["points"][0]["equal_budget_union_gain"] == pytest.approx(0.0)


def test_equal_budget_gain_reflects_traded_slots_not_free_extra_slots():
    """The corrected H21 question: trading Primary slots for Scout slots
    at a FIXED total budget. A row where the target is outside Primary's
    kept slots but inside Scout's must show a real gain; a row where
    Scout has nothing new must not."""
    rows = [
        _budget_row(5, [1, 2, 3, 4], [5, 9, 9, 9]),  # target drops out of primary's
                                                       # kept 2 slots at s=2, but
                                                       # scout's slot 0 has it
        _budget_row(6, [6, 1, 2, 3], [9, 9, 9, 9]),  # primary already covers with
                                                       # its kept slots; scout adds nothing
    ]
    m = compute_equal_budget_metrics(rows, total_budget=4, scout_slots=[2])
    point = m["points"][2]
    assert point["primary_slots"] == 2
    # row 1: primary kept slots [1,2] -> misses target 5; scout slots [5,9] -> covers.
    # row 2: primary kept slots [6,1] -> covers target 6 directly.
    assert point["coverage"] == pytest.approx(1.0)
    assert point["unique_scout_hit_rate"] == pytest.approx(0.5)


def test_equal_budget_metrics_rejects_stored_lists_shorter_than_budget():
    rows = [_budget_row(5, [1, 2], [1, 2])]
    with pytest.raises(ValueError, match="fewer than total_budget"):
        compute_equal_budget_metrics(rows, total_budget=4, scout_slots=[2])


def test_equal_budget_metrics_empty_rows_does_not_crash():
    m = compute_equal_budget_metrics([], total_budget=8, scout_slots=[4])
    assert m["rows"] == 0


# ------------------------------------------------------ compute_sustained_rescue_depth

def _depth_trace(pairs):
    return {"rows": [{"target_rank_under_primary": p, "target_rank_under_scout": s}
                     for p, s in pairs]}


def test_sustained_rescue_depth_counts_each_qualifying_position_independently():
    """Two adjacent positions that both independently miss-then-rescue
    produce two opportunities with their own (possibly different) depths
    -- see the function's own docstring for why this is not
    double-counting."""
    trace = _depth_trace([(50, 1), (50, 1), (50, 50), (1, 1)])
    d = compute_sustained_rescue_depth([trace], k=8)
    assert d["rescue_opportunities"] == 2
    assert d["mean_depth"] == pytest.approx(1.5)
    assert d["depth_probabilities"][1] == pytest.approx(1.0)
    assert d["depth_probabilities"][2] == pytest.approx(0.5)
    assert d["depth_probabilities"][4] == pytest.approx(0.0)


def test_sustained_rescue_depth_is_none_with_no_opportunities():
    trace = _depth_trace([(1, 1), (1, 1)])  # primary always covers
    d = compute_sustained_rescue_depth([trace], k=8)
    assert d["rescue_opportunities"] == 0
    assert d["mean_depth"] is None


def test_sustained_rescue_depth_requires_scout_to_actually_cover_to_count():
    trace = _depth_trace([(50, 50)])  # both miss -- not a rescue at all
    d = compute_sustained_rescue_depth([trace], k=8)
    assert d["rescue_opportunities"] == 0


# ------------------------------------------------------------- baseline ladder (B0-B4)

def test_constant_frequency_baseline_prefers_the_common_symbol():
    train = [np.array([0, 0, 0, 0, 1])]
    held = [np.array([0, 0])]
    nll = constant_frequency_baseline_nll(train, held, n_symbols=2)
    # symbol 0 is 4/5 of training mass, so predicting it should cost
    # noticeably less than -log(0.5) (the uninformed-coin-flip cost).
    assert nll < -math.log(0.5)


def test_multinomial_logistic_regression_separates_a_linearly_separable_case():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] > 0).astype(int)
    model = MultinomialLogisticRegression.fit(X, y, n_classes=2, seed=0)
    probs = model.predict_proba(X)
    predicted = probs.argmax(axis=1)
    accuracy = (predicted == y).mean()
    assert accuracy > 0.95


def test_logistic_regression_baseline_beats_chance_on_separable_data():
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(200, 3))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_held = rng.normal(size=(100, 3))
    y_held = (X_held[:, 0] + X_held[:, 1] > 0).astype(int)
    nll = logistic_regression_baseline_nll(X_train, y_train, X_held, y_held, n_classes=2)
    assert nll < -math.log(0.5)  # must beat an uninformed coin flip


def test_logistic_regression_baseline_nll_handles_empty_split():
    nll = logistic_regression_baseline_nll(
        np.zeros((0, 2)), np.zeros(0, dtype=int), np.zeros((0, 2)), np.zeros(0, dtype=int), 2)
    assert np.isnan(nll)


def test_logistic_regression_is_invariant_to_feature_scale():
    """H22's real features span very different magnitudes (entropy in
    nats, margin in raw logit units, divergence bounded by ln 2, 0/1
    one-hots). Without internal standardization, fixed-step gradient
    descent is dominated by the largest column and underfits -- which
    would make B2/B3/B4 look weak for optimizer reasons and bias G4b
    toward a false negative. Rescaling a column must not change the fit.
    """
    rng = np.random.default_rng(0)
    n = 600
    small = rng.uniform(0, 3, n)
    large = rng.uniform(0, 30, n)
    y = (large > 15).astype(int)
    split = n // 2

    raw = np.column_stack([small, large])
    # Same information, one column scaled by 100x.
    rescaled = np.column_stack([small, large * 100.0])

    nll_raw = logistic_regression_baseline_nll(
        raw[:split], y[:split], raw[split:], y[split:], 2, seed=0)
    nll_rescaled = logistic_regression_baseline_nll(
        rescaled[:split], y[:split], rescaled[split:], y[split:], 2, seed=0)
    assert nll_raw == pytest.approx(nll_rescaled, abs=1e-9)
    # And the fit must actually be good, not merely consistent.
    assert nll_raw < 0.2


def test_logistic_regression_handles_a_constant_feature_column():
    """A one-hot history column for a bucket that never occurs in a given
    fold is all zeros -- standardizing it must not divide by zero."""
    rng = np.random.default_rng(1)
    n = 200
    X = np.column_stack([rng.normal(size=n), np.zeros(n)])
    y = (X[:, 0] > 0).astype(int)
    nll = logistic_regression_baseline_nll(
        X[:100], y[:100], X[100:], y[100:], 2, seed=0)
    assert np.isfinite(nll)


def test_logistic_regression_nll_from_model_matches_the_fit_and_score_helper():
    """The split-out scorer must agree exactly with the combined helper,
    since the H22 pipeline fits once per fold and then scores each
    trajectory through the former."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    combined = logistic_regression_baseline_nll(
        X[:100], y[:100], X[100:], y[100:], 2, seed=3)
    model = MultinomialLogisticRegression.fit(X[:100], y[:100], 2, seed=3)
    split_out = logistic_regression_nll_from_model(model, X[100:], y[100:])
    assert combined == pytest.approx(split_out)


# --------------------------------------------------------- select_n_states_by_held_out_nll

def test_select_n_states_prefers_the_true_generator_state_count():
    """On data actually generated by a 2-state HMM, validation NLL should
    not favor an unnecessarily large state count -- 2 states should win
    over needlessly fitting 6, PROVIDED the validation split itself is
    large enough. This needs real data to demonstrate, not an assumption:
    with only 10 held-out sequences the validation NLL gap between 2, 3,
    and 6 states is smaller than sampling noise (differences of ~0.005
    nats on ~0.67 nats, empirically measured during development) and
    selection becomes unreliable -- see the companion test below, which
    documents that failure mode deliberately rather than treating it as
    a flake. 60 held-out sequences is what actually resolves it here."""
    train, val = _sticky_two_state_hmm_sequences(seed=0, n_train=40, n_held_out=60, length=25)
    result = select_n_states_by_held_out_nll(
        train, val, n_symbols=2, candidate_n_states=(2, 3, 6), restarts=5,
        max_iterations=50, seed=0)
    assert result["selected_n_states"] == 2
    assert len(result["candidates"]) == 3


def test_select_n_states_is_unreliable_with_too_little_validation_data():
    """Documents a real, measured failure mode rather than hiding it: with
    only 10 held-out sequences, state selection on this same 2-state
    generator picks 6 states, not the true 2 -- the validation NLLs are
    within noise of each other. This is exactly why minimum_positions_
    for_hmm exists, and why H22's own script must enforce a minimum
    validation size before trusting a selected state count, not just a
    minimum training size."""
    train, val = _sticky_two_state_hmm_sequences(seed=0, n_train=20, n_held_out=10, length=25)
    result = select_n_states_by_held_out_nll(
        train, val, n_symbols=2, candidate_n_states=(2, 3, 6), restarts=2,
        max_iterations=30, seed=0)
    nlls = {c["n_states"]: c["validation_nll"] for c in result["candidates"]}
    spread = max(nlls.values()) - min(nlls.values())
    assert spread < 0.01, (
        "if this spread grows large, the small-validation-set instability "
        "this test documents may no longer hold and the test itself should "
        "be revisited, not just its assertion loosened")


def test_select_n_states_reports_every_candidate_even_when_not_selected():
    train, val = _sticky_two_state_hmm_sequences(seed=0, n_train=10, n_held_out=5, length=15)
    result = select_n_states_by_held_out_nll(
        train, val, n_symbols=2, candidate_n_states=(2, 4), restarts=1,
        max_iterations=10, seed=0)
    reported = {c["n_states"] for c in result["candidates"]}
    assert reported == {2, 4}


# ---------------------------------------------------------------------- grouped_k_fold

def test_grouped_k_fold_partitions_every_group_exactly_once():
    splits = grouped_k_fold(n_groups=10, k=5, seed=0)
    assert len(splits) == 5
    all_test_indices = sorted(idx for _, test in splits for idx in test)
    assert all_test_indices == list(range(10))


def test_grouped_k_fold_train_and_test_never_overlap():
    for train_idx, test_idx in grouped_k_fold(n_groups=12, k=4, seed=1):
        assert set(train_idx).isdisjoint(test_idx)


def test_grouped_k_fold_rejects_more_folds_than_groups():
    with pytest.raises(ValueError):
        grouped_k_fold(n_groups=3, k=5, seed=0)


# --------------------------------------------------------------- bootstrap_nll_difference_ci

def test_bootstrap_ci_excludes_zero_for_a_consistent_positive_effect():
    ci = bootstrap_nll_difference_ci([0.5, 0.6, 0.55, 0.52, 0.58, 0.51], seed=0)
    assert ci["excludes_zero"] is True
    assert ci["ci_low"] > 0.0


def test_bootstrap_ci_does_not_exclude_zero_for_a_noisy_null_effect():
    ci = bootstrap_nll_difference_ci([0.1, -0.1, 0.05, -0.05, 0.02, -0.02], seed=0)
    assert ci["excludes_zero"] is False


def test_bootstrap_ci_handles_empty_input():
    ci = bootstrap_nll_difference_ci([], seed=0)
    assert ci["mean"] is None
    assert ci["excludes_zero"] is None


# ----------------------------------------------------------------- minimum_positions_for_hmm

def test_minimum_positions_scales_with_state_and_symbol_count():
    small = minimum_positions_for_hmm(2, 2)
    large = minimum_positions_for_hmm(6, 9)
    assert large > small


def test_minimum_positions_matches_the_documented_free_parameter_formula():
    # 4 states, 5 symbols: 4*(5-1) emission + 4*(4-1) transition + (4-1) initial = 16+12+3=31
    assert minimum_positions_for_hmm(4, 5, observations_per_free_parameter=20) == 31 * 20
