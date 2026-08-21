import json

import pytest
import torch

from afterimage.runtime.spec_policy import (
    AdaEDLPolicy, FixedPolicy, GammaTunePolicy, HazardCostPolicy, SweepRecord,
    NeuralUtilityPolicy, ThresholdPolicy, build_policy,
)


def test_fixed_policy_never_changes():
    p = FixedPolicy(k=8)
    assert p.choose_k() == 8
    p.update(SweepRecord(k_used=8, n_accepted=0, sweep_seconds=1.0))
    assert p.choose_k() == 8


def test_gamma_tune_expands_k_on_high_acceptance():
    p = GammaTunePolicy(k_init=8, k_max=16, high=0.7, low=0.4)
    for _ in range(5):
        p.update(SweepRecord(k_used=p.choose_k(), n_accepted=p.choose_k(), sweep_seconds=1.0))
    assert p.choose_k() > 8


def test_gamma_tune_contracts_k_on_low_acceptance():
    p = GammaTunePolicy(k_init=8, k_min=1, high=0.7, low=0.4)
    for _ in range(5):
        p.update(SweepRecord(k_used=p.choose_k(), n_accepted=0, sweep_seconds=1.0))
    assert p.choose_k() < 8


def test_gamma_tune_respects_bounds():
    p = GammaTunePolicy(k_init=8, k_min=2, k_max=10, high=0.5, low=0.9)
    for _ in range(20):
        p.update(SweepRecord(k_used=p.choose_k(), n_accepted=p.choose_k(), sweep_seconds=1.0))
    assert p.choose_k() <= 10
    p2 = GammaTunePolicy(k_init=8, k_min=2, k_max=10, high=0.9, low=0.5)
    for _ in range(20):
        p2.update(SweepRecord(k_used=p2.choose_k(), n_accepted=0, sweep_seconds=1.0))
    assert p2.choose_k() >= 2


class _FakeProb:
    """Stand-in for a torch tensor: only .max() is used by trim_by_confidence."""
    def __init__(self, v):
        self._v = v

    def max(self):
        return self._v


def test_threshold_policy_trims_at_first_low_confidence_position():
    p = ThresholdPolicy(k_min=1, k_max=16, confidence_threshold=0.5)
    probs = [_FakeProb(0.9), _FakeProb(0.8), _FakeProb(0.3), _FakeProb(0.9)]
    assert p.trim_by_confidence(probs) == 2


def test_threshold_policy_never_trims_below_k_min():
    p = ThresholdPolicy(k_min=2, k_max=16, confidence_threshold=0.5)
    probs = [_FakeProb(0.1), _FakeProb(0.1), _FakeProb(0.1)]
    assert p.trim_by_confidence(probs) == 2


def test_threshold_policy_keeps_everything_above_threshold():
    p = ThresholdPolicy(k_min=1, k_max=16, confidence_threshold=0.5)
    probs = [_FakeProb(0.9), _FakeProb(0.8), _FakeProb(0.99)]
    assert p.trim_by_confidence(probs) == 3


def test_threshold_policy_raises_bar_when_acceptance_low_and_no_speed_gain():
    p = ThresholdPolicy(confidence_threshold=0.5)
    p.update(SweepRecord(k_used=8, n_accepted=8, sweep_seconds=1.0))  # sets best_tps high
    before = p.confidence_threshold
    p.update(SweepRecord(k_used=8, n_accepted=1, sweep_seconds=5.0))  # slow AND low acceptance
    assert p.confidence_threshold > before


def test_build_policy_dispatch():
    assert isinstance(build_policy("fixed", 8), FixedPolicy)
    assert isinstance(build_policy("gamma", 8), GammaTunePolicy)
    assert isinstance(build_policy("threshold", 8), ThresholdPolicy)
    assert isinstance(build_policy("adaedl", 8), AdaEDLPolicy)
    assert isinstance(build_policy("hazard_cost", 8), HazardCostPolicy)
    assert isinstance(build_policy("neural_utility", 8), NeuralUtilityPolicy)
    with pytest.raises(ValueError):
        build_policy("nonsense", 8)


def test_save_load_round_trip_preserves_state(tmp_path):
    p = GammaTunePolicy(k_init=8)
    for _ in range(4):
        p.update(SweepRecord(k_used=p.choose_k(), n_accepted=p.choose_k(), sweep_seconds=1.0))
    path = tmp_path / "policy.json"
    p.save(path)

    p2 = GammaTunePolicy(k_init=8)
    p2.load(path)
    assert p2.k == p.k
    assert p2.mean_acceptance == p.mean_acceptance


def test_load_missing_file_is_a_noop(tmp_path):
    p = FixedPolicy(k=5)
    p.load(tmp_path / "does_not_exist.json")
    assert p.choose_k() == 5


def test_load_wrong_policy_name_raises(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"policy": "gamma", "state": {}}))
    p = FixedPolicy(k=5)
    with pytest.raises(ValueError, match="gamma"):
        p.load(path)


def test_sweep_record_derived_fields():
    r = SweepRecord(k_used=8, n_accepted=5, sweep_seconds=2.0)
    assert r.acceptance == pytest.approx(5 / 8)
    assert r.tokens_per_second == pytest.approx(6 / 2.0)


def test_adaedl_uses_published_entropy_lower_bound_and_updates_threshold():
    policy = AdaEDLPolicy(k_min=1, threshold=0.7, gamma=0.2)
    concentrated = torch.tensor([0.99, 0.01])
    diffuse = torch.tensor([0.5, 0.5])
    assert not policy.should_stop([concentrated, concentrated])
    assert policy.should_stop([concentrated, diffuse])
    before = policy.threshold
    policy.update(SweepRecord(k_used=4, n_accepted=0, sweep_seconds=1.0))
    assert policy.threshold > before


def test_hazard_policy_treats_unseen_tail_as_censored():
    policy = HazardCostPolicy(k_max=4, n_bins=2)
    policy.update(SweepRecord(k_used=4, n_accepted=1, sweep_seconds=1.0,
                              draft_confidences=(0.9, 0.9, 0.9, 0.9),
                              draft_seconds=0.4, target_seconds=2.0))
    assert policy.accept[0][1] == 1
    assert policy.reject[1][1] == 1
    assert sum(policy.reject[2]) == 0
    assert sum(policy.reject[3]) == 0
    assert policy.target_token_s == pytest.approx(1.0)


def test_neural_utility_learns_censored_acceptance_without_tail_labels():
    policy = NeuralUtilityPolicy(k_max=4, minimum_observations=1,
                                 learning_rate=0.08, training_steps=8)
    before_high = policy._predict(policy._features(0.95, 0.1, 0))
    before_low = policy._predict(policy._features(0.10, 2.0, 1))
    for _ in range(40):
        policy.update(SweepRecord(
            k_used=4, n_accepted=1, sweep_seconds=2.0,
            draft_confidences=(0.95, 0.10, 0.99, 0.99),
            draft_entropies=(0.1, 2.0, 0.01, 0.01),
            draft_seconds=0.8, target_seconds=1.2))
    after_high = policy._predict(policy._features(0.95, 0.1, 0))
    after_low = policy._predict(policy._features(0.10, 2.0, 1))
    assert after_high > before_high
    assert after_low < before_low
    # Only the accepted first position and first rejection are observable;
    # the confident-looking tail must not be treated as either label.
    assert policy.n_observations == 80


def test_neural_utility_state_round_trip_and_cost_aware_stop(tmp_path):
    policy = NeuralUtilityPolicy(k_max=4, minimum_observations=1)
    policy.n_observations = 10
    policy.draft_token_s = 2.0
    policy.target_sweep_s = 1.0
    diffuse = torch.tensor([0.5, 0.5])
    assert policy.should_stop([diffuse, diffuse])
    assert policy.decision_stops == 1
    assert policy.last_stop_position == 1
    path = tmp_path / "neural.json"
    policy.save(path)
    loaded = NeuralUtilityPolicy(k_max=4, minimum_observations=1)
    loaded.load(path)
    assert loaded.n_observations == policy.n_observations
    assert loaded.target_sweep_s == policy.target_sweep_s
    assert loaded.decision_stops == 0
