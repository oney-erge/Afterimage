import pytest

from afterimage.experiments import HYPOTHESES
from afterimage.protocols import (
    HYPOTHESIS_PROTOCOLS, assess_paired_effect, protocol_for,
    protocol_payload, validate_protocol_registry,
)


def test_every_hypothesis_has_one_valid_protocol():
    validate_protocol_registry(HYPOTHESES)
    assert set(HYPOTHESIS_PROTOCOLS) == set(HYPOTHESES)
    assert protocol_for("h12-bayesian-prefetch").family == "online I/O scheduling"
    assert protocol_for("h13-qubo-residency").family == "offline tensor placement"
    assert protocol_for("h14-coalesced-storage").family == (
        "physical storage request geometry")
    assert protocol_for("h15-extent-qubo-residency").family == (
        "offline tensor placement")
    payload = protocol_payload()
    assert payload["evidence_levels"]["L1"].startswith("mechanism smoke")


def test_l1_effect_can_never_be_reported_as_performance_support():
    result = assess_paired_effect(
        [10, 10, 10], [5, 5, 5], minimum_effect=0.05, level="L1", seed=1)
    assert result["decision"] == "mechanism_only"
    assert not result["confirmation_eligible"]


def test_l2_can_stop_for_futility_or_advance_without_confirming():
    futile = assess_paired_effect(
        [10] * 8, [10.1] * 8, minimum_effect=0.05, level="L2", seed=2)
    assert futile["decision"] == "stop_futility"

    promising = assess_paired_effect(
        [10] * 8, [9] * 8, minimum_effect=0.05, level="L2", seed=2)
    assert promising["decision"] == "advance_to_confirmation"
    assert not promising["confirmation_eligible"]


def test_l3_requires_positive_lower_bound_and_practical_effect():
    result = assess_paired_effect(
        [10] * 8, [9] * 8, minimum_effect=0.05, level="L3", seed=3)
    assert result["decision"] == "supported"
    assert result["confirmation_eligible"]


def test_effect_rejects_unpaired_or_nonpositive_inputs():
    with pytest.raises(ValueError, match="equal"):
        assess_paired_effect([1, 2], [1], minimum_effect=0.1, level="L2")
    with pytest.raises(ValueError, match="positive"):
        assess_paired_effect([1], [0], minimum_effect=0.1, level="L2")
