import pytest

from afterimage.bench.multifidelity import successive_halving


def test_successive_halving_promotes_only_strong_configs():
    quality = {"a": 1.0, "b": 3.0, "c": 2.0, "d": 0.0}
    result = successive_halving(quality, [1, 4, 16],
                                lambda config, fidelity: quality[config])
    assert result.survivors_by_fidelity[1] == ("b", "c")
    assert result.survivors_by_fidelity[4] == ("b",)
    assert result.best_config_id == "b"
    assert len(result.observations) == 7


def test_successive_halving_supports_lower_is_better_and_payload():
    result = successive_halving(["slow", "fast"], [2],
                                lambda config, fidelity: {
                                    "score": 1 if config == "fast" else 2,
                                    "tokens": fidelity,
                                }, higher_is_better=False)
    assert result.best_config_id == "fast"
    assert result.observations[0].payload["tokens"] == 2


def test_successive_halving_rejects_invalid_fidelity():
    with pytest.raises(ValueError, match="positive"):
        successive_halving(["a"], [0], lambda *_: 1)
