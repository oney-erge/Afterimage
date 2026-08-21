"""Deterministic multi-fidelity screening for expensive runtime profiles.

The tuner implements successive halving: evaluate every configuration on a
cheap fidelity, promote only the strongest fraction, and spend full benchmark
time on the survivors.  It deliberately avoids fitting a fragile surrogate to
the very small datasets typical of one-GPU systems research.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Iterable


@dataclasses.dataclass(frozen=True)
class FidelityObservation:
    config_id: str
    fidelity: int
    score: float
    payload: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class MultiFidelityResult:
    best_config_id: str
    observations: tuple[FidelityObservation, ...]
    survivors_by_fidelity: dict[int, tuple[str, ...]]


def successive_halving(config_ids: Iterable[str], fidelities: Iterable[int],
                       evaluator: Callable[[str, int], float | dict], *,
                       promotion_fraction: float = 0.5,
                       higher_is_better: bool = True) -> MultiFidelityResult:
    """Screen configs at increasing positive fidelities with no hidden state.

    ``evaluator`` may return a score or ``{"score": ..., ...}``. Ties are
    broken by config id so a manifest plus evaluator is reproducible.
    """
    active = sorted(set(config_ids))
    levels = sorted(set(int(value) for value in fidelities))
    if not active:
        raise ValueError("at least one configuration is required")
    if not levels or levels[0] < 1:
        raise ValueError("fidelities must be positive")
    if not (0.0 < promotion_fraction <= 1.0):
        raise ValueError("promotion_fraction must be in (0, 1]")

    observations: list[FidelityObservation] = []
    survivors: dict[int, tuple[str, ...]] = {}
    last_scores: dict[str, float] = {}
    for level_index, fidelity in enumerate(levels):
        last_scores = {}
        for config_id in active:
            raw = evaluator(config_id, fidelity)
            if isinstance(raw, dict):
                if "score" not in raw:
                    raise ValueError("evaluator result dictionary requires score")
                score = float(raw["score"])
                payload = {key: value for key, value in raw.items() if key != "score"}
            else:
                score, payload = float(raw), {}
            if not math.isfinite(score):
                raise ValueError("non-finite score for %s" % config_id)
            last_scores[config_id] = score
            observations.append(FidelityObservation(config_id, fidelity, score, payload))
        ranked = sorted(active, key=lambda key: (
            -last_scores[key] if higher_is_better else last_scores[key], key))
        keep = (len(ranked) if level_index == len(levels) - 1 else
                max(1, math.ceil(len(ranked) * promotion_fraction)))
        active = ranked[:keep]
        survivors[fidelity] = tuple(active)

    best = min(active, key=lambda key: (
        -last_scores[key] if higher_is_better else last_scores[key], key))
    return MultiFidelityResult(best, tuple(observations), survivors)
