"""Residency planner (IMPLEMENTATION_PLAN.md Phase 3.1).

Start static: fill the VRAM budget with whole layers, highest priority first,
until the budget is exhausted, then stream the rest. The plan explicitly
calls for measuring this before adding anything adaptive -- an ATSInfer-style
tensor-granularity dynamic planner is a later, separately-justified upgrade,
not a default.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class LayerInfo:
    key: str
    nbytes: int
    priority: float = 1.0  # higher = more valuable to keep resident


def plan_static_residency(layers: list[LayerInfo], budget_bytes: int) -> tuple[list[str], list[str]]:
    """Greedy by priority density (priority / size), the standard fractional-
    knapsack heuristic, restricted to whole-layer (0/1) choices since layers
    can't be partially resident. Returns (resident_keys, streamed_keys)."""
    ordered = sorted(layers, key=lambda l: l.priority / max(l.nbytes, 1), reverse=True)
    resident, streamed = [], []
    used = 0
    for layer in ordered:
        if used + layer.nbytes <= budget_bytes:
            resident.append(layer.key)
            used += layer.nbytes
        else:
            streamed.append(layer.key)
    return resident, streamed


def uniform_priority(keys: list[str]) -> dict[str, float]:
    return {k: 1.0 for k in keys}
