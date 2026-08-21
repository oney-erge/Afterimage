"""Exact physical-representation planning.

The logical tensor is immutable.  A representation is merely a physical way
to store or stage its exact bits.  The planner chooses one option per tensor
under VRAM/RAM budgets and minimizes measured critical-path preparation time.
"""
from __future__ import annotations

import dataclasses
import json
import math
import pathlib


@dataclasses.dataclass(frozen=True)
class RepresentationOption:
    tensor_key: str
    name: str
    vram_bytes: int = 0
    ram_bytes: int = 0
    storage_bytes: int = 0
    prepare_s: float = 0.0
    exact: bool = True
    artifact: str | None = None


@dataclasses.dataclass(frozen=True)
class RepresentationPlan:
    choices: dict[str, RepresentationOption]
    vram_bytes: int
    ram_bytes: int
    storage_bytes: int
    predicted_prepare_s: float
    feasible: bool = True
    reason: str = ""
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "choices": {key: dataclasses.asdict(value) for key, value in self.choices.items()},
            "vram_bytes": self.vram_bytes, "ram_bytes": self.ram_bytes,
            "storage_bytes": self.storage_bytes,
            "predicted_prepare_s": self.predicted_prepare_s,
            "feasible": self.feasible, "reason": self.reason,
            "schema_version": self.schema_version,
        }

    def save(self, path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> "RepresentationPlan":
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported representation-plan schema")
        choices = {key: RepresentationOption(**value)
                   for key, value in payload["choices"].items()}
        return cls(choices=choices, vram_bytes=int(payload["vram_bytes"]),
                   ram_bytes=int(payload["ram_bytes"]),
                   storage_bytes=int(payload["storage_bytes"]),
                   predicted_prepare_s=float(payload["predicted_prepare_s"]),
                   feasible=bool(payload.get("feasible", True)),
                   reason=str(payload.get("reason", "")),
                   schema_version=int(payload["schema_version"]))


def _prune_dominated(states: dict[tuple[int, int], tuple[float, int, dict]]) -> dict:
    """Drop states no better in either memory dimension or objective."""
    kept = {}
    ordered = sorted(states.items(), key=lambda item: (item[0][0], item[0][1], item[1][0]))
    for state, value in ordered:
        cost, storage, _choices = value
        dominated = any(vr <= state[0] and ram <= state[1]
                        and other[0] <= cost and other[1] <= storage
                        for (vr, ram), other in kept.items())
        if not dominated:
            kept[state] = value
    return kept


def plan_representations(options: list[RepresentationOption], *, vram_budget_bytes: int,
                         ram_budget_bytes: int, storage_budget_bytes: int | None = None,
                         quantum_bytes: int = 16 << 20) -> RepresentationPlan:
    grouped: dict[str, list[RepresentationOption]] = {}
    for option in options:
        if not option.exact:
            continue
        grouped.setdefault(option.tensor_key, []).append(option)
    if not grouped:
        return RepresentationPlan({}, 0, 0, 0, 0.0, False,
                                  "no exact representation options were supplied")

    q = max(1, quantum_bytes)
    vcap = vram_budget_bytes // q
    rcap = ram_budget_bytes // q
    states: dict[tuple[int, int], tuple[float, int, dict]] = {(0, 0): (0.0, 0, {})}
    for tensor_key, choices in grouped.items():
        next_states = {}
        for (used_v, used_r), (cost, storage, selected) in states.items():
            for option in choices:
                ov = math.ceil(option.vram_bytes / q)
                oh = math.ceil(option.ram_bytes / q)
                nv, nr = used_v + ov, used_r + oh
                ns = storage + option.storage_bytes
                if nv > vcap or nr > rcap:
                    continue
                if storage_budget_bytes is not None and ns > storage_budget_bytes:
                    continue
                candidate = (cost + option.prepare_s, ns,
                             {**selected, tensor_key: option})
                current = next_states.get((nv, nr))
                if current is None or candidate[:2] < current[:2]:
                    next_states[(nv, nr)] = candidate
        states = _prune_dominated(next_states)
        if not states:
            return RepresentationPlan({}, 0, 0, 0, 0.0, False,
                                      "no representation for %s fits the budgets" % tensor_key)

    (used_v, used_r), (cost, storage, selected) = min(
        states.items(), key=lambda item: (item[1][0], item[1][1]))
    return RepresentationPlan(selected, used_v * q, used_r * q, storage, cost)


def validate_artifacts(plan: RepresentationPlan, store_dir) -> list[str]:
    """Return missing artifact paths; never silently substitute a method."""
    root = pathlib.Path(store_dir)
    missing = []
    for option in plan.choices.values():
        if option.artifact and not (root / option.artifact).exists():
            missing.append(option.artifact)
    return sorted(set(missing))
