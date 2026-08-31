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
    # Schema v2 records the TOTAL resource contract separately from the
    # persistent bytes selected by the representation planner.  Schema v1
    # plans treated the full VRAM budget as persistent residency and thereby
    # forgot the transient tensor/decode/activation reserve used by the
    # ordinary tier planner.
    vram_budget_bytes: int | None = None
    ram_budget_bytes: int | None = None
    vram_headroom_bytes: int = 0
    schema_version: int = 2

    def to_dict(self) -> dict:
        return {
            "choices": {key: dataclasses.asdict(value) for key, value in self.choices.items()},
            "vram_bytes": self.vram_bytes, "ram_bytes": self.ram_bytes,
            "storage_bytes": self.storage_bytes,
            "predicted_prepare_s": self.predicted_prepare_s,
            "feasible": self.feasible, "reason": self.reason,
            "vram_budget_bytes": self.vram_budget_bytes,
            "ram_budget_bytes": self.ram_budget_bytes,
            "vram_headroom_bytes": self.vram_headroom_bytes,
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
        if payload.get("schema_version") not in (1, 2):
            raise ValueError("unsupported representation-plan schema")
        choices = {key: RepresentationOption(**value)
                   for key, value in payload["choices"].items()}
        return cls(choices=choices, vram_bytes=int(payload["vram_bytes"]),
                   ram_bytes=int(payload["ram_bytes"]),
                   storage_bytes=int(payload["storage_bytes"]),
                   predicted_prepare_s=float(payload["predicted_prepare_s"]),
                   feasible=bool(payload.get("feasible", True)),
                   reason=str(payload.get("reason", "")),
                   vram_budget_bytes=(
                       int(payload["vram_budget_bytes"])
                       if payload.get("vram_budget_bytes") is not None else None),
                   ram_budget_bytes=(
                       int(payload["ram_budget_bytes"])
                       if payload.get("ram_budget_bytes") is not None else None),
                   vram_headroom_bytes=int(payload.get("vram_headroom_bytes", 0)),
                   schema_version=int(payload["schema_version"]))


def _prune_dominated(states: dict[tuple[int, int], tuple[float, int, object]]) -> dict:
    """Drop states no better in either memory dimension or objective."""
    # Representation alternatives commonly share the same persistent-store
    # bytes (for example, one exact compressed artifact staged from disk,
    # compressed RAM, decoded RAM, or VRAM).  In that case the original
    # all-pairs dominance scan is needlessly quadratic.  Sweep VRAM/RAM in
    # ascending order and query the best preparation cost at any smaller RAM
    # usage with a Fenwick prefix-min tree: O(n log n), exactly the same
    # dominance relation.  Keep the general four-dimensional fallback for
    # alternatives whose storage sizes really differ.
    storage_values = {value[1] for value in states.values()}
    if len(storage_values) <= 1 and states:
        ordered = sorted(states.items(),
                         key=lambda item: (item[0][0], item[0][1], item[1][0]))
        ram_values = sorted({state[1] for state in states})
        ram_index = {value: i + 1 for i, value in enumerate(ram_values)}
        tree = [math.inf] * (len(ram_values) + 1)

        def query(index: int) -> float:
            best = math.inf
            while index > 0:
                best = min(best, tree[index])
                index -= index & -index
            return best

        def update(index: int, value: float) -> None:
            while index < len(tree):
                tree[index] = min(tree[index], value)
                index += index & -index

        kept = {}
        for state, value in ordered:
            index = ram_index[state[1]]
            if query(index) <= value[0]:
                continue
            kept[state] = value
            update(index, value[0])
        return kept

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
                         quantum_bytes: int = 16 << 20,
                         vram_headroom_bytes: int = 0) -> RepresentationPlan:
    if vram_budget_bytes < 0 or ram_budget_bytes < 0:
        raise ValueError("representation budgets must be non-negative")
    if not 0 <= vram_headroom_bytes <= vram_budget_bytes:
        raise ValueError("VRAM headroom must fit inside the total VRAM budget")
    grouped: dict[str, list[RepresentationOption]] = {}
    for option in options:
        if not option.exact:
            continue
        grouped.setdefault(option.tensor_key, []).append(option)
    if not grouped:
        return RepresentationPlan(
            {}, 0, 0, 0, 0.0, False,
            "no exact representation options were supplied",
            vram_budget_bytes=vram_budget_bytes,
            ram_budget_bytes=ram_budget_bytes,
            vram_headroom_bytes=vram_headroom_bytes)

    q = max(1, quantum_bytes)
    vcap = (vram_budget_bytes - vram_headroom_bytes) // q
    rcap = ram_budget_bytes // q
    # Keep choices as linked back-pointers.  Copying a growing dictionary for
    # every DP transition made a real 441-tensor plan consume gigabytes and
    # minutes even though dominance pruning discarded almost all copies.
    # Back-pointers preserve the exact selected path with O(1) transition
    # memory; reconstruct the one winning dictionary only at the end.
    states: dict[tuple[int, int], tuple[float, int, object]] = {
        (0, 0): (0.0, 0, None)
    }
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
                             (selected, tensor_key, option))
                current = next_states.get((nv, nr))
                if current is None or candidate[:2] < current[:2]:
                    next_states[(nv, nr)] = candidate
        states = _prune_dominated(next_states)
        if not states:
            return RepresentationPlan(
                {}, 0, 0, 0, 0.0, False,
                "no representation for %s fits the budgets" % tensor_key,
                vram_budget_bytes=vram_budget_bytes,
                ram_budget_bytes=ram_budget_bytes,
                vram_headroom_bytes=vram_headroom_bytes)

    (used_v, used_r), (cost, storage, selected_node) = min(
        states.items(), key=lambda item: (item[1][0], item[1][1]))
    selected = {}
    while selected_node is not None:
        selected_node, tensor_key, option = selected_node
        selected[tensor_key] = option
    # The DP state is conservatively quantized for tractability, but the
    # runtime contract and paper metadata must report real selected bytes.
    actual_vram = sum(option.vram_bytes for option in selected.values())
    actual_ram = sum(option.ram_bytes for option in selected.values())
    return RepresentationPlan(
        selected, actual_vram, actual_ram, storage, cost,
        vram_budget_bytes=vram_budget_bytes,
        ram_budget_bytes=ram_budget_bytes,
        vram_headroom_bytes=vram_headroom_bytes)


def validate_artifacts(plan: RepresentationPlan, store_dir) -> list[str]:
    """Return missing artifact paths; never silently substitute a method."""
    root = pathlib.Path(store_dir)
    missing = []
    for option in plan.choices.values():
        if option.artifact and not (root / option.artifact).exists():
            missing.append(option.artifact)
    return sorted(set(missing))
