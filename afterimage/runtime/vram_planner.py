"""User-dictated VRAM budgeting.

The head-to-head against AirLLM exposed the gap this module closes: our
engine used 5.10 GB against AirLLM's 1.57 GB on the same 14B model, purely
because it kept embed_tokens and lm_head permanently resident (3.1 GB of
that 5.1 GB) while AirLLM streams them like any other layer. We were faster
(24.93 vs 32.23 s/token) but far heavier, and a 4 GB cap was impossible for
us and trivial for AirLLM.

So residency stops being hardcoded. You state a budget; the planner decides
what fits.

Which tensors to keep resident, when they cannot all fit
--------------------------------------------------------
Every weight in a dense decoder is used exactly once per token, so
"frequency of use" cannot rank them -- it is identical for all. The thing
that does differ is how much streaming each one COSTS relative to the VRAM
it occupies:

    value density = compressed_bytes / uncompressed_bytes

A tensor that compresses poorly costs nearly its full size in bus traffic
every token, so pinning it in VRAM avoids the most I/O per byte of VRAM
spent. A highly compressible tensor is cheap to re-stream, so it is the
better candidate to evict. Ranking by this ratio and filling the budget
greedily is the fractional-knapsack solution to "minimize bytes streamed
per token subject to a VRAM ceiling" -- exact for the fractional case and a
good approximation here, where items are whole tensors.

This inverts the intuitive answer (keep the biggest / most important
tensors), and it only becomes visible once compression is on the streaming
path, which is why no existing offloading engine does it.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TensorInfo:
    key: str
    orig_bytes: int          # size when materialized in VRAM (bf16)
    comp_bytes: int          # size on disk / bytes streamed if not resident
    is_layer_weight: bool    # part of a decoder layer (vs embedding/head/norm)

    @property
    def value_density(self) -> float:
        """Bus traffic avoided per byte of VRAM spent keeping this resident."""
        return self.comp_bytes / max(self.orig_bytes, 1)


@dataclasses.dataclass
class VramPlan:
    budget_bytes: int
    headroom_bytes: int
    resident_keys: list
    streamed_keys: list
    resident_bytes: int
    streamed_bytes_per_token: int
    feasible: bool
    reason: str

    @property
    def budget_gb(self) -> float:
        return self.budget_bytes / 1e9

    @property
    def resident_gb(self) -> float:
        return self.resident_bytes / 1e9

    @property
    def streamed_gb_per_token(self) -> float:
        return self.streamed_bytes_per_token / 1e9

    def describe(self) -> str:
        lines = [
            "VRAM plan for a %.2f GB budget:" % self.budget_gb,
            "  resident            : %.2f GB across %d tensors"
            % (self.resident_gb, len(self.resident_keys)),
            "  streamed per token  : %.2f GB across %d tensors"
            % (self.streamed_gb_per_token, len(self.streamed_keys)),
            "  working headroom    : %.2f GB" % (self.headroom_bytes / 1e9),
            "  feasible            : %s" % self.feasible,
        ]
        if not self.feasible:
            lines.append("  reason              : " + self.reason)
        return "\n".join(lines)


DEFAULT_SCRATCH_BYTES = 512 * 1024 * 1024


def plan_residency(tensors, budget_gb: float,
                   headroom_bytes: int = None,
                   scratch_bytes: int = DEFAULT_SCRATCH_BYTES,
                   pin_layer_weights: bool = False) -> VramPlan:
    """Decide what stays in VRAM for a given budget.

    headroom_bytes: VRAM reserved for transient work. Defaults to
    (largest tensor + scratch), because streaming a tensor requires its
    materialized output live at once plus a bounded slice of decode
    scratch. It is NOT 2x the largest tensor: an earlier version assumed
    that, which predated compressed_store's slice-level decoding and so
    demanded 3.12 GB of headroom for a 1.56 GB embedding, rejecting
    budgets that are in fact workable.

    Reserving this up front is what turns "budget too small" into an
    immediate, explainable refusal rather than an OOM on the first layer --
    the failure mode that killed the earlier fixed-residency design at both
    4 GB and 6 GB.

    KNOWN LIMITATION, and the reason AirLLM reaches 1.57 GB where this
    planner's floor is ~2.1 GB on a 14B: an embedding table does not
    actually need to be materialized in full. Only the rows for the current
    tokens are read, so gathering those rows (on CPU, or row-wise from the
    compressed store) would drop its contribution from 1.56 GB to
    kilobytes. Until that is implemented, the largest tensor sets a hard
    floor on any achievable budget.
    """
    tensors = list(tensors)
    if not tensors:
        return VramPlan(int(budget_gb * 1e9), 0, [], [], 0, 0, True, "")

    budget = int(budget_gb * 1e9)
    largest = max(t.orig_bytes for t in tensors)
    if headroom_bytes is None:
        headroom_bytes = largest + scratch_bytes

    available = budget - headroom_bytes
    if available < 0:
        return VramPlan(
            budget, headroom_bytes, [], [t.key for t in tensors], 0,
            sum(t.comp_bytes for t in tensors), False,
            "budget %.2f GB is below the %.2f GB needed just to materialize "
            "the largest tensor and its decode scratch"
            % (budget / 1e9, headroom_bytes / 1e9))

    # Highest bus-traffic-avoided per VRAM byte first (see module docstring).
    ranked = sorted(tensors, key=lambda t: t.value_density, reverse=True)
    if pin_layer_weights:
        ranked = ([t for t in ranked if t.is_layer_weight]
                  + [t for t in ranked if not t.is_layer_weight])

    resident, streamed = [], []
    used = 0
    for t in ranked:
        if used + t.orig_bytes <= available:
            resident.append(t.key)
            used += t.orig_bytes
        else:
            streamed.append(t.key)

    return VramPlan(
        budget_bytes=budget,
        headroom_bytes=headroom_bytes,
        resident_keys=resident,
        streamed_keys=streamed,
        resident_bytes=used,
        streamed_bytes_per_token=sum(
            t.comp_bytes for t in tensors if t.key in set(streamed)),
        feasible=True,
        reason="",
    )


def plan_from_manifest(manifest: dict, budget_gb: float, **kw) -> VramPlan:
    """Build a plan directly from a compressed store's manifest."""
    infos = []
    for key, meta in manifest["tensors"].items():
        infos.append(TensorInfo(
            key=key,
            orig_bytes=meta["orig_bytes"],
            comp_bytes=meta["comp_bytes"],
            is_layer_weight=key.startswith("model.layers."),
        ))
    return plan_residency(infos, budget_gb, **kw)
