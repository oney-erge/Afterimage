"""User-dictated VRAM + RAM budgeting -- three-tier residency planning.

The head-to-head against AirLLM exposed the gap this module closes: our
engine used 5.10 GB against AirLLM's 1.57 GB on the same 14B model, purely
because it kept embed_tokens and lm_head permanently resident while AirLLM
streams them like any other layer. We were faster but far heavier, and a
4 GB cap was impossible for us and trivial for AirLLM.

So residency stops being hardcoded. You state a VRAM budget (and optionally
a RAM budget); the planner decides what fits where.

Three tiers, not two
---------------------
The original version of this planner only chose "resident in VRAM" vs
"streamed from disk every token" -- a binary choice. But on this machine
~19 GB of RAM sits almost entirely idle while every token re-reads the same
bytes from a ~2 GB/s disk. RAM is ~5x faster than that disk and large enough
to hold a real fraction of a compressed model. The planner now fills THREE
tiers greedily, in order of decreasing bandwidth:

    VRAM tier  -- decoded once, lives on GPU permanently, ~free to touch
    RAM  tier  -- decoded once, lives pinned in host RAM, one memcpy/token
    disk tier  -- read + decoded from disk EVERY token (the old-and-only path)

Which tensors to keep resident, when they cannot all fit
----------------------------------------------------------
Every weight in a dense decoder is used exactly once per token, so
"frequency of use" cannot rank them -- it is identical for all. The thing
that does differ is how much streaming each one COSTS relative to the
residency budget it occupies:

    value density = compressed_bytes / uncompressed_bytes

A tensor that compresses poorly costs nearly its full size in bus traffic
every token, so pinning it (in VRAM, then in RAM) avoids the most traffic
per byte of residency spent. A highly compressible tensor is cheap to
re-stream from disk, so it is the better candidate to leave there. Ranking
by this ratio and filling each budget greedily, VRAM first then RAM, is the
fractional-knapsack solution to "minimize bytes moved per token subject to
two residency ceilings" -- exact for the fractional case and a good
approximation here, where items are whole tensors.

This inverts the intuitive answer (keep the biggest / most important
tensors), and it only becomes visible once compression is on the streaming
path, which is why no existing offloading engine does it.

Self-speculation breaks the "used exactly once" assumption (draft-aware ranking)
----------------------------------------------------------------------------------
The paragraph above is only true for plain streaming. Under self-speculative
decoding (docs/archive/PROPOSAL_ADAPTIVE.md mechanism A), the model drafts using its
own first N layers, run once per proposed token in a chain of length k, THEN
all layers run once more to verify -- so layers [0, N) are actually touched
(k + 1) times per sweep while layers [N, n_layers) are touched once. They are
no longer equal, and TensorInfo.uses (default 1, so every existing caller is
unaffected) is what lets the ranking say so: value_density is scaled by
`uses`, so a draft layer's bus traffic saved by pinning it is counted
(k + 1) times over, exactly matching how often eviction would actually cost
a re-stream. This is why self-drafting WITHOUT this awareness (pin_layer_weights
alone is not enough) is expected to be slower, not faster: the draft layers
get re-read from disk up to k times per sweep with no signal telling the
planner they are hot.

Row-gathered tensors are excluded from the knapsack entirely
--------------------------------------------------------------
An earlier version of this planner charged a row-gathered embedding table
its FULL uncompressed size as "bytes streamed per token if evicted" --
correct for a normal tensor, wrong here, because row-gather (see
streaming_engine.py) never materializes the table at all: it reads only the
handful of rows the current input actually needs, on the order of
kilobytes per token, not gigabytes. That is neither a VRAM-resident cost
nor a disk-streamed cost in the sense this planner models, so the honest
fix is to leave row-gathered tensors out of the tiering decision rather
than assign them a number that would either overstate their cost (as
before) or require inventing a per-call-size-dependent cost this planner
has no way to know ahead of time.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TensorInfo:
    key: str
    orig_bytes: int          # size when materialized (bf16), VRAM or RAM
    comp_bytes: int          # bytes read from disk per token if not resident
    is_layer_weight: bool    # part of a decoder layer (vs embedding/head/norm)
    row_gather: bool = False  # excluded from tiering -- see module docstring
    uses: int = 1             # times touched per sweep -- see module docstring
                              # ("Self-speculation breaks..."). 1 for every
                              # tensor under plain streaming; self-draft
                              # layers pass (spec_k + 1) when pin_draft_layers
                              # is set.
    stream_only: bool = False  # never resident: always streamed in blocks
                               # (chunked lm_head -- see materialize_bytes)
    materialize_override: int | None = None  # peak bytes live at once, if it
                               # differs from orig_bytes (chunked projection)
    profiled_value_s: float = 0.0  # measured preparation time avoided
    critical_value_s: float = 0.0  # measured critical-path time avoided
    profiled_observed: bool = False
    critical_observed: bool = False

    @property
    def materialize_bytes(self) -> int:
        """Peak bytes this tensor forces live at once, which is what the
        VRAM headroom reserve must cover. Equals orig_bytes for an ordinary
        tensor; a chunked-projection tensor (lm_head) only ever holds one
        row block, so its override is far smaller and the engine's minimum
        feasible budget drops by the difference."""
        return self.materialize_override or self.orig_bytes

    @property
    def value_density(self) -> float:
        """Bus traffic avoided per byte of residency spent keeping this
        resident, one tier up from wherever it would otherwise be read.
        Scaled by `uses` so a tensor touched multiple times per sweep (a
        self-draft layer) is ranked by the traffic pinning it ACTUALLY
        avoids, not by a single-touch assumption that no longer holds."""
        return (self.uses * self.comp_bytes) / max(self.orig_bytes, 1)

    def placement_density(self, policy: str) -> float:
        if policy == "traffic_density":
            return self.value_density
        if policy == "profiled_knapsack":
            return ((self.uses * self.profiled_value_s) / max(self.orig_bytes, 1)
                    if self.profiled_observed else self.value_density)
        if policy == "critical_path":
            return ((self.uses * self.critical_value_s) / max(self.orig_bytes, 1)
                    if self.critical_observed else self.value_density)
        raise ValueError("unknown placement policy %r" % policy)


@dataclasses.dataclass
class TierPlan:
    vram_budget_bytes: int
    ram_budget_bytes: int
    vram_headroom_bytes: int

    vram_keys: list
    ram_keys: list
    disk_keys: list
    row_gather_keys: list

    vram_bytes: int
    ram_bytes: int
    disk_bytes_per_token: int

    feasible: bool
    reason: str

    @property
    def vram_budget_gb(self) -> float:
        return self.vram_budget_bytes / 1e9

    @property
    def ram_budget_gb(self) -> float:
        return self.ram_budget_bytes / 1e9

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / 1e9

    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / 1e9

    @property
    def disk_gb_per_token(self) -> float:
        return self.disk_bytes_per_token / 1e9

    def tier_of(self, key: str) -> str:
        if key in self.vram_keys:
            return "vram"
        if key in self.ram_keys:
            return "ram"
        if key in self.row_gather_keys:
            return "row_gather"
        return "disk"

    def describe(self) -> str:
        lines = [
            "Tier plan -- VRAM %.2f GB / RAM %.2f GB budget:" % (
                self.vram_budget_gb, self.ram_budget_gb),
            "  vram tier           : %.2f GB across %d tensors"
            % (self.vram_gb, len(self.vram_keys)),
            "  ram  tier           : %.2f GB across %d tensors"
            % (self.ram_gb, len(self.ram_keys)),
            "  disk tier (streamed): %.2f GB/token across %d tensors"
            % (self.disk_gb_per_token, len(self.disk_keys)),
        ]
        if self.row_gather_keys:
            lines.append("  row-gather (excluded): %d tensors" % len(self.row_gather_keys))
        lines.append("  vram headroom        : %.2f GB" % (self.vram_headroom_bytes / 1e9))
        lines.append("  feasible             : %s" % self.feasible)
        if not self.feasible:
            lines.append("  reason               : " + self.reason)
        return "\n".join(lines)


DEFAULT_SCRATCH_BYTES = 512 * 1024 * 1024

# Peak transient bytes the sliced decoder holds per weight in a slice, from
# compressed_store.decompress_layer_gpu + _recombine: the decoded exponent
# (uint8) and the sign/mantissa slice (uint8), promoted to int16 for the
# recombine (e16, sm16, bits) plus a bool mask. 1+1+2+2+2+1 = 9, rounded up
# to 10 for allocator slack.
DECODE_SCRATCH_BYTES_PER_ELEM = 10

# Logits, hidden states and the KV cache -- everything live during a forward
# pass that is not a weight. Small next to a multi-GB weight tensor, but not
# zero, and a budget that ignores it entirely would OOM on the first token
# rather than being refused up front.
DEFAULT_ACTIVATION_SLACK_BYTES = 128 * 1024 * 1024


def decode_scratch_bytes(decode_slice_elems: int) -> int:
    """Transient VRAM the sliced decoder needs for one slice.

    This is what makes low VRAM budgets reachable. The planner used to
    reserve a flat 512 MB regardless of how finely the decoder actually
    slices its work, which set a ~2.06 GB floor on a 14B model (1.56 GB
    lm_head + 0.5 GB) even when the real measured peak at small slices was
    1.62 GB -- so budgets the engine could genuinely honour were being
    refused. Deriving the reserve from decode_slice_elems ties the estimate
    to the thing that actually determines it.
    """
    return decode_slice_elems * DECODE_SCRATCH_BYTES_PER_ELEM


def plan_tiers(tensors, vram_budget_gb: float, ram_budget_gb: float = 0.0,
               vram_headroom_bytes: int = None,
               scratch_bytes: int = None,
               decode_slice_elems: int = None,
               activation_slack_bytes: int = DEFAULT_ACTIVATION_SLACK_BYTES,
               pin_layer_weights: bool = False,
               forced_ram_keys=(),
               placement_policy: str = "traffic_density") -> TierPlan:
    """Decide what stays in VRAM, what stays in RAM, and what streams from
    disk every token, for the given budgets.

    vram_headroom_bytes: VRAM reserved for transient decode/activation work,
    on top of whatever is assigned VRAM-resident. Defaults to
    (largest eligible tensor + decode scratch + activation slack), because
    streaming a disk-tier tensor requires its materialized output live at
    once plus a bounded slice of decode scratch. It is NOT 2x the largest
    tensor: an earlier version assumed that, which predated
    compressed_store's slice-level decoding and so demanded headroom far
    larger than actually needed.

    scratch_bytes / decode_slice_elems: two ways to say the same thing.
    Pass decode_slice_elems (preferred -- it is the engine's actual knob,
    see EngineConfig) and the reserve is computed from it; pass
    scratch_bytes to state a flat reserve directly, which existing callers
    and tests do. Neither given falls back to DEFAULT_SCRATCH_BYTES, which
    preserves the original conservative behaviour exactly.

    Reserving this up front is what turns "budget too small" into an
    immediate, explainable refusal rather than an OOM on the first layer.

    ram_budget_gb defaults to 0.0 (no RAM tier -- legacy two-tier
    behaviour), not None, so this function always returns a fully-formed
    TierPlan; callers that want "RAM tier disabled" just don't pass it.
    """
    if scratch_bytes is None:
        scratch_bytes = (decode_scratch_bytes(decode_slice_elems)
                         if decode_slice_elems is not None
                         else DEFAULT_SCRATCH_BYTES)
        scratch_bytes += activation_slack_bytes
    all_tensors = [t for t in tensors if not t.row_gather]
    row_gather_keys = [t.key for t in tensors if t.row_gather]
    forced_ram_keys = set(forced_ram_keys)
    by_key = {t.key: t for t in all_tensors}
    invalid_forced = forced_ram_keys - set(by_key)
    if invalid_forced:
        raise ValueError("forced RAM keys are not placement candidates: %s" %
                         ", ".join(sorted(invalid_forced)))
    forced_stream_only = {key for key in forced_ram_keys if by_key[key].stream_only}
    if forced_stream_only:
        raise ValueError("stream-only tensors cannot be forced into RAM: %s" %
                         ", ".join(sorted(forced_stream_only)))

    if not all_tensors:
        return TierPlan(int(vram_budget_gb * 1e9), int(ram_budget_gb * 1e9), 0,
                        [], [], [], row_gather_keys, 0, 0, 0, True, "")

    vram_budget = int(vram_budget_gb * 1e9)
    ram_budget = int(ram_budget_gb * 1e9)
    largest = max(t.materialize_bytes for t in all_tensors)
    if vram_headroom_bytes is None:
        vram_headroom_bytes = largest + scratch_bytes

    vram_available = vram_budget - vram_headroom_bytes
    if vram_available < 0:
        return TierPlan(
            vram_budget, ram_budget, vram_headroom_bytes, [], [],
            [t.key for t in all_tensors], row_gather_keys, 0, 0,
            sum(t.comp_bytes for t in all_tensors), False,
            "vram_budget %.2f GB is below the %.2f GB needed just to "
            "materialize the largest eligible tensor (%.2f GB) plus decode "
            "scratch and activations (%.2f GB). Lowering decode_slice_elems "
            "shrinks the scratch term and lowers this floor."
            % (vram_budget / 1e9, vram_headroom_bytes / 1e9,
               largest / 1e9, (vram_headroom_bytes - largest) / 1e9))

    # stream_only tensors are never residency candidates -- they are
    # computed in blocks and never held whole, so "keep it resident" is not
    # a choice that exists for them. They still count toward per-token disk
    # traffic and toward the headroom reserve (via materialize_bytes).
    stream_only = [t for t in all_tensors if t.stream_only]
    forced_ram = [by_key[key] for key in sorted(forced_ram_keys)]
    forced_ram_bytes = sum(t.orig_bytes for t in forced_ram)
    if forced_ram_bytes > ram_budget:
        return TierPlan(
            vram_budget, ram_budget, vram_headroom_bytes, [], [],
            [t.key for t in all_tensors], row_gather_keys, 0, 0,
            sum(t.comp_bytes for t in all_tensors), False,
            "forced RAM tensors need %.2f GB, above the %.2f GB RAM budget"
            % (forced_ram_bytes / 1e9, ram_budget / 1e9))
    candidates = [t for t in all_tensors
                  if not t.stream_only and t.key not in forced_ram_keys]

    # Highest bus-traffic-avoided per byte first (see module docstring).
    ranked = sorted(candidates,
                    key=lambda t: t.placement_density(placement_policy), reverse=True)
    if pin_layer_weights:
        ranked = ([t for t in ranked if t.is_layer_weight]
                  + [t for t in ranked if not t.is_layer_weight])

    vram_keys, ram_keys, disk_keys = [], [t.key for t in forced_ram], []
    vram_used = 0
    remainder = []
    for t in ranked:
        if vram_used + t.orig_bytes <= vram_available:
            vram_keys.append(t.key)
            vram_used += t.orig_bytes
        else:
            remainder.append(t)

    # RAM tier fills from whatever VRAM couldn't take, same density order --
    # no separate headroom concept: a RAM-tier tensor is a plain pinned
    # buffer holding an already-decoded result, not a live decode target.
    disk_keys.extend(t.key for t in stream_only)

    ram_used = forced_ram_bytes
    for t in remainder:
        if ram_used + t.orig_bytes <= ram_budget:
            ram_keys.append(t.key)
            ram_used += t.orig_bytes
        else:
            disk_keys.append(t.key)

    return TierPlan(
        vram_budget_bytes=vram_budget,
        ram_budget_bytes=ram_budget,
        vram_headroom_bytes=vram_headroom_bytes,
        vram_keys=vram_keys,
        ram_keys=ram_keys,
        disk_keys=disk_keys,
        row_gather_keys=row_gather_keys,
        vram_bytes=vram_used,
        ram_bytes=ram_used,
        disk_bytes_per_token=sum(
            t.comp_bytes for t in all_tensors if t.key in set(disk_keys)),
        feasible=True,
        reason="",
    )


def _layer_index(key: str) -> int | None:
    if not key.startswith("model.layers."):
        return None
    try:
        return int(key.split(".")[2])
    except (IndexError, ValueError):
        return None


def plan_from_manifest(manifest: dict, vram_budget_gb: float,
                       ram_budget_gb: float = 0.0,
                       draft_layer_indices=None, draft_uses: int = 1,
                       stream_only: dict | None = None,
                       critical_path_profile=None,
                       placement_policy: str = "traffic_density",
                       **kw) -> TierPlan:
    """Build a tier plan directly from a compressed store's manifest.

    draft_layer_indices / draft_uses: mark tensors belonging to decoder
    layers with index in `draft_layer_indices` as touched `draft_uses`
    times per sweep instead of once -- see TensorInfo.uses and the module
    docstring's "Self-speculation breaks..." section. Pass the layer
    indices covered by self-drafting (range(draft_exit_layer)) and
    draft_uses=spec_k + 1 to make the planner prioritize pinning them.
    Neither argument given reproduces plain single-use ranking exactly.

    stream_only: {key: peak_bytes_live_at_once} for tensors computed in
    blocks and never held whole (chunked lm_head). They are removed from
    residency ranking entirely, forced to the disk tier, and contribute
    only peak_bytes_live_at_once -- not their full size -- to the headroom
    reserve, which is what lowers the minimum feasible vram_budget_gb.
    """
    hot = set(draft_layer_indices) if draft_layer_indices else set()
    stream_only = stream_only or {}
    def measured(key: str, field: str) -> float:
        if critical_path_profile is None:
            return 0.0
        cost = critical_path_profile.tensors.get(key)
        return float(getattr(cost, field)) if cost is not None else 0.0

    infos = [
        TensorInfo(
            key=key,
            orig_bytes=meta["orig_bytes"],
            comp_bytes=meta["comp_bytes"],
            is_layer_weight=key.startswith("model.layers."),
            row_gather=bool(meta.get("row_gather")),
            uses=(draft_uses if _layer_index(key) in hot else 1),
            stream_only=key in stream_only,
            materialize_override=stream_only.get(key),
            profiled_value_s=(measured(key, "read_s") + measured(key, "decode_s")
                              + measured(key, "transfer_s")),
            critical_value_s=measured(key, "counterfactual_s"),
            profiled_observed=(critical_path_profile is not None
                               and key in critical_path_profile.tensors),
            critical_observed=(critical_path_profile is not None
                               and key in critical_path_profile.tensors),
        )
        for key, meta in manifest["tensors"].items()
    ]
    return plan_tiers(infos, vram_budget_gb, ram_budget_gb,
                      placement_policy=placement_policy, **kw)
