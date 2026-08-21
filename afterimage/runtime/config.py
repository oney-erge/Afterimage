"""Engine configuration.

The default is strictly lossless, and that is deliberate: the comparison this
project is built around is against AirLLM, which does not quantize either.
Turning quantization on makes the head-to-head measure two different things,
so it is opt-in and never silent.

This is also the single object that drives residency (vram_budget_gb /
ram_budget_gb -> vram_planner.plan_tiers) and I/O depth
(io_prefetch_depth). Before this, StreamingLosslessModel took a handful of
ad-hoc constructor kwargs (vram_cap_gb, empty_cache_every, prefetch) and this
class existed but was never passed to it -- chunk_size/block_chunks/max_bits
were plumbed through separate function arguments instead, and
vram_budget_gb did not exist as a field at all despite docs referencing it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json


@dataclasses.dataclass
class EngineConfig:
    """
    quantize
        None   -- strictly lossless (DEFAULT). Output is bit-identical to the
                  original bf16 model: verified on real models by comparing
                  every weight, the forward-pass logits (max abs diff 0.0),
                  and the generated token ids.
        "q8"   -- OPT-IN LOSSY. Group-wise 8-bit quantization applied before
                  entropy coding. Measured on real layers: ~2.0x compression
                  at 0.55% mean relative output error, versus ~1.46x at 0.00%
                  for lossless. It compresses better than lossless does, and
                  0.55% is small enough that most deployments would not
                  notice -- but it is NOT lossless, so any run using it must
                  not be reported as such.

    chunk_size
        Symbols per independently-decodable chunk. Larger amortizes per-chunk
        overhead; smaller gives the GPU more parallel work. 1024 measured
        best on this hardware.

    block_chunks
        Chunks decoded per Triton program. Must be a power of 2. 32 (the
        NVIDIA warp width) measured fastest on this hardware -- see
        gpu_decode_v2.decode_gpu_v2. AMD wavefronts are 64 wide; this default
        has only ever been tuned against NVIDIA hardware and should be
        re-autotuned, not assumed, on ROCm.

    max_bits
        Ceiling on Huffman code length, which bounds the decode LUT to
        2**max_bits entries. Only an upper bound; the real table is usually
        smaller.

    vram_budget_gb
        How much VRAM residency (beyond decode/activation scratch) the user
        allows the planner to spend keeping tensors permanently on-GPU.
        None means "use the old fixed policy" (embed_tokens/lm_head/norms
        resident, every layer streamed) for backward compatibility; a number
        hands the decision to vram_planner.plan_tiers instead.

    ram_budget_gb
        How much pinned host RAM the planner may spend caching DECODED
        tensors that don't fit the VRAM budget, so they come from a memcpy
        instead of a disk read + Huffman decode on every subsequent token.
        None disables the RAM tier (current/legacy behaviour: everything not
        VRAM-resident is read from disk every token).

    vram_cap_gb
        Hard ceiling passed to torch.cuda.set_per_process_memory_fraction --
        a safety net independent of vram_budget_gb's residency PLANNING.
        Planning decides what SHOULD live where; this is what stops the
        allocator from silently creeping past it. None disables the cap.

    io_prefetch_depth
        How many layers ahead the background reader tries to stay, using a
        small pool of reader threads (each with its own file handle, so
        depth-N reads never contend for one seek cursor). 1 matches the
        original single-thread-ahead design; deeper values give slower NVMe
        more queued work at the cost of more RAM held as in-flight buffers.

    decode_slice_elems
        Weights decoded per bounded GPU slice. Decoding a tensor in
        chunk-ranges rather than all at once is what keeps peak VRAM to
        (full output + one slice of scratch) instead of (full output + a
        full second copy) -- see compressed_store.decompress_layer_gpu.
        Smaller values shrink transient decode scratch (measured: the
        default costs ~300 MB of scratch on a 778M-weight lm_head, which
        is most of the residual VRAM gap against AirLLM, since AirLLM
        stores weights uncompressed and needs no decode scratch at all)
        at the cost of more kernel launches. Purely a memory/throughput
        knob: it can never change the decoded VALUES, and
        tests/test_sliced_decompress.py asserts exactly that.

    empty_cache_every
        Release cached GPU blocks back to the driver every N layer frees.
        0 disables. Non-zero costs a synchronize per call, so it is a
        memory/throughput tradeoff, not a free win.

    progress
        Print a live layer/token progress bar. Streaming is slow by
        construction, so a silent run is indistinguishable from a hang.

    ram_tier_format
        "decoded" (default, for backward compatibility) -- a RAM-tier
        tensor is decoded ONCE at load and cached as a PINNED bf16 tensor,
        so every subsequent token pays only a memcpy to GPU.
        "compressed" -- cache the COMPRESSED bytes instead (plain, unpinned
        host memory). Fits ~1.45x more tensors in the same ram_budget_gb
        (docs/PROPOSAL.md H1), at the cost of a real GPU Huffman decode on
        every token instead of a memcpy.

        ENVIRONMENT NOTE, found while measuring this on the real 14B:
        "decoded" first calls pin_memory(), which page-locks host RAM via
        the OS's mlock. WSL2 commonly limits locked memory to 64 MB, smaller
        than one realistic tensor. The engine then emits a RuntimeWarning
        and uses correctness-preserving pageable RAM; results expose the
        affected keys because transfer performance is no longer the intended
        pinned-RAM experiment. Check `ulimit -l` before interpreting speed.

    lm_head_slice_rows
        Rows of lm_head computed per block, or 0 (default) to materialize
        the whole tensor as before.

        *** OPT-IN AND NOT BIT-EXACT. Setting this makes is_lossless False. ***
        Measured on this hardware at real 14B dimensions (hidden=5120):
        splitting the output projection into row blocks changes the logits
        by up to 2.0 absolute. The decompressed WEIGHTS are still exact --
        the deviation is in the matmul. cuBLAS selects a different kernel
        (and split-K reduction strategy) per output shape, so a blocked
        product accumulates in a different order than a single full one,
        and bf16 rounding then differs. Forcing fp32 accumulation
        (allow_bf16_reduced_precision_reduction=False) does NOT fix it --
        that was tested, and the deviation only fell from 2.0 to 1.0.
        At small dimensions (hidden=64) it happens to be bit-identical,
        which is exactly why this was verified at production shape rather
        than trusted from the tiny-model tests.

        lm_head is the single largest tensor in a decoder-only model
        (1.556 GB on a 14B: 151936 vocab x 5120 hidden, bf16), and
        materializing it whole is what sets this engine's minimum VRAM
        budget -- the planner must reserve headroom for the largest tensor
        it will ever have live, so no budget under ~1.7 GB was expressible
        regardless of how little else was resident. That is also the floor
        AirLLM sits at, for the same reason.

        Logits are a concatenation over output rows
        (logits[..., a:b] = x @ W[a:b].T, no interaction between blocks),
        so the projection can be computed in row blocks with only one
        block's weights live at a time. Setting this to e.g. 8192 drops
        lm_head's contribution to peak VRAM from 1.556 GB to ~84 MB,
        lowering the whole engine's floor by well over a gigabyte.

        Cost: the compressed bytes are read once per token as before (this
        trades no I/O, only peak memory), and lm_head is always streamed,
        never VRAM- or RAM-resident, since the point is not holding it.
        The real cost is the bit-exactness above -- which is why this is
        off by default and why no lossless head-to-head number may be
        reported from a run that enables it.

    draft_mode
        "none" (default) -- generate_adaptive is unavailable; use
        generate_greedy / generate_speculative as before. Nothing below
        changes any existing behaviour.
        "model" -- generate_adaptive drafts with a separate, fully-resident
        small model (same mechanism as generate_speculative's draft_model).
        "self" -- generate_adaptive drafts using THIS model's own first
        `draft_exit_layer` layers, reusing the existing model.norm/lm_head
        as the exit head (LayerSkip-style self-speculation,
        docs/PROPOSAL_ADAPTIVE.md mechanism A). No new parameters, nothing
        trained: the draft literally IS the target, run shallow.

    draft_exit_layer
        Required when draft_mode="self". The draft forward runs layers
        [0, draft_exit_layer) then applies model.norm + lm_head directly to
        that hidden state, skipping [draft_exit_layer, n_layers). Must be
        >= 1 and < the model's real layer count (checked at draft time,
        once n_layers is known from the loaded model).

    spec_k
        Draft chain length. For spec_k_policy="fixed" this is constant, as
        in generate_speculative. For adaptive policies it is the maximum or
        initial chain length -- see runtime/spec_policy.py.

    spec_k_policy
        "fixed" (default) -- spec_k unchanged for the whole run; the control
        arm every adaptive policy is measured against.
        "gamma" -- GammaTune-style EWMA over recent acceptance; expands k
        when acceptance is high, contracts it when low.
        "threshold" -- SpecDec++-style: stop drafting once the draft's own
        confidence for the next token drops below a learned threshold.
        "hazard_cost" -- a survival/cost rule learned from accepted prefixes
        and first rejections.
        "neural_utility" -- a six-hidden-unit censored-survival MLP that uses
        confidence, entropy, and position to stop when another draft token is
        predicted to reduce committed tokens per second.
        Every policy is safe to explore aggressively with because the
        accept/reject step in runtime/verify.py guarantees the SAME output
        distribution for any k -- a bad choice costs a slow sweep, never a
        wrong token. See runtime/spec_policy.py's module docstring.

    pin_draft_layers
        Only meaningful with draft_mode="self". When True, the residency
        planner treats layers [0, draft_exit_layer) as used (spec_k + 1)
        times per sweep instead of once, and ranks them accordingly (see
        vram_planner's `uses` field and docs/PROPOSAL_ADAPTIVE.md mechanism
        C). Requires vram_budget_gb to be set: there is no tiering decision
        to influence under the legacy fixed-residency policy. Default False
        because self-drafting WITHOUT pinning re-streams the draft layers
        spec_k times per sweep and is expected to be slower, not faster --
        this flag is not a free win, it is the fix for a cost this mode
        introduces.

    spec_policy_state
        Optional path. If set, the chosen SpecPolicy's state is loaded from
        this file at the start of generate_adaptive and saved back at the
        end, so a bandit/EWMA policy accumulates evidence ACROSS separate
        runs -- a single run is typically only a handful of sweeps, too few
        to converge from scratch every time. None (default) means no
        persistence: the policy starts fresh and its learning is discarded
        when the call returns.

    spec_policy_learn
        True (default) updates policy state from each sweep. False loads a
        previously calibrated state but freezes it for held-out evaluation.

    placement_policy
        "traffic_density" (default) preserves the existing compression-aware
        residency ranking. "profiled_knapsack" ranks measured latency saved
        per resident byte. "critical_path" uses event-DAG slack/criticality in
        addition to the measured latency. "replay_cem" loads a whole-set plan
        learned offline by cross-entropy search against event-DAG replays. It
        captures placement interactions but is never allowed to explore on a
        live generation request.

    prefetch_policy
        "fixed" (default) preserves io_prefetch_depth exactly. "pi" and
        "mpc" use queue/starvation feedback to select a depth in
        [0, io_prefetch_max_depth]. Both alter scheduling only, never bytes or
        tensor values.

    execution_policy
        Request-boundary selection among complete, validated method profiles.
        "static" is the default. Learned choices are opt-in and never mutate
        the profiles they select.

    representation_policy / expert_codec
        Opt-in physical-layout research controls. They are consumed by the
        representation and exact-reference planners; a runtime must refuse a
        requested representation when its store lacks the required artifact.

    lm_head_policy
        "full" preserves the current path. "certified_mips" is greedy-only
        exact argmax search with a full fallback; it is never used for
        sampling. "ram_overlay" keeps the exact decoded output head in pinned
        host RAM and copies it into VRAM only for the final projection, after
        decoder-layer weights have died. The existing lm_head_slice_rows option
        remains the explicit lossy/chunked path for backward compatibility.
        The MIPS CPU float64 index is bounded by mips_index_ram_limit_gb (16 GB
        by default).
    """

    quantize: str | None = None
    chunk_size: int = 1024
    block_chunks: int = 32
    max_bits: int = 16

    vram_budget_gb: float | None = None
    ram_budget_gb: float | None = None
    vram_cap_gb: float | None = None
    io_prefetch_depth: int = 1
    decode_slice_elems: int = 1 << 25
    empty_cache_every: int = 0
    progress: bool = False
    ram_tier_format: str = "decoded"
    lm_head_slice_rows: int = 0

    draft_mode: str = "none"
    draft_exit_layer: int | None = None
    spec_k: int = 8
    spec_k_policy: str = "fixed"
    pin_draft_layers: bool = False
    spec_policy_state: str | None = None
    spec_policy_learn: bool = True

    placement_policy: str = "traffic_density"
    critical_path_profile: str | None = None
    replay_plan_state: str | None = None
    prefetch_policy: str = "fixed"
    io_prefetch_max_depth: int = 8
    prefetch_target_ready: float = 0.85
    prefetch_kp: float = 2.0
    prefetch_ki: float = 0.25
    execution_policy: str = "static"
    execution_policy_state: str | None = None
    representation_policy: str = "uniform"
    lm_head_policy: str = "full"
    mips_index_ram_limit_gb: float = 16.0
    expert_codec: str = "independent"
    trace_events: bool = False
    trace_output: str | None = None

    def __post_init__(self) -> None:
        if self.quantize not in (None, "q8"):
            raise ValueError(
                "quantize must be None (lossless) or 'q8', got %r" % (self.quantize,))
        if self.block_chunks & (self.block_chunks - 1) != 0:
            raise ValueError(
                "block_chunks must be a power of 2 (Triton tl.arange), got %d"
                % self.block_chunks)
        if not (1 <= self.max_bits <= 16):
            raise ValueError("max_bits must be in [1, 16], got %d" % self.max_bits)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1, got %d" % self.chunk_size)
        if self.io_prefetch_depth < 0:
            raise ValueError(
                "io_prefetch_depth must be >= 0 (0 disables prefetch), got %d"
                % self.io_prefetch_depth)
        if self.decode_slice_elems < 1:
            raise ValueError("decode_slice_elems must be >= 1, got %d"
                             % self.decode_slice_elems)
        if self.ram_budget_gb is not None and self.vram_budget_gb is None:
            raise ValueError(
                "ram_budget_gb requires vram_budget_gb to also be set -- the "
                "three-tier planner ranks VRAM residency first and RAM "
                "residency second, so a RAM budget with no VRAM budget is "
                "an incompletely-specified plan, not a sensible default")
        if self.ram_tier_format not in ("decoded", "compressed"):
            raise ValueError(
                "ram_tier_format must be 'decoded' or 'compressed', got %r"
                % (self.ram_tier_format,))
        if self.lm_head_slice_rows < 0:
            raise ValueError("lm_head_slice_rows must be >= 0 (0 disables), got %d"
                             % self.lm_head_slice_rows)
        if self.draft_mode not in ("none", "model", "self"):
            raise ValueError(
                "draft_mode must be 'none', 'model' or 'self', got %r"
                % (self.draft_mode,))
        if self.draft_mode == "self" and (
                self.draft_exit_layer is None or self.draft_exit_layer < 1):
            raise ValueError(
                "draft_mode='self' requires draft_exit_layer >= 1, got %r"
                % (self.draft_exit_layer,))
        if self.spec_k < 1:
            raise ValueError("spec_k must be >= 1, got %d" % self.spec_k)
        if self.spec_k_policy not in ("fixed", "gamma", "threshold", "adaedl",
                                      "hazard_cost", "neural_utility"):
            raise ValueError(
                "spec_k_policy must be 'fixed', 'gamma', 'threshold', "
                "'adaedl', 'hazard_cost' or 'neural_utility', got %r"
                % (self.spec_k_policy,))
        if self.pin_draft_layers and self.draft_mode != "self":
            raise ValueError(
                "pin_draft_layers only means something with draft_mode='self' "
                "(there are no draft layers to pin otherwise)")
        if self.pin_draft_layers and self.vram_budget_gb is None:
            raise ValueError(
                "pin_draft_layers requires vram_budget_gb -- there is no "
                "tiering decision to influence under the legacy fixed-"
                "residency policy")
        if self.placement_policy not in (
                "traffic_density", "profiled_knapsack", "critical_path",
                "replay_cem"):
            raise ValueError("unknown placement_policy %r" % self.placement_policy)
        if (self.placement_policy in ("profiled_knapsack", "critical_path")
                and not self.critical_path_profile):
            raise ValueError(
                "placement_policy=%r requires critical_path_profile with measured "
                "per-tensor costs" % self.placement_policy)
        if self.placement_policy == "replay_cem" and not self.replay_plan_state:
            raise ValueError(
                "placement_policy='replay_cem' requires replay_plan_state built "
                "from separate calibration traces")
        if self.placement_policy != "traffic_density" and self.vram_budget_gb is None:
            raise ValueError(
                "placement_policy=%r requires vram_budget_gb -- without a budget "
                "there is no placement decision to optimize" % self.placement_policy)
        if self.prefetch_policy not in ("fixed", "pi", "mpc"):
            raise ValueError("prefetch_policy must be 'fixed', 'pi' or 'mpc', got %r"
                             % self.prefetch_policy)
        if self.io_prefetch_max_depth < 0:
            raise ValueError("io_prefetch_max_depth must be >= 0")
        if (self.prefetch_policy != "fixed"
                and self.io_prefetch_max_depth < self.io_prefetch_depth):
            raise ValueError("io_prefetch_max_depth must be >= io_prefetch_depth")
        if not (0.0 < self.prefetch_target_ready <= 1.0):
            raise ValueError("prefetch_target_ready must be in (0, 1]")
        if self.prefetch_kp < 0 or self.prefetch_ki < 0:
            raise ValueError("prefetch PI gains must be non-negative")
        if self.execution_policy not in ("static", "linucb", "thompson", "model_based_rl"):
            raise ValueError("unknown execution_policy %r" % self.execution_policy)
        if self.representation_policy not in ("uniform", "per_tensor", "multi_state"):
            raise ValueError("unknown representation_policy %r" % self.representation_policy)
        if self.lm_head_policy not in ("full", "certified_mips", "ram_overlay"):
            raise ValueError(
                "lm_head_policy must be 'full', 'certified_mips' or "
                "'ram_overlay', got %r" % self.lm_head_policy)
        if self.lm_head_policy != "full" and self.lm_head_slice_rows:
            raise ValueError("non-full lm_head policies and lm_head_slice_rows are "
                             "mutually exclusive")
        if self.lm_head_policy == "ram_overlay" and self.ram_budget_gb is None:
            raise ValueError("lm_head_policy='ram_overlay' requires ram_budget_gb")
        if (self.lm_head_policy == "ram_overlay"
                and self.ram_tier_format != "decoded"):
            raise ValueError("lm_head_policy='ram_overlay' requires decoded RAM")
        if (self.lm_head_policy == "ram_overlay"
                and self.placement_policy == "replay_cem"):
            raise ValueError("ram_overlay and replay_cem are separate v1 experiments")
        if self.mips_index_ram_limit_gb <= 0:
            raise ValueError("mips_index_ram_limit_gb must be positive")
        if self.expert_codec not in ("independent", "xor_reference"):
            raise ValueError("expert_codec must be 'independent' or 'xor_reference', got %r"
                             % self.expert_codec)
        if self.trace_output and not self.trace_events:
            raise ValueError("trace_output requires trace_events=True")


    @property
    def is_lossless(self) -> bool:
        """Bit-identical output, which is the claim the AirLLM comparison
        rests on. Both known ways to break it are opt-in and counted here:
        quantization changes the weights, and a chunked lm_head changes the
        matmul's reduction order (see lm_head_slice_rows)."""
        return self.quantize is None and self.lm_head_slice_rows == 0

    @property
    def uses_tiered_residency(self) -> bool:
        return self.vram_budget_gb is not None

    @property
    def exactness_contract(self) -> str:
        """The strongest contract this configuration can honestly claim."""
        if self.quantize is not None or self.lm_head_slice_rows:
            return "approximate"
        if self.lm_head_policy == "certified_mips":
            return "greedy_token_exact"
        if self.draft_mode != "none":
            return "distribution_exact"
        return "reference_execution_equivalent"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "EngineConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError("unknown EngineConfig fields: %s" % ", ".join(unknown))
        return cls(**values)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        if self.is_lossless:
            lossless = "LOSSLESS (bit-exact output)"
        else:
            why = []
            if self.quantize is not None:
                why.append("quantize=%s" % self.quantize)
            if self.lm_head_slice_rows:
                why.append("lm_head_slice_rows=%d" % self.lm_head_slice_rows)
            lossless = ("LOSSY: %s -- output is NOT bit-exact"
                        % ", ".join(why))
        if not self.uses_tiered_residency:
            return lossless + " | residency: legacy fixed policy"
        return "%s | residency: VRAM %.2f GB%s | io_prefetch_depth=%d" % (
            lossless, self.vram_budget_gb,
            (" + RAM %.2f GB" % self.ram_budget_gb) if self.ram_budget_gb else "",
            self.io_prefetch_depth)
