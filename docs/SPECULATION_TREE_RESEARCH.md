# Speculation tree research: H19-H26

This project's own controlled results put fixed `k=8` at the top of the
policies actually tried (fixed, gamma/EWMA, confidence threshold, AdaEDL
entropy stopping, hazard/cost stopping, neural utility stopping) -- see
`afterimage/experiments.py`'s H2/H11 entries. Hazard-cost measured 6.4%
slower and the neural utility policy never diverged from the fixed one.

That result answers "what is the best K for a linear chain" about as far
as it can be pushed. This document reframes the research question as:

> What is the best shape and size of the candidate future we give the
> expensive streamed target?

That is a substantially larger space than chain length, and it is the
subject of a real, current literature (SpecInfer, Sequoia, OPT-Tree,
SpecExec, and the ICLR 2026 CAST paper on inference-cost-aware tree
construction). This document registers the hypothesis sequence and the
configuration architecture for that space, in the order they must actually
be built in, so nothing downstream gets guessed before it can be measured.

**Status as of this document:** H19's measurement primitive is
implemented and tested (`StreamingLosslessModel.measure_candidate_sweep_latency`
in `afterimage/runtime/streaming_engine.py`, driven by
`scripts/run_h19_candidate_sweep.py`). Nothing past H19 is built. See
[Why this lives in a document, not the live registry](#why-this-lives-in-a-document-not-the-live-registry).

## Why this lives in a document, not the live registry

`afterimage/experiments.py`'s `HYPOTHESES` dict and `afterimage/protocols.py`'s
`PROTOCOLS`/`HYPOTHESIS_PROTOCOLS` are tightly coupled, live infrastructure:
`registry_payload()` calls `validate_protocol_registry(HYPOTHESES)`, which
requires an **exact** 1:1 mapping between every hypothesis id and a
protocol id, checked on every `GET /api/experiments` call the product's own
Lab UI makes. Every `TestProtocol`/`EvidenceStage` in that file is shaped
around a **paired candidate-vs-control comparison** (blocks, paired
repeats, tokens per case).

H19 is a monotonic sweep over one arm (candidate positions), not a paired
comparison, and H20-H26 depend on tree-attention machinery that does not
exist yet. Registering any of them in the live dict today would mean
either misrepresenting a sweep as a paired comparison, or extending shared
protocol infrastructure that backs a running product's UI without the
GPU access to verify the change end to end. Neither is acceptable for
something this load-bearing. This document is the pre-registration
instead: the claim, the control, and the kill criterion are written down
before anything runs, which is the actual discipline the live registry
exists to enforce -- just not wired into that specific shared object yet.

When a hypothesis here is ready to become a real paired comparison (H20
onward, once real tree strategies exist), it belongs in `experiments.py`
properly, with a matching protocol entry, the same way H0-H18 are.

## First: a clean configuration architecture (planned, not yet added to EngineConfig)

`afterimage/runtime/config.py` states its own principle plainly: "every
field below is real and load-bearing -- nothing here is scaffolding for a
future feature." Tree-strategy knobs with no implementation behind them
would violate that, so they are **not** added to `EngineConfig` in this
pass. This section records the intended shape for when H20 onward actually
builds something to back it.

Today, `spec_k_policy` conflates two different questions: "how do I choose
the length of a **linear** chain" (fixed / gamma / threshold / adaedl /
hazard_cost / neural_utility). The cleaner split, once tree strategies
exist:

```
spec_strategy:
    none
    linear              # today's spec_k / spec_k_policy chain, unchanged
    exhaustive_tree
    specinfer_tree
    sequoia_tree
    opt_tree
    specexec
    cost_aware_tree
```

with a common knob group every tree strategy would share:

```
draft_model: Qwen/Qwen3-0.6B     # WHO makes predictions

# Linear (already real)
spec_k: 8
spec_k_policy: fixed

# Trees (planned)
spec_node_budget: 128            # HOW MUCH speculative work is allowed
spec_max_depth: 16
spec_branch_factor: 4
spec_beam_width: 64
spec_probability_floor: null

# Runtime adaptation (planned)
spec_adaptive_budget: false
spec_min_budget: 16
spec_max_budget: 512

# Verification/cache (already real)
spec_target_cache: true

# Cost-aware strategies (planned)
spec_cost_profile: null
```

`draft_model` = who makes predictions. `spec_strategy` = what shape of
future is constructed. `spec_node_budget`/`spec_k` = how much speculative
work is allowed. The tree policy = which candidate futures get that
budget. Keeping those four questions separate is what makes the
experiment matrix below comparable across strategies instead of each one
reinventing its own ad hoc config.

## The configurations

| Config | Mental model | Primary knobs | Literature role | Status |
|---|---|---|---|---|
| S0 Exact | No speculation | -- | Control | Exists (`draft_mode="none"`) |
| S1 Linear Fixed | One road | `spec_k` | Leviathan-style baseline / current Afterimage | Exists (`spec-fixed`) |
| S2 Linear Adaptive | One road, variable length | threshold/entropy/etc. | SpecDec++, AdaEDL | Exists (H2, H11) |
| S3 Exhaustive Tree | Explore every top-B branch | branch, depth, budget | Diagnostic control | Planned |
| S4 SpecInfer Tree | Heuristic likely-futures tree | nodes, depth, beam | SpecInfer | Planned |
| S5 Sequoia | Hardware-optimized fixed tree | nodes + hardware cost | Sequoia | Planned |
| S6 OPT-Tree | Probability-optimized tree every round | node budget | OPT-Tree (TACL) | Planned |
| S7 SpecExec | Massive future-probability cache | 64-1024+ nodes | SpecExec | Planned, highest priority |
| S8 Cost-Aware Tree | Optimize probability *and* Afterimage's real cost | dynamic budget/shape | CAST-inspired | Planned, highest novelty |

S1 remains the gold baseline every tree strategy must beat: sweep
`k = 1, 2, 4, 8, 16, 32` at the same target/draft/store/residency/prompts/
temperature/cold-cache protocol already established. The current
controlling result is 9.150 s/token, 2.93x AirLLM 3.2.0 at `k=8`.

## H19 -- Candidate-Amortization Hypothesis (implemented)

**Run this before building any tree.**

> Because Afterimage target inference is dominated by moving/decompressing
> streamed weights, increasing the number of candidate positions evaluated
> in one target sweep will initially increase target-sweep latency much
> more slowly than candidate count.

This does not generate real speculative trees. It makes the target
process different numbers of already-known candidate positions -- their
token *values* do not matter, only their count -- and measures one
verification sweep's wall-clock cost as a function of that count:

```
positions    target-sweep latency
1
2
4
8
16
32
64
128
256
512
1024
```

If the curve looks like

```
positions       target sweep
1               8.0 sec
8               8.1 sec
32              8.4 sec
64              8.8 sec
128             9.6 sec
256            11.0 sec
512            17.5 sec
```

then one sweep containing ~128 candidate positions costs only 20% more
than one position -- meaning `k=8` is probably nowhere near this
hardware's real parallel candidate capacity. This is exactly the regime
SpecExec's authors target for parameter-offloaded models, where hundreds
or thousands of candidate positions can be processed for close to the
cost of one because parameter movement dominates. The knee in that curve
(call it `N_free` / `spec_parallel_knee`) sets every subsequent tree
budget in this document, rather than copying a paper's number tuned for
different hardware.

**Implementation.** `StreamingLosslessModel.measure_candidate_sweep_latency`
reuses the real `forward_logits()` verification path -- the same call
`generate_speculative`'s own target sweep makes -- so the measured latency
is representative of an actual sweep's cost, not a separate synthetic
benchmark. Driven by `scripts/run_h19_candidate_sweep.py`, which sweeps
the counts above with repeats, cooldown, and a dirty-tree gate, matching
this project's existing benchmark discipline, and writes a
`candidate_amortization_curve` with each point's ratio to the N=1
baseline -- the number that actually answers H19.

**Gate:** find the knee, not a fixed pass/fail threshold. This is a
diagnostic measurement, the same role H0 plays for the contextual-control
research line.

## H20 -- Branching Rescue Hypothesis

> At equal target-verification node budget, a branching candidate tree
> commits more tokens per target sweep than a linear chain, because an
> early draft mismatch does not invalidate every remaining candidate.

**S3 Exhaustive top-B tree.** Take the draft's top-B possibilities at
every explored node (not exhaustive over the ~151k vocabulary -- exhaustive
over the top-B choices at each explored prefix). `branch_factor=2` grows
roughly 14/30/62/126 nodes at depth 3/4/5/6; `spec_node_budget` truncates
the tree when needed. Configuration sketch:

```
spec_strategy: exhaustive_tree
spec_branch_factor: 2
spec_max_depth: 6
spec_node_budget: 128
```

Compare approximately equal target work (linear 64 positions vs. a top-2
tree at ~64 nodes) on committed tokens/sweep and seconds/token. If the
tree does not beat the chain even here, the case for the rest of this
document's tree-based strategies weakens substantially -- this is the
cleanest possible test of whether branching helps Afterimage at all, which
is why it is worth building even though it will probably not become the
final algorithm.

**Control:** S1 at an equal node budget. **Gate:** higher tok/s and
committed tokens/sweep than the equal-budget linear chain.

## H21 -- Hardware-Optimized Tree Hypothesis

> A tree optimized using Afterimage's measured target-verification-cost
> curve achieves higher throughput than an equal-budget heuristic
> SpecInfer/exhaustive tree.

**S4 SpecInfer-style tree** spends a limited node budget where the draft
is actually confident, sharing common prefixes, instead of branching
equally everywhere:

```
spec_strategy: specinfer_tree
spec_node_budget: 128
spec_max_depth: 16
spec_beam_width: 32
spec_branch_factor: 4
```

**S5 Sequoia** adds tree-shape optimization and hardware-aware budget
selection: offline calibration measures target-verification cost at
16/32/64/128/... nodes and how often the draft succeeds at each level,
then a DP-style optimizer picks the best *fixed* tree for this machine --
adaptive to the hardware at calibration time, fixed at runtime.

```
spec_strategy: sequoia_tree
spec_node_budget: 128
spec_cost_profile: rtx3080_qwen14b.json
```

Sequoia's own paper specifically reports large gains in an offloading
setting, which is why it belongs in this comparison rather than being
skipped for a datacenter-only method. Sweep budgets 32/64/128/256/512.

**Control:** S3/S4 at equal budget. **Gate:** >5-8% throughput.

## H22 -- Context-Adaptive Tree Hypothesis

> At an equal node budget, dynamically allocating nodes according to
> current draft probabilities yields more committed tokens per target
> sweep than a fixed hardware-optimized tree.

**S6 OPT-Tree** differs from Sequoia in a specific way: Sequoia learns one
good tree shape for this model/hardware; OPT-Tree recomputes the tree
shape every round from the *current* continuation's probabilities --
narrow and deep when the draft is confident, wide when it is not. The
TACL paper reports it outperforms fixed draft structures and keeps
increasing acceptance length at large node budgets when the drafter is
strong, and it is explicitly designed to work with arbitrary
autoregressive draft models (so the existing Qwen3-0.6B drafter applies
directly).

```
spec_strategy: opt_tree
spec_node_budget: 128
spec_max_depth: 32
```

Sweep budget 32/64/128/256/512. **Control:** S5 (Sequoia) at equal
budget -- this is the comparison that cleanly tests OPT-Tree against
Sequoia. **Gate:** >5-8% throughput.

## H23 -- Massive Speculative Cache Hypothesis

> For compressed/offloaded Afterimage inference, a SpecExec-style large
> future-prefix cache reduces expensive target sweeps per generated token
> enough to outperform both linear speculation and smaller
> acceptance-optimized trees.

**S7 SpecExec** asks a different question than OPT-Tree. OPT-Tree asks
"what candidate tree maximizes the expected accepted sequence." SpecExec
asks "what are the most probable future prefixes I can precompute with my
candidate budget" -- the target evaluates that large tree once, and the
result becomes a cache of target probability distributions generation can
walk through until it reaches an unprecomputed state. SpecExec was
explicitly designed for offloaded consumer-device inference (unusually
close to Afterimage's own regime) and reports up to ~20 tokens per
target-model iteration. The official implementation itself sweeps
16/32/64/128/256/512/1024 nodes (larger-model ablations go to 2048/4096).

**Do not start at 4096. H19 says where to stop.**

```
# SX-64 / SX-128 / SX-256 / SX-512
spec_strategy: specexec
spec_node_budget: 64   # then 128, 256, 512, and 1024 only if H19 supports it
```

**Correction to the existing record:** H16 (`afterimage/experiments.py`)
cites SpecExec, but H16 was speculation-conditioned critical-path
residency plus fixed speculation -- not an implementation of SpecExec's
massive-cache tree -- and it regressed 2.75% in that specific composition.
**H16 failing does not mean SpecExec failed in Afterimage.** Actual
SpecExec remains untested; H23 is that test.

**Control:** S6 (OPT-Tree) and S1 (linear) at equal budget. **Gate:**
lower s/token than both.

## H24 -- I/O-Aware Cost-Optimized Tree Hypothesis

> An online tree builder that optimizes expected target-sweep time saved
> minus measured marginal candidate cost will outperform probability-only
> tree construction on an offloaded compressed target.

The ICLR 2026 CAST paper's core criticism is worth taking seriously:
maximizing acceptance is not the same thing as maximizing speed.
EAGLE/OPT-style methods can decide 200 candidates are probabilistically
valuable when, on this GPU, 256 candidates cost 14s against 128's 9s --
the extra acceptance is not worth the extra cost. CAST accounts for
inference costs like GPU setup and batch size and reports improvements
over existing dynamic tree approaches.

For Afterimage the cost model is not just GPU candidate compute -- it is
disk read + Huffman decode + host/GPU transfer + resident tensors + layer
compute x candidate count + attention + output head + draft cost. The
decision this project's version of the idea should make:

```
expected value of adding a candidate node =
    P(it saves a future target sweep) x cost(that future target sweep)
    vs.
    marginal cost of adding this node to the current target sweep
```

```
spec_strategy: cost_aware_tree
spec_min_budget: 16
spec_max_budget: 512
spec_adaptive_budget: true
spec_cost_profile: afterimage_rtx3080_qwen14b.json
```

Same request, different regime: confident text might get a deep 128-node
tree; uncertain text a wide 256-node tree; expensive long context 64
nodes; a disk-bound cold run 512 nodes.

**Novelty note:** CAST means the general concept "cost-aware speculative
tree" cannot be claimed as novel. Incorporating Afterimage's own
compressed-storage/decompression/residency/I/O state into the tree
utility function is a more specific, potentially distinctive systems
formulation -- but that claim needs a dedicated novelty search before
being made in anything published, not assumed here.

**Control:** the better of H22/H23's winner at equal budget. **Gate:**
>5-8% throughput.

## H25 -- meta-controller, deliberately not built yet

BanditSpec (found in this project's own literature review) adaptively
chooses speculative-decoding hyperparameters online via multi-armed
bandits and approaches the oracle configuration in its own experiments.
**Do not implement this now.** H0 found only a 2.56% semantic-only oracle
gap between the limited policies it tested (see the corrected framing of
H0 in `afterimage/experiments.py`), and H3's contextual controller then
captured essentially none of it -- but that action space was tiny (a
handful of linear-chain policies).

> Online strategy selection helps only if the action-space oracle gap,
> recomputed across `{linear, OPT-Tree, SpecExec-128/256/512,
> cost-aware}`, is large.

**Gate to even attempt this:** recompute best-per-request vs. best-global
oracle gap after H20-H24 exist. If it is above 10-15%, resurrect
BanditSpec/meta-selection. If it is still 2-3%, do not bother -- the
larger action space does not change what H0/H3 already found about how
much a per-request controller can capture.

## H26 -- better drafting, deliberately last

> Better drafting increases cache depth enough to justify its cost.

**Do not train a new drafter before H19-H24 exist.** If a new drafter
appears to improve performance before then, there is no way to tell
whether it helped because of better token accuracy, better tree coverage,
different confidence calibration, or because one tree architecture
happened to suit it. Freeze `Qwen/Qwen3-0.6B`, answer "what is the best
way to spend 128/256/512 candidate positions" first, then ask the more
interesting question this hypothesis is actually about: can a draft model
be designed specifically to maximize the number of expensive Afterimage
target sweeps avoided per millisecond and per GB of VRAM.

## Measurement schema

Every speculative run in this research line should emit:

```
spec_strategy
draft_model

draft_tokens_generated
draft_seconds

candidate_nodes
tree_depth
tree_width
tree_build_seconds

target_sweeps
target_sweep_seconds

candidate_nodes_per_sweep
committed_tokens_per_sweep
accepted_draft_tokens

target_cache_hits
target_cache_misses
cache_walk_length

bytes_read
decode_seconds
io_seconds
compute_seconds

seconds_per_token
tokens_per_second
peak_vram
```

H19 specifically also needs (already implemented, see
`measure_candidate_sweep_latency`'s return shape):

```
candidate_positions
verification_sweep_seconds
io_seconds / decode_seconds / compute_seconds  (already present)
```

**The metric this whole research line is really chasing:** GB of target
weights read per generated token. Compression attacks the numerator once
(29.5 GB -> ~20.3 GB for Qwen3-14B). Speculation attacks how often that
20.3 GB has to be read at all. If SpecExec-class speculation reaches 8
committed tokens per target sweep, that is ~2.5 GB of streamed target
weights per generated token instead of 20.3 GB/token -- a larger lever
than any further improvement to the Huffman codec itself.

## Implementation order

1. **H19 first.** Done -- see Status above. Do not build sophisticated
   trees before knowing this machine's actual candidate-capacity curve.
2. **A generic tree representation and tree-attention verifier.**
   Exhaustive, SpecInfer, OPT-Tree, and SpecExec should share one
   underlying representation rather than four separate implementations.
   This is real correctness-sensitive engine work (attention masking,
   position ids, tree topology) that needs GPU verification against a
   reference forward pass, the same bit-exactness standard
   `tests/test_streaming_engine_gpu.py` already holds every other engine
   change to -- not something to write blind.
3. **Exhaustive top-B and a simple SpecInfer-style tree** (H20). Validates
   that branching actually helps and establishes the tree-strategy
   controls.
4. **OPT-Tree** (H22). A relatively clean probability-adaptive benchmark
   that applies directly to the existing Qwen3-0.6B drafter.
5. **Real SpecExec** (H23). Sweep node budgets from H19's actual measured
   curve rather than copying 1024/2048 from the paper. SpecExec's offload
   assumptions are unusually close to Afterimage's own.
6. **H24 (cost-aware) only after the cost curves exist** -- built from
   H19's real marginal verification cost plus draft probabilities, not
   guessed.
7. **Only then touch the drafter neural network** (H26).

## Sources

SpecInfer, Sequoia, OPT-Tree (TACL), SpecExec, CAST (ICLR 2026),
BanditSpec -- see `docs/LITERATURE.md` Part III for the fuller survey this
document's framing builds on.
