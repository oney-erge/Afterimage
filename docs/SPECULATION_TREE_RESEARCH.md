# Speculation tree research: H19-H34

This project's own controlled results put fixed `k=8` at the top of the
policies actually tried (fixed, gamma/EWMA, confidence threshold, AdaEDL
entropy stopping, hazard/cost stopping, neural utility stopping) -- see
`afterimage/experiments.py`'s H2/H11 entries. Hazard-cost measured 6.4%
slower and the neural utility policy never diverged from the fixed one.

That result answers "what is the best K for a linear chain" about as far
as it can be pushed. This document reframes the research question twice,
in the order the reframing actually happened:

1. From "what is the best K" to "what is the best **shape and size** of
   the candidate future we give the expensive streamed target" (tree-based
   speculation: SpecInfer, Sequoia, OPT-Tree, SpecExec, the ICLR 2026 CAST
   paper on inference-cost-aware tree construction).
2. From "what shape of tree" to a genuinely different architecture: treat
   cheap models as **imperfect sensors of the expensive target**, fuse
   them into a temporally evolving **belief** about how the target is
   likely to disagree with the drafters, and plan the tree in **belief
   space** rather than directly from one drafter's own probabilities. This
   is "BeliefSpec" below, and it is the closer analogue of robotics
   belief-space planning (POMDP/DESPOT) and actor-critic/sensor-fusion
   ideas than of a classical GAN.

**Status as of this document.** "Implemented" below means the tooling
that PRODUCES evidence exists and is tested; it does not mean the
hypothesis itself has been confirmed. No GPU campaign has been run yet
for any of these -- see [Novelty and evidence
status](#novelty-and-evidence-status) for the full tooling-vs-evidence
distinction this table used to compress away.

| Hypothesis | Tooling | Real evidence |
|---|---|---|
| H19 Candidate Amortization | **Implemented.** `StreamingLosslessModel.measure_candidate_sweep_latency`, `scripts/run_h19_candidate_sweep.py` (warmup, randomized dispatch order, VRAM/RAM-budget configuration, per-case knee detection) | Pending |
| H20 Branching Rescue | Planned -- needs a real tree-attention verifier, GPU-correctness-sensitive, not written blind. CPU-only `SpecNode`/`SpecTree` structural representation is implemented ([SpecTree structure](#spectree-structure-implemented-ahead-of-h20)) | Pending (blocked on the verifier) |
| H21 Multi-Source Oracle Headroom | **Implemented,** as a three-stage pipeline: `scripts/run_h21_collect_target_corpus.py` (target corpus, once), `scripts/run_h21_score_source.py` (any candidate source, reusable), `scripts/run_h21_combine_sources.py` (coverage, equal-total-budget comparison, sustained rescue depth) | Pending |
| H22a Temporal Disagreement Persistence / H22b Online Disagreement Prediction | **Implemented** (split from the original single H22 -- see [H22 detail](#h22a-h22b-persistent-disagreement-state-implemented-split-into-two-claims)), `scripts/run_h22_disagreement_hmm.py`: baseline ladder B0-B5, validation-selected hidden-state count, grouped cross-validation, bootstrapped confidence intervals | Pending |
| H23 Established Tree Baselines | Planned -- same GPU-correctness dependency as H20 | Pending |
| H24 Actual SpecExec | Planned -- budgets set from H19's real measured knee, not guessed | Pending |
| H25-H34 (BeliefSpec-Triad) | Planned, gated on H21/H22's real results (see [Hard research gates](#hard-research-gates)) | Pending |

See [Why the implemented pieces are offline analysis, not live runtime
code](#why-the-implemented-pieces-are-offline-analysis-not-live-runtime-code)
for why H19/H21/H22 could be built now with real confidence while H20 and
everything past it could not.

## Why the implemented pieces are offline analysis, not live runtime code

`afterimage/experiments.py`'s `HYPOTHESES` dict and `afterimage/protocols.py`'s
`PROTOCOLS`/`HYPOTHESIS_PROTOCOLS` are tightly coupled, live infrastructure:
`registry_payload()` calls `validate_protocol_registry(HYPOTHESES)`, which
requires an **exact** 1:1 mapping between every hypothesis id and a
protocol id, checked on every `GET /api/experiments` call the product's own
Lab UI makes. Every `TestProtocol`/`EvidenceStage` in that file is shaped
around a **paired candidate-vs-control comparison**. None of H19-H34 is
that shape -- H19/H21/H22 are diagnostic sweeps and offline analyses over
one arm, and H20 onward would need tree-attention machinery that does not
exist. Registering any of them in the live dict today would mean either
misrepresenting the experiment or extending shared protocol infrastructure
without the GPU access to verify the change end to end. This document is
the pre-registration instead -- the actual discipline the live registry
exists to enforce, just not wired into that specific shared object yet.

`afterimage/runtime/config.py` separately states its own principle: "every
field below is real and load-bearing -- nothing here is scaffolding for a
future feature." Tree-strategy config knobs with no implementation behind
them would violate that, so `EngineConfig` is untouched by this document.
The intended shape is recorded under
[Configuration architecture](#configuration-architecture-planned) for when
there is something real to back it.

**What separates H19/H21/H22 from H20 onward** is not importance, it is
verifiability. H19 and H21 reuse the real `forward_logits()`/
`generate_greedy()` paths this engine already trusts -- they add
measurement, not new execution logic, so there is nothing about them that
needs a NEW correctness proof. H22 is pure CPU statistics (a discrete HMM)
with no engine involvement at all, verified here against synthetic data
with known ground truth (`tests/test_speculation_oracle.py`), including
directly demonstrating both a positive case (the HMM measurably beats a
memoryless baseline on data with real persistent hidden structure) and,
in an ad hoc real run against synthetic zero-structure trace data, the
correct negative case (no benefit on pure noise) -- exactly the dual-
direction verification this kind of tool needs before being trusted on
real collected traces. H20 and every tree/planner/critic hypothesis past
it requires new tree-attention masking, position handling, and (for the
Triad) multi-source distribution-exact sampling math that a bug in would
not crash, it would silently corrupt output -- `tests/
test_streaming_engine_gpu.py` holds every other engine change to bit-
exact-vs-reference before trusting it, and that verification needs a real
GPU this environment does not have.

## The BeliefSpec architecture

Instead of one drafter feeding one target, or several drafters merged
into one candidate pool, give the cheap models **deliberately different
jobs**:

```
                   PRIMARY PROPOSER P
                  "most likely future"
                          |
                     candidates
                          v
                  CANDIDATE SPACE  <---------------+
                          |                        |
                          |                RESCUE SCOUT S
                          |               "what P missed"
                          |                        ^
                     P + S distributions -----------+
                          |
                 DISAGREEMENT CRITIC C
        "how is the target likely to disagree?"
                          |
                posterior target belief
                          |
                BELIEF-SPACE PLANNER
      posterior / scenarios / probes / MPC / MCTS
                          |
                 verification tree
                          |
              AFTERIMAGE TARGET 14B (one expensive sweep)
                          |
              exact target observations
                    /            \
                output        update belief
```

**Primary Proposer (P).** The anchor, not the research question -- today's
resident `Qwen/Qwen3-0.6B` drafter. Its job is depth: be very good at the
single most likely continuation. Fixed `k=8` is already this project's
strongest stable measured configuration (9.150 s/token), which is exactly
why P stays as-is rather than being replaced.

**Rescue Scout (S).** Not a second model competing with P on ordinary
next-token accuracy -- a model whose job is specifically to cover target
continuations **when P is wrong**. The metric that matters is not
standalone accuracy, it is conditional rescue recall:

```
RescueRecall = P(target token in Scout's top-k | target token NOT in Primary's top-k)
```

A scout with 60% standalone accuracy that makes *different* mistakes than
P can be more valuable than a 70%-accurate scout that makes the *same*
mistakes -- that is the actual ensemble-diversity argument, and it is what
H21 measures directly rather than assumes. Candidate scouts, in the risk
order this document's own gates enforce (see
[H21](#h21-multi-source-oracle-headroom-implemented)): same tokenizer,
different training lineage first (lowest risk -- this is what H21's
default `Qwen/Qwen3-1.7B` scout tests); a Mamba/SSM source next (fast
sequential drafting from a structurally different architecture); a
diffusion source after that (parallel-position proposal rather than
autoregressive, per DiffuSpec/D²SD's precedent of a two-role diffusion
drafting architecture); a genuinely different tokenizer/family last, via
something like Universal Assisted Decoding's token-space translation --
tested last because it is the highest-complexity option, not because it is
uninteresting.

**Disagreement Critic (C).** Not a GAN discriminator outputting real/fake
(KOALA already does adversarial speculative-drafting training, so that
framing buys nothing new) -- a model that estimates the **error structure**
of P and S, never generates or verifies a token itself, and has no
permission to commit output. Given P's and S's distributions, their
disagreement, and history (previous acceptance lengths, where rejections
happened, target rank under P and S, recent divergence), it predicts a
*structured* posterior, not one scalar confidence: target-rank-relative-
to-P distribution, target-candidate-location distribution (covered by
both / P only / S only / neither), a latent disagreement-regime posterior
(aligned / weak disagreement / broad ambiguity / severe divergence -- H22's
own hidden states), and branch-survival-by-depth probabilities.

**Belief-space Planner.** The central departure from every existing tree
method. OPT-Tree, for instance, builds `T* = f(q_draft)` -- the tree that
maximizes expected acceptance under the *drafter's own* proposal
probabilities. BeliefSpec instead builds `T* = f(P(target path | q_P, q_S,
history, z_t))` -- the tree optimized against a **posterior belief about
what the target believes**, fused from multiple imperfect sensors, not
against any one sensor's raw output.

**Target.** Unchanged in role: the only thing with authority to commit
output. See [the exactness boundary](#exactness-boundary-never-compromise-this-accidentally).

### Novelty framing

The August 11, 2026 speculative-decoding survey (a preprint, 1,652
candidates considered, 304 works consolidated across draft source,
geometry, verification, and runtime) returns no hits for POMDP, belief,
critic, or DESPOT inside that corpus. That does not prove novelty -- an
obscure patent, unpublished manuscript, or differently-named formulation
could still exist, and this needs a dedicated novelty search before any
claim like this appears in a paper or patent. The narrowest defensible
framing after that search: *structured, history-conditioned inference of
target-drafter disagreement, using heterogeneous proposal sources,
followed by belief-space planning of an exact target-verification tree
under measured offload cost.* Two drafters, heterogeneous drafters, GAN-
trained drafters, and companion acceptance critics are each independently
low-novelty (prior art exists for all of them individually); the
role-separated composition, the posterior-target-tree formulation, the
DESPOT-style scenario tree, and multi-sweep belief-space MPC are where
this document currently sees the clearest unoccupied space -- see
[novelty ranking](#novelty-ranking) for the full table.

## Configuration architecture (planned)

Keep `spec_k_policy` exactly as it is for linear-chain behavior (fixed /
gamma / threshold / adaedl / hazard_cost / neural_utility, all real
today). Add a new, separate layer once something exists to back it:

```
speculation:
  strategy: scenario_tree
  # none / linear / exhaustive_tree / specinfer / opt_tree / sequoia /
  # specexec / posterior_tree / scenario_tree / belief_mcts

  node_budget: 128
  max_depth: 32
  branch_factor: 4
  adaptive_budget: true

  sources:
    - role: proposer
      model: Qwen/Qwen3-0.6B
      weight: 1.0
    - role: scout
      model: null
      weight: 1.0

  critic:
    type: none        # none / hmm / mlp / temporal_transformer
    history_window: 8
    rank_buckets: [1, 2, 4, 8, inf]

  belief:
    scenario_count: 64

  planning:
    probe_fraction: 0.0
    horizon_sweeps: 1
    mcts_simulations: 0
    progressive_widening: false

  target:
    cache: true
    exactness: target_authoritative
```

`draft_model`/`sources` answer WHO makes predictions. `strategy` answers
WHAT SHAPE of future is constructed. `node_budget`/`spec_k` answer HOW
MUCH speculative work is allowed. The tree/planner answers WHICH candidate
futures get that budget. Every hypothesis below is then a configuration,
not special-case code.

### Software architecture (planned)

Five generic interfaces, so every research method shares one correctness
surface instead of each implementing subtly different tree/verification
logic:

```
SpecCandidateSource     PrimaryModelSource, ScoutModelSource,
                         MambaSource, DiffusionSource, ...

SpecTree                 One packed representation: node id, parent id,
                         token id, depth, source id, source probability,
                         posterior target probability, probe flag.

DisagreementModel        NoCritic, HMMCritic, MLPCritic, TemporalCritic.

SpecPlanner              LinearPlanner, ExhaustivePlanner, OPTTreePlanner,
                         SpecExecPlanner, PosteriorPlanner,
                         ScenarioPlanner, BeliefMCTSPlanner.

TargetTreeVerifier        ONE verifier shared by every planner above.
```

### Exactness boundary: never compromise this accidentally

**P, S, C, and the planner may decide WHAT the target evaluates. None of
them may decide WHAT gets committed. Only the target verifier commits.**

Phase A (build first): greedy, temperature=0. If the candidate tree
contains the target's real greedy path, committing that path is
straightforward to validate exactly against `generate_greedy`'s own
reference. Phase B (only after Phase A is solid): derive and test a
genuinely distribution-exact multi-source tree sampler -- this project's
existing linear verifier (`runtime/verify.py`'s `speculative_sample_step`)
correctly uses target/draft probability ratios and residual correction,
and a generalized sampler merging P and S must preserve the same target
distribution mathematically, not just combine logits and call the result
exact. Phase C (last): temperature > 0 benchmarks. Do not mix these three
phases into one experiment.

## The hypothesis sequence

### H19 -- Candidate-Amortization Hypothesis (implemented)

Run this before building any tree. See its own full writeup in this
document's prior revision (preserved below under
[H19 detail](#h19-detail)) -- unchanged by this update, EXCEPT for the
hardening below, applied after real-run pre-flight review found three
gaps: `scripts/run_h19_candidate_sweep.py` now runs an untimed warmup
sweep before the timed loop (so lazy CUDA/kernel compilation cannot land
uniquely on whichever candidate count happens to be measured first,
which the N=1 baseline is especially vulnerable to since every other
point is reported relative to it); randomizes candidate-count dispatch
order within each repeat with a saved `--seed` (breaking the correlation
between candidate size and thermal state / `cool_down()` duration that a
fixed ascending sweep would otherwise create); and accepts multiple
`--case-ids` so the measured knee is checked across more than one prompt
length rather than fitted to a single prompt. **Called a "Candidate
Vectorization/Amortization Curve," never a "Tree Amortization Curve"** --
this measures known candidate positions in a linear sequence, not
arbitrary tree nodes under real tree-attention masking; that distinction
only collapses once H20's verifier exists.

### H20 -- Branching Rescue Hypothesis (planned)

> At equal target-verification node budget, a branching candidate tree
> commits more tokens per target sweep than a linear chain, because an
> early draft mismatch does not invalidate every remaining candidate.

Requires a real `SpecTree` representation and `TargetTreeVerifier` (see
Software architecture above) -- tree-attention masking that a bug in
would silently corrupt output rather than crash, so this is not written
without GPU access to verify it against a reference forward pass, the
same standard `tests/test_streaming_engine_gpu.py` holds every other
engine change to. **Control:** S1 (linear) at an equal node budget.
**Gate (G2):** tree beats equal-budget chain end-to-end, or the case for
every tree-based strategy in this document weakens substantially.

#### SpecTree structure (implemented ahead of H20)

`afterimage/runtime/spec_tree.py` is a CPU-only, pure-Python structural
representation for a candidate tree -- built now so the STRUCTURE a real
verifier will eventually consume can be validated and reasoned about
before any GPU-side tree-attention masking code exists, without
front-running the GPU-correctness work H20 itself requires. It
implements none of tree-attention masking, `TargetTreeVerifier`, a
draft-tree planner, or any CUDA path -- consistent with this document's
own exactness-boundary rule and the "no scaffolding for a future
feature" principle already used elsewhere in `EngineConfig`.

`SpecNode` is a frozen dataclass: `node_id`, `parent_id`, `token_id`,
`depth`, `source` (a plain string label, e.g. `"primary"`/`"scout"`),
`source_prob` (the proposing source's own probability for this token
given its parent path), `posterior_prob` (`None` until H25's
PosteriorTree combines multiple sources' beliefs -- kept as a field now
so that mechanism does not require restructuring the type later), and
`probe` (a bool marking a node inserted purely to gather information for
H29's ProbeSpec rather than to propose a real continuation).

`SpecTree` wraps a list of `SpecNode`s and does NOT validate on
construction -- a caller building a tree incrementally can hold a
temporarily-invalid intermediate state, then call `validate()` once
before handing the tree to anything that trusts its invariants.
`validate()` checks: every non-root `parent_id` refers to a node present
in the tree, every node's `depth` is consistent with its parent's
(`depth == parent.depth + 1`, and `depth == 0` for roots), and no node
is part of a cycle (checked via a bounded hop-count walk to a root,
independent of the depth check, so a malformed input that happens to
satisfy local depth consistency cannot slip a cycle past validation).
`path_to(node_id)` returns the root-to-node path a verifier would need
to reconstruct the actual token sequence a leaf represents, and detects
cycles on its own during the walk rather than assuming `validate()` was
already called. `budget()` returns total node count -- the quantity
every node-budget constraint (H19's own measured `N_free`, or a future
planner's configured cap) actually counts against.

`merge_prefixes(other)` merges two trees that share a common
token-sequence prefix (e.g. Primary and Scout proposing continuations
from the same already-accepted context) into one tree with that prefix
represented once -- the real cost saving a tree-based scheme is supposed
to realize over independently verifying each source's full path. Sharing
is decided level by level and uniformly: two nodes at the same position
are the same candidate when their `(token_id, source)` pair matches, and
matched nodes recurse into their children; anything unmatched on either
side is copied in whole, subtree included. Node IDs are never assumed to
agree between the two inputs, since they come from independent sources
with no shared ID space; output identity is freshly assigned, contiguous
from 0, and output `depth` is recomputed from the merged position rather
than inherited. The merge's own output is run through `validate()`
before being returned.

`SpecTree` is a rooted **forest**, not necessarily a single tree, and
`validate()` deliberately does not require one root. When two sources'
first candidate tokens differ there is genuinely no shared node to hang
them under, and inventing a synthetic root would fabricate a shared
candidate the target would then be asked to verify. Because the root
level is merged by the same rule as every other level, merging is
closed: a multi-root result is itself a legal input to another
`merge_prefixes` call, so three or more sources fold together pairwise.

Two properties of this are worth stating plainly rather than leaving
implicit. First, matching on `(token_id, source)` means Primary and
Scout proposing the SAME token at the same position produce two separate
nodes, each consuming a verification slot -- exact per-source credit
attribution, bought at the cost of not deduplicating agreement, which is
the common case. Which of those matters more is a real open question for
H20/H27, deliberately not settled by this structural layer. Second,
where two matched nodes carry different `source_prob`, self's value
wins silently.

Covered by `tests/test_spec_tree.py` (23 tests): construction,
`children`/`path_to`, every `validate()` failure mode individually
(missing parent, bad root depth, bad child depth, a genuine cycle
distinguished from a depth-consistency failure on the same malformed
input), and `merge_prefixes` across linear, branching, pure-prefix,
disjoint, and multi-root-closure shapes, plus a randomized property
check that a merge never loses or duplicates a node (result size always
within `[max(a, b), a + b]`). That property check exists because an
earlier lockstep implementation of `merge_prefixes` silently dropped
every remaining subtree whenever the two trees diverged by BRANCHING
rather than by a token mismatch -- a 5-node merge returned 1 node, and
`validate()` passed it, because a smaller tree is still a structurally
valid one. Every merge test at the time used linear single-path trees,
which is exactly the shape that cannot expose that bug.

### H21 -- Multi-Source Oracle Headroom (implemented)

> Does a second model actually cover target paths the primary drafter
> misses, before integrating it into any runtime mechanism?

A real, pre-flight-reviewed problem with the first version of this
hypothesis: comparing P-top-8 coverage against P-top-8 UNION S-top-8
coverage lets Scout add candidate slots for free -- an 8+8=16-candidate
union covering more often than an 8-candidate source alone is not
evidence a second model earns its keep, only that more candidates help,
which was never in dispute. The corrected question holds the TOTAL
candidate budget fixed and asks whether TRADING Primary slots for Scout
slots is worth it.

Answered by a three-stage, reusable pipeline (the collection/scoring
split exists specifically so the expensive 14B target trajectory is
generated exactly once, not re-generated for every new candidate source
worth comparing):

1. `scripts/run_h21_collect_target_corpus.py` -- loads ONLY the target,
   generates the real greedy continuation per prompt (the reference
   trajectory; `decoding_mode: greedy` is recorded explicitly in the
   output, since every coverage number that follows is conditioned on
   the argmax path, not a sampled one), writes a target corpus.
2. `scripts/run_h21_score_source.py` -- loads exactly ONE candidate
   model (Primary, Scout, or any future candidate; `--role` is a label,
   not a code path), teacher-forces it against the target corpus, and
   records rank, entropy, top-1/top-2 margin, and top-k IDs+probabilities
   per position. Adding a new candidate to compare costs one run of this
   script, not a full target re-collection.
3. `scripts/run_h21_combine_sources.py` -- joins a target corpus with
   two scored sources (Primary, Scout) and runs the actual analysis:
   - `compute_oracle_coverage_stats` (`afterimage/runtime/
     speculation_oracle.py`): primary/scout/union coverage at several
     rank-based `k`, and conditional rescue recall (`None`, never a
     misleading `0.0`, whenever Primary never missed at that k).
   - `compute_equal_budget_metrics`: the CORRECTED question above. At a
     fixed total candidate budget (default 16), sweeps how many slots go
     to Scout (default 0/4/8/12) versus Primary, and reports
     `equal_budget_union_gain` (coverage minus the Scout=0 baseline at
     the SAME total budget), `marginal_scout_coverage_per_slot`,
     `unique_scout_hit_rate`, and `primary_scout_overlap`.
   - `compute_sustained_rescue_depth`: how FAR a rescue extends, not
     just whether the position immediately after a Primary miss is
     covered -- `P(rescue depth >= d | Primary missed)` for d in
     {1,2,4,8}, teacher-forced on the real trajectory (free from the
     already-collected data; free-running Scout-generates-its-own-
     tokens agreement length is a different, harder-to-define quantity
     that would need new generation, not implemented here).

Default scout is `Qwen/Qwen3-1.7B` against the `Qwen/Qwen3-0.6B` primary
(same tokenizer, different training depth -- lowest risk per this
document's own scout ordering). **Gate (G3):** see [Hard research
gates](#hard-research-gates) for the exact pre-registered thresholds --
"meaningful" is deliberately not a number a paper can move after seeing
results.

### H22a H22b -- Persistent Disagreement State (implemented, split into two claims)

> Is draft/target disagreement temporally predictable beyond the current
> position's own confidence?

A real wording problem in the original single H22: its own baseline
(`memoryless_baseline_nll`) predicts the next observation from the
CURRENT observation, which is `target_rank_under_primary` -- the
target's OWN rank under Primary, known only after a target sweep has
already happened. "Current draft confidence" (Primary's entropy or
margin, known BEFORE that sweep) is a different, and for a live system
much more useful, quantity. The hypothesis is therefore split into what
it actually tests and what it would need to test to be online-usable:

- **H22a (Temporal Disagreement Persistence):** does the SEQUENCE of past
  target-rank observations contain temporal structure a memoryless
  "current bucket alone" model cannot capture? Establishes that state
  exists. Cannot be computed before the target sweep that produces the
  observations it uses.
- **H22b (Online Disagreement Prediction):** can CHEAP information
  available BEFORE the next target sweep (Primary's entropy/margin,
  Scout's entropy/margin, an approximate Primary/Scout Jensen-Shannon
  divergence, recent observation history) predict the upcoming
  disagreement bucket? This is the quantity the belief-space Critic line
  (H25 onward) actually needs, since it cannot see the target's real
  rank before running the sweep that produces it. **H22a passing does
  NOT imply H22b passes** -- a real hidden state can exist in the target-
  rank sequence while being invisible to any signal available before the
  sweep, and the gates below are evaluated independently for exactly
  that reason.

Both are answered by ONE script (`scripts/run_h22_disagreement_hmm.py`)
running a baseline ladder, since forcing every comparison onto the same
folds and the same held-out data is what makes B0-B5 comparable on one
axis:

- **B0** constant unconditional bucket frequency (the floor every other
  baseline must clear).
- **B1** the original memoryless baseline (current bucket only) --
  answers H22a's control.
- **B2** cheap Primary-only features (entropy, margin) via a from-scratch
  multinomial logistic regression (`MultinomialLogisticRegression`,
  no new dependency, same "implement standard statistics locally"
  precedent as `DiscreteHMM`).
- **B3** B2 plus Scout features and the approximate P/S divergence.
- **B4** B3 plus recent observation history (a one-hot of the current
  bucket) -- H22b's actual answer.
- **B5** the fitted `DiscreteHMM` -- H22a's actual answer.

Hidden states are never pre-labeled ("aligned," "severe divergence," ...)
anywhere in the code or its output: a `DiscreteHMM`'s state indices carry
no identity across independent fits (state 0 in one run can be state 2 in
the next), so interpretive labels belong only in a human's post-hoc
reading of one specific fitted model's emission distributions, applied
after the fact, never baked into a field name a reader might mistake for
a stable category.

Hidden-state count is chosen on a VALIDATION split reserved before any
cross-validation happens (`select_n_states_by_held_out_nll`, candidates
2-6 by default), never on the final test folds -- goalpost-moving after
seeing results is exactly what pre-registering the gates below exists to
prevent. `minimum_positions_for_hmm` derives a required minimum
observation count from the actual state/symbol count (20 observations
per free parameter, a conventional MLE stability floor) and the script
refuses to proceed below it without an explicit `--allow-insufficient-
data` override; `tests/test_speculation_oracle.py` demonstrates the
failure mode this guards against empirically -- with only 10 held-out
sequences, state selection on a KNOWN 2-state generator picked 6 states,
because the validation NLL gap between candidates was smaller than
sampling noise at that sample size.

Final evaluation is grouped 5-fold cross-validation (`grouped_k_fold`,
by whole trajectory, never splitting a trajectory's own positions across
folds) with `bootstrap_nll_difference_ci` on the paired per-fold
(baseline − model) NLL differences -- a single point estimate of "the
HMM won by X" says nothing about whether that margin could plausibly be
zero given fold-to-fold variance, which is exactly what the CI-excludes-
zero requirement below checks. **Gates (G4a/G4b):** see [Hard research
gates](#hard-research-gates) for the exact pre-registered thresholds.

### H23 -- Established Tree Baselines (planned)

Required comparisons before claiming anything novel: simple exhaustive
top-B tree, SpecInfer-style heuristic tree, OPT-Tree (probability-
optimized every round, TACL, explicitly designed to work with arbitrary
autoregressive drafters so the existing Qwen3-0.6B drafter applies
directly), and Sequoia-style hardware-optimized fixed tree (offline
calibration of target-verification cost at 16/32/64/128/... nodes, a
DP-style optimizer picks the best fixed tree for this machine). Same
GPU-correctness dependency as H20 -- built on the same `SpecTree`/
`TargetTreeVerifier`.

### H24 -- Actual SpecExec (planned)

> Does massive future-prefix caching exploit Afterimage's I/O-dominated
> offload regime?

SpecExec asks a different question than OPT-Tree: not "what tree maximizes
expected acceptance" but "what are the most probable future prefixes I can
precompute with my candidate budget" -- the target evaluates that large
tree once and the result becomes a cache of target probability
distributions generation can walk through. Explicitly designed for
offloaded consumer-device inference (unusually close to Afterimage's own
regime); reports up to ~20 tokens per target-model iteration in its own
paper. Sweep node budgets **from H19's real measured curve**, not from the
paper's own 1024/2048 tuned for different hardware.

**Correction to the existing record:** H16 (`afterimage/experiments.py`)
cites SpecExec, but H16 was speculation-conditioned critical-path
residency plus fixed speculation -- not an implementation of SpecExec's
massive-cache tree -- and it regressed 2.75% in that specific composition.
H16 failing does not mean SpecExec failed in Afterimage. Actual SpecExec
remains untested; H24 is that test.

### H25 -- Posterior Target Tree (planned, first true BeliefSpec mechanism)

> A candidate tree optimized using estimated target-path probabilities
> will outperform an equal-budget tree optimized directly from drafter
> probabilities.

Where OPT-Tree builds its tree from P's own `.70/.15/.10`, BeliefSpec
combines P's distribution, S's distribution, the current disagreement
regime, and historical evidence of what the target actually picked when
P/S looked like this, into a **target posterior** (e.g. `.44/.30/.19`
instead of `.70/.15/.10`), and builds the tree from that. **Control:**
H23's OPT-Tree at equal budget.

### H26 -- ScenarioSpec (planned, DESPOT-style)

> Does belief-sampled target trajectory coverage beat deterministic tree
> optimization?

Rather than one deterministic optimal tree, sample many possible target
futures from the belief model (e.g. 64 scenarios), merge their common
prefixes into one sparse tree, and verify that. Directly analogous to
scenario-based belief-space planning (DESPOT) rather than ordinary
probability-greedy tree expansion:
`max_T E[covered target-path length] subject to C_verify(T) <= B`.

### H27 -- Complementary Scout (planned)

> Does Primary + Scout beat Primary alone, end to end?

Only attempted after H21 proves the scout carries real complementary
information. This is the first rung of the ablation ladder in H27/H28.

### H28 -- Disagreement Critic (planned)

> Does structured P+S+history prediction beat raw confidence or a simple
> companion predictor?

The ablation this document insists on before crediting BeliefSpec with
anything: P only -> P+S raw candidate union -> P+S+simple confidence
fusion -> P+S+structured disagreement critic -> P+S+critic+ScenarioSpec.
That separates the benefit of a second model, of candidate diversity, of
the critic, and of belief-space planning from each other -- claiming the
full stack's win without this ladder would not actually show which part
of it did the work.

### H29 -- ProbeSpec (planned, Afterimage-specific active sensing)

> Can nearly-free branches (per H19's own knee) gather useful target
> information instead of only exploiting the most likely continuation?

If H19 shows, say, 128 nodes at 9.00s and 140 nodes at 9.04s, reserve a
few of those nearly-free slots for diagnostic probes -- branches chosen
because the target's response there would distinguish between competing
disagreement-regime hypotheses, not because they are the most likely
continuation:
`U(v) = E[future target sweeps saved] + beta * I(regime; observation at v) - lambda * marginal_cost(v)`.
Speculative Verification already uses a companion model's information to
predict speculation accuracy, so "information gain" itself is not novel;
the narrower claim is spending otherwise-cheap verification capacity on
intentionally informative off-path branches specifically to improve the
belief state used for *future* trees. No exact match for that mechanism
found in the current search. **Gate (G7):** the future benefit must pay
for the sacrificed candidate slots, measured, not assumed.

### H30 -- Receding-Horizon Belief Planning (planned, MPC)

> Does planning over 2-3 future target sweeps beat maximizing only this
> sweep?

A myopic planner can pick a tree with more expected tokens THIS round but
worse information for the next one; a receding-horizon planner can
trade a little of this round for a better next round. Only worth
attempting once H29 shows information genuinely has measurable future
value (**gate G8**: horizon > 1 beats horizon 1 on held-out end-to-end
time).

### H31 -- Belief-MCTS / POMCP (planned, solver of last resort)

Only after ScenarioSpec, not instead of it. State = belief + current tree
+ system cost; actions = expand candidate X/Y, add a scout branch, add a
probe, stop; simulation samples a target trajectory from the critic/belief
model; reward = tokens advanced x target-sweep cost avoided, minus draft/
verification/planner cost. Progressive widening bounds the enormous
vocabulary. **Gate (G9):** if ScenarioSpec already captures most of the
gain for a fraction of the planning overhead, MCTS is not worth its own
complexity -- complexity is not the goal here.

### H32 -- Residual Scout Training (planned, Phase II: new models)

> Can a scout be trained specifically to cover what the Primary misses?

`Utility(S) = P(target in S, target NOT in P) - lambda * Cost(S) - rho * Redundancy(P, S)` --
boosting/ensemble residual-fitting philosophy applied to speculative
candidate economics: rewarded for finding what Primary missed, penalized
for redundantly re-telling Primary's own story. Multi-drafter cooperative
methods already exist in the literature (MetaSD dynamically allocates
among heterogeneous drafters), so this is a component, not the central
novelty claim.

### H33 -- Neural Temporal Critic (planned, Phase II)

> Can a tiny learned critic beat the simple HMM?

Not a second full language model -- a small embedding of P/S distribution
summaries, P/S disagreement, verification history, and system-cost state,
fed through a tiny temporal Transformer or GRU, outputting the same
structured posterior H22's HMM targets (rank posterior, coverage
distribution, regime posterior, branch survival, expected cache-walk
length). Dramatically less capacity than reproducing the target's full
vocabulary distribution, because it is a critic of the proposal system,
not a second generator.

### H34 -- Meta-Controller (planned, only if warranted)

> Should the runtime switch among Linear/SpecExec/BeliefSpec online?

BanditSpec (found in the literature review) adaptively selects
speculative-decoding hyperparameters via multi-armed bandits and
approaches its own oracle in its experiments. **Do not implement this
until the gate says to.** H0 found only a 2.56% semantic-only oracle gap
across the (small) action space it tested, and H3's contextual controller
then captured essentially none of it. **Gate (G10):** recompute the
per-request-vs-global oracle gap after H20-H33 exist, across
`{linear, OPT-Tree, SpecExec, BeliefSpec}`. Above 10-15%, resurrect
meta-selection. Still 2-3%, do not bother -- a larger action space does
not change what H0/H3 already found about how much a per-request
controller can capture.

## Hard research gates

Written down now specifically so months are not spent chasing an
interesting-but-useless mechanism -- the same discipline that already
correctly killed hazard-cost stopping (H2) and the neural-utility policy
(H11) in favor of measured fixed `k=8`.

Numbers below are pre-registered NOW, before any GPU campaign for these
hypotheses has run, specifically so they cannot be adjusted after seeing
results -- a gate a paper claim later contradicts is a real, reportable
finding; a gate quietly loosened to fit the result is not evidence of
anything. Where a specific number is genuinely a judgment call (not
derived from first principles), that is said explicitly rather than
dressed up as more principled than it is.

| Gate | Continue only if... |
|---|---|
| G1 Candidate capacity (H19) | `N_free` (the largest candidate count within 10% of the N=1 sweep cost, `scripts/run_h19_candidate_sweep.py`'s `find_knee`) is >= 32, AND stable within 2x across every measured prompt length. A looser overhead threshold (e.g. 25%) passes almost trivially for a streaming-dominated engine and is not used here for exactly that reason. |
| G2 Branching (H20) | tree beats equal-budget chain end to end |
| G3 Scout (H21/H27) | at the project's default equal-total-budget=16 sweep, `equal_budget_union_gain` >= +0.03 (3 percentage points) at some tested Scout-slot allocation, AND `conditional_rescue_recall` at k=8 >= 0.20, AND `depth_probabilities[2]` >= 0.30 at k=8 -- which is `P(sustained depth >= 2 \| Primary missed AND Scout covered)`, conditioned on a rescue having happened, NOT on the miss alone (the miss-alone version is this times `conditional_rescue_recall`, and reading it as the former overstates it by exactly that factor). All three, not any one alone -- a Scout that only wins on one of these has not shown real, usable headroom. |
| G4a Temporal persistence (H22a) | grouped-CV paired NLL improvement of B5 (HMM) over B1 (memoryless) has a >= 5% relative improvement AND its bootstrap 95% CI excludes zero (`scripts/run_h22_disagreement_hmm.py`'s own `G4a_temporal_persistence` gate, both conditions required). The CI must be a **cluster bootstrap over held-out trajectories**, one paired difference per trajectory -- not over CV folds; see the note below. |
| G4b Online predictability (H22b) | grouped-CV paired NLL improvement of B4 (cheap online features) over B1 has a >= 3% relative improvement (a lower bar than G4a's, since B4 is working with strictly less information than B5) AND its bootstrap 95% CI excludes zero, same trajectory-level clustering. **G4a passing does NOT authorize proceeding past this gate** -- H25 onward needs online-usable prediction specifically, not just proof that latent state exists. |
| G5 Posterior tree (H25) | beats OPT-Tree/SpecExec at matched target cost |
| G6 Critic (H28) | calibrated, and costs only a small fraction of the savings it produces |
| G7 Probes (H29) | future benefit pays for the sacrificed candidate slots |
| G8 MPC (H30) | horizon > 1 beats horizon 1 on held-out end-to-end time |
| G9 MCTS (H31) | exceeds simple ScenarioSpec enough to justify the planner cost |
| G10 Meta-controller (H34) | oracle gap across the full strategy set exceeds 10-15% |

G1/G3/G4a/G4b's specific numbers (32, 3pp, 20%, 30%, 5%, 3%) are this
project's own judgment calls, not derived from an external standard --
recorded here so a future revision that changes them has to say so
explicitly, in a diff, rather than silently.

**Why G4a/G4b's bootstrap clusters over trajectories, not CV folds.** An
earlier revision bootstrapped the five per-fold NLL differences directly.
That is wrong twice over: with 5 folds any two training splits share
about three quarters of their data, so fold-to-fold spread understates
real uncertainty; and five numbers whose signs happen to agree produce an
interval excluding zero almost regardless of effect size. This was not
theoretical -- on the same synthetic fixture, the per-fold version
reported a ~2% effect as significant (CI [0.0137, 0.0354], excludes
zero), while the trajectory-level cluster bootstrap on identical data
reported CI [-0.0055, 0.0639], which does not. Trajectories are the right
unit because positions within one are correlated, which is exactly what
H22 hypothesizes; each contributes one unweighted observation so a few
long trajectories cannot dominate. The two-part gate (significance AND
effect size) is what caught the difference in both directions, and is the
reason both halves are required rather than either alone.

## Measurement schema

Every speculative run in this research line should emit:

```
spec_strategy
draft_model

draft_tokens_generated
draft_seconds

candidate_nodes
candidate_nodes_by_source     # P vs S contribution
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

approx_js_divergence           # sparse top-k approximation, H21/H22
conditional_rescue_recall     # H21
target_path_coverage          # planner quality

critic_time
critic_brier_ece_nll          # calibration
regime_prediction_accuracy    # validates the belief state (H22)
probe_information_gain        # validates ProbeSpec (H29)

bytes_read
decode_seconds
io_seconds
compute_seconds

seconds_per_token
tokens_per_second
peak_vram
```

The fields above are the eventual, full end-to-end schema for a real
speculative run once H20+ exist. What is actually implemented and
producing real numbers today is narrower and lives in each script's own
`SpeculationTrace v2` output (`schema_version: 2` on every result file):

H19 (`scripts/run_h19_candidate_sweep.py`, `measure_candidate_sweep_latency`
+ `build_amortization_curve`): `candidate_positions`,
`verification_sweep_seconds`, `io_seconds`/`decode_seconds`/
`compute_seconds` per cell; per-case curves keyed by `case_id`, each point
carrying `median_seconds`/`min_seconds`/`max_seconds`/`samples`,
`throughput_candidates_per_second`, `relative_to_n1` (absent when N=1
was not measured), and `marginal_cost_seconds_vs_half` (`None` when N/2
was not measured); `candidate_parallelism_knee_by_case` from `find_knee`.

H21 (three-stage pipeline -- see the H21 section above -- final analysis
from `scripts/run_h21_combine_sources.py`, calling
`compute_oracle_coverage_stats`, `compute_equal_budget_metrics`,
`compute_sustained_rescue_depth`): per-`k` `primary_coverage`/
`scout_coverage`/`union_coverage`/`conditional_rescue_recall`,
`jaccard_overlap_at_collection_k`; per-Scout-slot-count `points` with
`equal_budget_union_gain`, `marginal_scout_coverage_per_slot`,
`unique_scout_hit_rate`, `primary_scout_overlap`; `rescue_opportunities`,
`mean_depth`, `median_depth`, `depth_probabilities` (per tested depth).

H22a/H22b (`scripts/run_h22_disagreement_hmm.py`): per-fold
`b0_constant_frequency_nll` .. `b5_hmm_nll` (position-pooled, reported for
auditability only); `cross_validation.bootstrap_unit` (`"trajectory"`),
`scored_trajectories`, and `b1_reference_nll_per_trajectory`;
`gates.G4a_temporal_persistence` and `gates.G4b_online_predictability`,
each with a `paired_nll_improvement` block (`mean`/`ci_low`/`ci_high`/
`excludes_zero`/`n_paired_observations` from
`bootstrap_nll_difference_ci`, where `n_paired_observations` is the
trajectory count, not the fold count), a `threshold_relative_improvement`,
the measured `relative_improvement`, and a combined `passes` boolean;
`hmm_vs_cheap_features` (B4 vs B5, interpretive only -- not a gate);
`state_selection.underpowered` / `cross_validation.underpowered` flags
from `minimum_positions_for_hmm`.

**The metric this whole research line is really chasing:**

```
bytes/token = compressed target bytes per sweep / tokens advanced per sweep
```

Compression attacks the numerator once (29.5 GB -> ~20.3 GB for
Qwen3-14B). Speculation attacks how often that 20.3 GB has to be read at
all. If SpecExec-class speculation reaches 8 committed tokens per target
sweep, that is ~2.5 GB of streamed target weights per generated token
instead of 20.3 GB/token -- a larger lever than any further improvement to
the Huffman codec itself, and the clean integration point between the
compression work and the speculation research.

## Implementation order

1. **H19 first.** Done.
2. **A generic `SpecTree` representation and `TargetTreeVerifier`.** Real
   correctness-sensitive engine work needing GPU verification against a
   reference forward pass -- not written blind.
3. **Exhaustive top-B and a simple SpecInfer-style tree** (H20/part of
   H23). Validates that branching helps at all and establishes controls.
4. **OPT-Tree** (H23). Clean, applies directly to the existing drafter.
5. **Real SpecExec** (H24), budgets from H19's measured curve.
6. **Offline multi-source oracle experiment** (H21). Done. Evaluate
   candidate scouts without integrating any of them into the runtime yet.
7. **Offline HMM disagreement analysis** (H22). Done. Establish or kill
   persistent latent regimes before building anything that assumes them.
8. **Posterior Target Tree** (H25) -- the first true BeliefSpec mechanism.
9. **ScenarioSpec** (H26) -- DESPOT-style posterior target trajectories.
10. **Integrate the best scout** (H27), only if H21's oracle experiment
    passed.
11. **Structured Disagreement Critic** (H28) -- simple first, not neural.
12. **ProbeSpec** (H29).
13. **2-3 sweep MPC** (H30).
14. **POMCP/MCTS** (H31), only if the offline oracle shows remaining
    planning headroom past ScenarioSpec.
15. **Train Residual Scout** (H32).
16. **Train tiny Temporal Disagreement Critic** (H33).
17. **Full BeliefSpec-Triad benchmark** against fixed k=8, OPT-Tree,
    Sequoia-style optimization, SpecInfer, and SpecExec.
18. **Cross-model/cross-family replication.**
19. **Meta-controller** (H34), only if G10's oracle-gap recheck warrants
    it.

Do not train anything (H32/H33) before step 8 the same way this project
never trains anything before knowing whether the mechanism it would serve
has real headroom: if a new drafter or critic appears to help before H21/
H22/H25/H26 establish that headroom exists, there is no way to tell
whether it helped because of better token accuracy, better tree coverage,
different confidence calibration, or one tree architecture happening to
suit it.

## Novelty ranking

| Idea | Confidence |
|---|---|
| Two drafters | Low |
| Heterogeneous drafters | Low |
| GAN-trained drafter | Low |
| Companion acceptance critic | Low |
| Three-model qualifier | Low |
| Different-architecture drafter | Low |
| Cost-aware adaptive tree | Low (CAST, ICLR 2026, already claims the general concept) |
| Structured target-disagreement posterior | Medium-high |
| Primary + rescue-specific Scout + structured Critic, as a role-separated composition | Medium |
| Posterior Target Tree | Medium-high |
| DESPOT-like target-scenario verification tree | High |
| Information-probe branches spending spare offload verification capacity | Medium-high |
| Multi-sweep belief-space dual control / MPC | High |
| Full BeliefSpec-Triad + Afterimage I/O-cost model | Strongest overall candidate |

None of these should be stated as established novelty in a patent or
paper yet -- this is where the literature search currently leaves the
clearest unoccupied space, not a claim that the space is confirmed empty.
See [SPECULATION_NOVELTY_LEDGER.md](SPECULATION_NOVELTY_LEDGER.md) for the
closest-known-prior-art detail behind each row above, with search dates
and corpus, so a future revision of this table has a paper trail rather
than a re-derivation from memory.

## H19 detail

**Run this before building any tree.**

> Because Afterimage target inference is dominated by moving/decompressing
> streamed weights, increasing the number of candidate positions evaluated
> in one target sweep will initially increase target-sweep latency much
> more slowly than candidate count.

This does not generate real speculative trees. It makes the target
process different numbers of already-known candidate positions -- their
token *values* do not matter, only their count -- and measures one
verification sweep's wall-clock cost as a function of that count, from
1 up through 1024. If the curve stays flat well past `k=8`, that is
exactly the regime SpecExec's authors target for parameter-offloaded
models, where hundreds or thousands of candidate positions can be
processed for close to the cost of one because parameter movement
dominates. The knee in that curve (`N_free` / `spec_parallel_knee`) sets
every subsequent tree budget in this document, rather than copying a
paper's number tuned for different hardware.

**Implementation.** `StreamingLosslessModel.measure_candidate_sweep_latency`
reuses the real `forward_logits()` verification path -- the same call
`generate_speculative`'s own target sweep makes. Driven by
`scripts/run_h19_candidate_sweep.py`, with repeats, cooldown, and a
dirty-tree gate matching this project's existing benchmark discipline,
writing a `candidate_amortization_curve` with each point's ratio to the
N=1 baseline -- the number that actually answers H19. **Gate (G1):** find
the knee, not a fixed pass/fail threshold; this is a diagnostic
measurement, the same role H0 plays for the contextual-control research
line.

## Sources

SpecInfer, Sequoia, OPT-Tree (TACL), SpecExec, CAST (ICLR 2026),
BanditSpec, DiffuSpec, D²SD, Universal Assisted Decoding, MetaSD,
Speculative Verification, KOALA, the August 11 2026 speculative-decoding
survey -- see `docs/LITERATURE.md` Part III for the fuller survey this
document's framing builds on.
