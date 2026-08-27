# Speculation Novelty Ledger

Companion to [SPECULATION_TREE_RESEARCH.md](SPECULATION_TREE_RESEARCH.md)'s
[Novelty ranking](SPECULATION_TREE_RESEARCH.md#novelty-ranking) table. That
table ranks confidence; this file records the actual search behind each
ranking -- what was searched, when, against what corpus, and the closest
prior art found -- so a novelty claim in a future paper or patent has a
paper trail instead of a vibe.

**None of the entries below establish novelty.** A "candidate/unproven"
status here means: the searches performed so far did not surface prior art
that fully subsumes the idea, not that a dedicated, professional novelty
or patentability search has been done. An obscure patent, an unpublished
manuscript, a workshop paper outside the searched corpus, or a differently
named formulation of the same idea could still exist. Treat every row as
"still worth a real search before this is asserted anywhere that matters,"
not as a green light.

## How to read a row

- **Idea**: the specific mechanism or composition being tracked.
- **Closest known prior art**: the nearest thing found so far, with enough
  detail to tell it apart from this project's version.
- **Distinguishing claim**: what, if anything, this project's version does
  differently from that prior art.
- **Novelty status**: `candidate` (searched, nothing fully subsuming found
  yet) / `unproven` (not yet searched with real rigor, or search was
  shallow) / `low` (prior art substantially covers this already, kept in
  the ledger so the reasoning is not silently lost).
- **Last searched**: date of the most recent real search pass, and its
  rough corpus (so "last searched 2026-08" against only arXiv preprints is
  visibly weaker evidence than a pass that also covered patents).

## Ledger

### Posterior Target Tree

- **Closest known prior art**: OPT-Tree (dynamic tree construction from
  draft-model confidence) and SpecExec (massively parallel candidate
  verification under parameter-offload latency dominance) each address one
  half of this -- tree shape from one model's confidence, or verification
  count under offload cost -- but neither combines a target-side posterior
  belief over *multiple heterogeneous proposal sources'* disagreement with
  an explicit tree-shape decision driven by that belief.
- **Distinguishing claim**: the tree shape is chosen from an explicit,
  fitted posterior over where the target is likely to disagree with
  Primary, conditioned on more than one proposal source, rather than from
  a single drafter's own confidence or a fixed branching heuristic.
- **Novelty status**: candidate.
- **Last searched**: 2026-08-11, against the same 1,652-candidate /
  304-consolidated speculative-decoding survey corpus referenced in
  [Novelty framing](SPECULATION_TREE_RESEARCH.md#novelty-framing).

### ScenarioSpec (DESPOT-style target-scenario verification tree)

- **Closest known prior art**: DESPOT (Determinized Sparse Partially
  Observable Tree) itself, from the POMDP planning literature -- the
  scenario-sampling and tree-determinization mechanism this idea borrows
  directly and by name.
- **Distinguishing claim**: applying DESPOT's determinized-scenario
  construction to target-verification tree building specifically, where
  "scenarios" are sampled continuations under a fitted disagreement
  posterior and the cost model is real measured offload bytes/token,
  not a generic POMDP reward. The survey corpus's speculative-decoding
  literature shows no POMDP/DESPOT-framed prior art in this specific
  application; the general DESPOT algorithm itself is well established
  and not being claimed as new.
- **Novelty status**: candidate (for the application), not novel (for the
  underlying DESPOT mechanism, which is prior art by design and cited
  as such, not disguised).
- **Last searched**: 2026-08-11, speculative-decoding survey corpus only --
  the POMDP/DESPOT side of this has not had its own dedicated planning-
  literature search pass yet, only the speculative-decoding side.

### ProbeSpec (active-sensing candidate branches)

- **Closest known prior art**: active learning / information-gain
  acquisition functions in general, and speculative decoding work on
  adaptive draft-length selection (e.g. AdaEDL-style hazard-cost policies,
  already implemented in this project's own `spec_k_policy`). Neither
  spends a scarce, already-budgeted *candidate verification slot*
  specifically to gather information about future target behavior rather
  than to propose a real continuation.
- **Distinguishing claim**: a probe node is a first-class tree member
  (see `SpecNode.probe` in
  [SpecTree structure](SPECULATION_TREE_RESEARCH.md#spectree-structure-implemented-ahead-of-h20))
  competing for the same verification budget as real candidates, evaluated
  by whether its information gain pays for the candidate slot it did not
  spend on a real proposal.
- **Novelty status**: unproven -- searched only incidentally as part of
  the broader 2026-08-11 pass, not with a dedicated active-sensing /
  information-gain query set.
- **Last searched**: 2026-08-11 (shallow, incidental).

### Belief-space MPC over multiple target sweeps

- **Closest known prior art**: receding-horizon / model-predictive control
  is standard outside speculative decoding; within speculative decoding,
  every found strategy (SpecExec, OPT-Tree, Sequoia, CAST) plans a single
  sweep's tree and re-plans greedily sweep-to-sweep, rather than
  explicitly optimizing a multi-sweep horizon under a belief state that
  persists across sweeps.
- **Distinguishing claim**: the horizon spans multiple target sweeps under
  a belief state carried forward between them (H22's persistence question
  is the empirical precondition for this even being worth building --
  see G4a in
  [Hard research gates](SPECULATION_TREE_RESEARCH.md#hard-research-gates)),
  not a per-sweep greedy tree choice.
- **Novelty status**: candidate.
- **Last searched**: 2026-08-11, speculative-decoding survey corpus;
  general MPC/POMDP literature not separately searched.

### Full BeliefSpec-Triad + Afterimage I/O-cost model composition

- **Closest known prior art**: none found that combines role-separated
  proposal sources (Primary/Scout), a fitted disagreement belief state,
  DESPOT-style scenario verification, and a real measured
  bytes-per-token offload cost model into one system. Each component
  piece individually has prior art (see the other rows in this ledger and
  [Novelty ranking](SPECULATION_TREE_RESEARCH.md#novelty-ranking)); the
  composition, and specifically grounding it in Afterimage's own measured
  weight-streaming cost rather than a generic FLOPs or latency proxy, is
  what the search has not found elsewhere.
- **Distinguishing claim**: this is the composition claim, not a single
  mechanism -- its strength depends on every gate in
  [Hard research gates](SPECULATION_TREE_RESEARCH.md#hard-research-gates)
  actually passing on real hardware, not on the novelty search alone.
- **Novelty status**: candidate, and explicitly the project's own
  strongest overall candidate per the ranking table -- also the claim
  most in need of a real, non-shallow search before it is asserted
  anywhere that matters.
- **Last searched**: 2026-08-11, speculative-decoding survey corpus only.

## Explicitly low-novelty (kept for the record, not because they're wrong)

These were checked and found to already have direct prior art. Recorded so
the reasoning is not silently lost, and so nobody re-discovers the same
prior art from scratch later.

- **Two drafters / heterogeneous drafters / GAN-trained drafter /
  companion acceptance critic / three-model qualifier / different-
  architecture drafter** -- each individually well covered by existing
  speculative-decoding literature. Low novelty as standalone ideas; see
  "Primary + rescue-specific Scout + structured Critic, as a role-
  separated composition" for where this project's actual contribution is
  argued to live instead.
- **Cost-aware adaptive tree** -- CAST (ICLR 2026) already claims the
  general concept of a cost-aware adaptive speculative tree. Low novelty;
  any claim in this document's line of work needs to distinguish itself
  from CAST specifically, not merely from older fixed-tree baselines.
- **Last searched**: 2026-08-11.
