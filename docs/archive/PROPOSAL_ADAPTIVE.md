# Proposal — Making the engine adaptive

**In one line: stop asking "how fast can I read the model?" and start asking
"how many words can I get out of each read?"**

Supersedes the adaptive half of [PROPOSAL.md](PROPOSAL.md). Evidence base:
[LITERATURE.md](LITERATURE.md) Part III. Implementation and real-hardware
measurements: [ADAPTIVE_TEST_PLAN.md](ADAPTIVE_TEST_PLAN.md) /
[RESULTS_LOG.md](RESULTS_LOG.md).

> **STATUS (2026-08-19): Implemented and measured on Qwen3-14B.** Mixed,
> honest result, not a clean sweep:
> - **Mechanism B (bandit/adaptive draft length) — infra built, NOT justified
>   yet.** Neither GammaTune nor threshold beat a tuned constant k at the
>   sweep counts a single short generation provides; ships opt-in,
>   defaults to `"fixed"`.
> - **Mechanism A (self-speculation) — FALSIFIED as tested.** Zero measured
>   draft acceptance at two depths (4 and 8 of 40 layers) on this untrained
>   checkpoint. `draft_mode="self"` is real, bit-exact, tested code — not
>   recommended for use without a trained early-exit head, which is out of
>   scope for this project.
> - **Mechanism C (draft-layer-aware VRAM planning) — CONFIRMED
>   mechanically** (pinning gives 1.13-1.57x over unpinned self-drafting,
>   scaling with depth exactly as hypothesized) **but has no working draft
>   mechanism to pair with today**, since A doesn't produce useful drafts.
> - **What's actually usable today: mechanism B's non-adaptive form** — a
>   small resident draft model through the new `generate_adaptive` entry
>   point, now with a real temperature=0 correctness proof (see
>   `verify.temperature_probs`). Confirmed 3.4x over this engine's OWN
>   greedy mode at matched (in fact slightly lower) peak VRAM.
> - **Against AirLLM specifically, stated carefully:** this arm does not run
>   at AirLLM's actual ~1.6 GB footprint at all (measured infeasible — the
>   draft model's own ~1.3 GB plus the target's own ~1.7 GB minimum floor is
>   structurally above it). At 4-6 GB it reaches 6.6-6.9x over this engine's
>   own greedy mode, i.e. roughly 9x over AirLLM's wall time -- but at 2.4-
>   3.7x AirLLM's memory, not matched VRAM. Greedy-vs-AirLLM (1.3-1.76x) is
>   the number that IS matched-VRAM. Full numbers and the reasoning behind
>   each verdict: RESULTS_LOG.md's "Adaptive speculation" / "T4" sections.

---

## 1. What's dumb about the engine today

The model is 20 GB compressed. It doesn't fit in an 8 GB GPU. So for **every
single word**, we read all 20 GB off disk, one layer at a time, and throw it
away.

That means:

- **Every word costs the same.** "The capital of France is ___" costs exactly as
  much as a genuinely hard word. We pay full price for "Paris."
- **One trip = one word.** ~18 GB moved, ~15 seconds, one word out.
- **Nothing is learned.** Same fixed plan every token, every prompt, forever.

It's like driving to the library and back for every word of a sentence —
including "the."

**This is the actual ceiling.** Not the compression (that's at 96% of its
mathematical limit), not the disk. The ceiling is *one word per trip*.

---

## 2. The reframe

| | words per trip | seconds per word |
|---|---|---|
| Today, plain | 1 | 14–20 |
| Today, with a small draft model (measured) | ~3 | 2.7–4.5 |
| **Target** | **6–10** | **~1.5–3** |

Everything below exists to raise the first column. Reading faster is a dead end;
*harvesting more per read* is not.

---

## 3. Three mechanisms

### A. Let the model draft for itself ("self-speculation")

> **STATUS: TESTED AND FALSIFIED (2026-08-19), as implemented on this
> untrained checkpoint.** `draft_self_logits` is bit-exact and works exactly
> as designed mechanically — the measured problem is upstream of the
> mechanism: an untrained Qwen3-14B's own early layers (tested at 4/40 and
> 8/40) don't produce logits close enough to the full model's to get
> accepted. Zero measured acceptance at both depths. LayerSkip's published
> gains require training the model FOR early exit; nothing here was
> trained. See RESULTS_LOG.md.

**Now:** a separate small model (Qwen3-0.6B) guesses the next few words; the big
model checks them all in one trip. Works — that's the ~3 words/trip above — but
the little model permanently occupies ~1.2 GB of the VRAM we're desperate for.

**Change:** guess using the **big model's own first ~8 layers** instead of a
second model. Then verify with all 40 layers. Same accept/reject math, so the
output is still exactly what the full model would have produced.

**Why it should be better:** the draft is literally made *out of* the target
model, so the two agree more often — and higher agreement means more accepted
words per trip. Meta's [LayerSkip](https://arxiv.org/abs/2404.16710) (ACL 2024)
reports up to 2.16x doing exactly this, and notes it uses less memory than
draft-model speculation.

**Honest tension:** those 8 layers get run *k times* per trip. If they're not
already in VRAM, we'd re-stream them k times and lose badly. So this only pays if
the draft layers are pinned — which is mechanism C.

### B. Learn how much to guess, and how deep to guess from

> **STATUS: TESTED, NOT JUSTIFIED YET (2026-08-19).** Neither GammaTune nor
> a threshold policy beat a well-tuned constant k across a 16-token run
> (~7 sweeps) — the pre-stated kill criterion. The likely cause is also
> pre-stated (RESULTS_LOG.md, PROPOSAL_ADAPTIVE §6): too few sweeps per
> generation for online adaptation to pay off. Cross-run persistence
> (`spec_policy_state`) is implemented and unit-tested but not yet verified
> live. Ships opt-in; `spec_k_policy="fixed"` is the current recommendation.

Two dials, both currently fixed constants:

- **How many words to guess ahead (k).**
- **How many layers to guess with (the exit depth).**

The right values change constantly. Easy stretch of text → guess 10 words from 6
layers. Hard stretch → guess 3 words from 20 layers. **A fixed setting is wrong
most of the time.**

**Change:** a small learner (a *bandit* — the simple end of reinforcement
learning) watches how many guesses got accepted and adjusts both dials live.

**Why this is safe here, and this is the important part:** these dials
**cannot change the answer.** The accept/reject step guarantees the exact same
output distribution for *any* k and *any* exit depth. So the learner is
optimizing pure speed with no accuracy to trade away — a bad guess costs a slow
word, never a wrong one. That means it can explore aggressively. Most RL-for-
efficiency work (pruning, quantization) has to be cautious precisely because
exploration risks quality. **We don't have that problem.**

Evidence it works: [GammaTune](https://arxiv.org/pdf/2504.00030) +15–16% from
adapting k alone; [BanditSpec](https://arxiv.org/pdf/2505.15141) (ICML'25) gets
near-oracle results with a plain bandit, no training;
[SpecDec++](https://arxiv.org/pdf/2405.19715) proves the *optimal* policy is a
simple threshold rule — so this needs a good signal and one learned number, not a
neural network.

Our own data says the headroom is real: across one temperature sweep, acceptance
swung **23–41%** and words-per-trip **2.67–4.00**. One fixed setting can't cover
that range.

### C. Re-plan memory around the fact that early layers are now hot

> **STATUS: MECHANICALLY CONFIRMED (2026-08-19), currently without a working
> consumer.** Pinning draft layers measurably beats not pinning them — 1.13x
> at 4/40 layers, 1.57x at 8/40, the gap widening with depth exactly as
> hypothesized (more unpinned layers = more re-streamed bytes/sweep). This
> mechanism does exactly what it was built to do. What it does NOT yet have
> is a draft mechanism worth pinning FOR, since mechanism A doesn't produce
> useful drafts on this checkpoint. Kept as tested, correct, opt-in
> infrastructure (`EngineConfig.pin_draft_layers`) — ready the moment A (or
> a future trained early-exit head) works.

**This is the non-obvious one, and it's specific to this engine.**

The memory planner's core rule, quoted from `vram_planner.py`:

> *"Every weight in a dense decoder is used exactly once per token, so
> 'frequency of use' cannot rank them — it is identical for all."*

Self-speculation **makes that false.** If we draft with layers 0–7, those layers
run k+1 times per trip while layers 8–39 run once. They stop being equal. They
become, by a wide margin, the most valuable things to keep in VRAM.

**Change:** teach the planner about draft-layer reuse, so it pins the layers that
are now touched repeatedly.

**Why nobody else has this:** LayerSkip and [KnapSpec](https://arxiv.org/pdf/2602.20217)
do adaptive layer selection — but for models that **fully fit in memory**, where
placement isn't a question. Sibyl and the storage-placement line optimize
placement — but with **no speculation**, so every weight really is used once. The
coupling only exists when you're simultaneously memory-starved *and* self-drafting.
That's this engine.

---

## 4. The hypothesis, stated plainly

> **If the model drafts using its own early layers, and a bandit tunes how far
> and how deep to guess, and the memory planner pins the layers that drafting
> now reuses — we get 6–10 words per trip instead of 1–3, and free the 1.2 GB
> the separate draft model was holding.**
>
> **Expected: 2–3x faster than today's speculative mode, 5–10x faster than plain
> streaming — with byte-identical output.**

The three parts are **not independent** — that's the whole claim. Self-drafting
without pinning is *slower* (you re-stream the draft layers). Pinning without
self-drafting is wasted VRAM. The bandit is what finds the operating point,
because the right one depends on the model, the budget, and the text.

---

## 5. How it gets tested (and what would kill it)

| Step | Test | Kill criterion |
|---|---|---|
| 1 | **Coupling check** — grid k × VRAM budget, 12 runs, *no new code* | If the best k doesn't shift with budget, the knobs are independent: tune separately, skip the joint machinery |
| 2 | **Self-draft vs small-draft** at equal VRAM | If acceptance doesn't beat the 23–41% baseline, keep the small model and stop |
| 3 | **Bandit vs best fixed setting** | If a single tuned constant matches it, ship the constant — adaptivity wasn't justified |
| 4 | **Pinned draft layers on/off** | If pinning doesn't pay for its VRAM, self-drafting is a dead end at this model size |

Step 1 runs first on purpose: it costs nothing and its answer decides whether
steps 2–4 can be done independently or must be co-designed.

**Every step is judged by end-to-end wall-clock, never a component benchmark.**
That rule is written in blood: the CPU-decode lever passed its isolated
throughput gate at 1.33 GB/s and was still **0.52x** end to end. A gate that
passes is licence to build, never licence to claim.

---

## 6. Risks, honestly

- **Draft layers may be too big to pin.** A 14B layer is ~0.66 GB; 8 of them is
  ~5 GB. That fits a 6 GB budget, not a 2 GB one. **Self-speculation may simply
  be unaffordable at the smallest budgets** — in which case the small draft model
  (1.2 GB) stays the better deal, and that's a legitimate outcome.
- **Few sweeps per run.** At ~15 s/trip, a 20-word answer is ~6 trips — too few
  for a bandit to converge *within* one run. The learner's state must persist
  across runs, or be a smoothed heuristic rather than something needing many
  samples.
- **Early-exit drafts may be poor without training.** LayerSkip *trains* models
  to exit early. Ours weren't. Untrained early-exit acceptance could be bad
  enough to sink the idea — which is exactly what step 2 measures before
  anything is built on it.

---

## 7. What stays off the table

- **No accuracy tradeoff.** CALM-style adaptive depth reaches 3x by letting the
  answer change within a statistical bound. That is a real technique and it is
  **not** being adopted: everything here keeps output byte-identical.
- **No CPU decode.** Tested, measured worse, removed (RESULTS_LOG.md). The CPU's
  remaining jobs are running the bandit (microseconds, off the critical path) and
  CPU-only fallback.
- **No compression gains.** 1.453x of a 1.51x mathematical ceiling. That well is dry.
