# Phase 0 Results — Real Measurement, 2026-08-17

**This is the decision-gate measurement IMPLEMENTATION_PLAN.md #2 and
EXECUTION_PLAN.md Stage C called for.** Run against a real transformer, on
real GPU hardware, not the toy model used everywhere else in this repo.

**Verdict: NO-GO.** Full data and reasoning below.

---

## 1. What was actually run

| | |
|---|---|
| Model | Qwen/Qwen2.5-1.5B-Instruct, fp16, on the RTX 3080 Laptop (8GB) via WSL2/CUDA |
| Layers probed | 6 MLP `down_proj` layers spread across depth (0, 5, 11, 16, 22, 27 of 28) |
| `d_in` per layer | 8960 |
| Workloads | 4 real hand-written text sets: focused code (20 ex.), multi-turn chat (12 ex.), long-form prose (6 ex.), adversarial topic-switching (38 ex., all three categories interleaved) |
| Ranks tested | 4, 8, 16, 32, 64, 128, 256 for the rank-vs-error curves; 8, 32, 128 for closed-loop |
| Measurement | `probe/spectra.py::layer_rank_report` (variance curve + functional-error curve + entropy effective rank, one shared SVD per layer) and `probe/closed_loop.py` (real forward-pass replay with truncation installed simultaneously in all 6 target layers) |

**Scope honestly stated:** this is a 1.5B proxy model, not the 27B target;
only MLP `down_proj` (not attention `q/k/v/o_proj`); ranks tested only up to
256 out of 8960 (2.9%), not up to the 0.5·d "confirmed barrier" threshold
IMPLEMENTATION_PLAN.md #2.4 specifies; and each workload has a handful of
examples, not a corpus. None of these limits change the qualitative
conclusion below — the trend is unambiguous and would need roughly an order
of magnitude more rank to even approach viability — but they are real limits
on this specific run, not on the method in principle.

---

## 2. Effective rank, by workload (entropy-based, out of `d_in=8960`)

| Workload | mean | min | max | % of `d_in` |
|---|---|---|---|---|
| long_form_prose | 228.0 | 182.2 | 251.4 | **2.5%** |
| multi_turn_chat | 367.2 | 296.3 | 418.7 | 4.1% |
| focused_code | 429.2 | 277.2 | 577.8 | 4.8% |
| adversarial_topic_switch | 990.6 | 752.2 | 1200.9 | **11.1%** |

The ordering matches the hypothesis's own prediction — the narrowest workload
(long-form prose, one topic sustained) has the lowest effective rank; the
deliberately topic-switching workload has the highest, at over 4x prose's
rank. **The mechanism the hypothesis describes is real and measurable.**

The problem is the absolute scale: even the *most favorable* workload uses
2.5% of the layer's width, not the sub-1% (or the outright single-digit
absolute rank) a practical cache needs. HYPOTHESIS.md's success threshold was
`r ≤ 256` capturing the layer; 256 is 2.9% of 8960, roughly the same order as
what real usage already occupies at rank ~230 — there is very little slack
between "how much rank a real session uses" and "how much rank the cache
budget can afford."

---

## 3. Functional error vs. rank — the number that actually matters

HYPOTHESIS.md #3.1 predicted variance-captured and functional-error would
diverge (the rogue-dimension effect). They do, sharply, confirming the
mechanism concern was correctly identified:

**`focused_code`, at rank=256 (2.9% of `d_in`), per layer:**

| Layer | variance captured | functional error |
|---|---|---|
| layers.0 | 97.8% | 24.7% |
| layers.5 | 99.4% | 44.0% |
| layers.11 | 88.3% | 41.9% |
| layers.16 | 82.9% | 44.9% |
| layers.22 | 89.0% | 36.0% |
| layers.27 | 98.6% | 27.3% |

A variance-based criterion would report this layer as "97-99% captured" —
looks nearly solved. The functional criterion (the one that actually predicts
output error) says 25-45% of the output is still wrong. **This is exactly the
rogue-dimension gap the hypothesis worried about, and it is large: roughly
2-3x, not a rounding difference.**

The success threshold was functional error **< 0.1% at rank ≤ 256**. The
measured functional error at rank 256 is **25-45%**, roughly 250-450x too
high. This is not a marginal miss.

---

## 4. Closed-loop end-to-end error — the real number, not a proxy

This is the measurement IMPLEMENTATION_PLAN.md #2.2 built the whole
closed-loop harness to get right: install rank-`r` truncation into all 6
target layers simultaneously, run one real forward pass, and compare the
final output against the exact model.

| Workload | rank=8 | rank=32 | rank=128 |
|---|---|---|---|
| narrow (focused_code) | 96.5% | 73.1% | 70.5% |
| adversarial (topic-switch) | 75.9% | 68.7% | 59.7% |

Relative error decreases with rank in both cases, as expected — the basic
mechanism works. But even at rank=128, across only 6 truncated layers (not
the whole ~28-layer, ~200-linear-layer model), the model's final output
differs from the exact model by **60-70%**. This is with only a *fraction*
of the model's linear layers replaced by a cache; a real deployment would
need this number near zero across *every* linear layer simultaneously.

**One further finding worth flagging plainly:** the narrow workload's
closed-loop error is *higher* than the adversarial workload's at every rank
tested (96.5% vs 75.9% at rank 8; 73.1% vs 68.7% at rank 32; 70.5% vs 59.7%
at rank 128) — the opposite of what the hypothesis hoped for. The sample
sizes here are small (10 calibration / 10 eval sequences for the narrow set,
19/19 for adversarial), so this specific ordering shouldn't be treated as a
settled fact on its own — but it is consistent across all three ranks tested,
not a one-off, and it does not offer any support for the "narrow sessions
compress better end-to-end" hypothesis. If anything it's a small negative
signal on top of an already-decisive negative result.

---

## 5. Verdict against the stated gate (IMPLEMENTATION_PLAN.md #2.4)

| Outcome | Threshold | Measured | Met? |
|---|---|---|---|
| Go | functional error < 1e-3 at r ≤ 256 | 25-45% at r=256 | **No, by ~2-3 orders of magnitude** |
| Marginal | holds at r ≥ 1024 | error still >25% at r=256; would need far more than 1024 given the observed slope | **No** |
| No-go | no useful rank below 0.5·d | 0.5·d = 4480; error remains catastrophic (60-96% end-to-end) at r=128, 3.5% of that threshold | **Yes — this is the outcome** |

**This is a NO-GO on real hardware, against a real (if small) model,
measured directly rather than inferred from corpus-level literature.**

---

## 6. What this changes

Per HYPOTHESIS.md §6 and EXECUTION_PLAN.md §6.1, stated in advance of running
this measurement: **a failed gate means stop, not push through.** The
Afterimage subspace cache is not a viable research direction on the evidence
gathered so far, and building the multi-week runtime (EXECUTION_PLAN.md
Stage D: real model integration, CUDA streaming, tree speculation) on top of
it is not justified by this result.

**What still stands, and is real:**

- Every mechanism in `afterimage/runtime/` (basis maintenance, the JL gate,
  batched fetch-once verification, the exact speculative-sampling guarantee)
  is implemented correctly and passes 67 tests, several of which caught real
  bugs during development. That code is sound; the workload it was built for
  does not have enough exploitable structure on this evidence.
- The measurement methodology itself — closed-loop replay, the
  variance/functional gap, masked activation capture, per-workload
  reporting — worked correctly on a real model and is reusable for any
  future rank-based compression idea, independent of this specific
  hypothesis's fate.
- The honest fallback was scoped in advance: ship residency + tree
  speculation (SubSpec/SpecExec-class methods) as the actual 27B-on-8GB
  product. That is unaffected by this result — it was never dependent on the
  cache working.

**What should not happen next:** scaling this exact measurement up to the
27B target model to "double check." The gap between the measured functional
error (25-45% at 2.9% of layer width) and the required threshold (<0.1%) is
too large for a bigger model to plausibly close on its own — nothing in the
literature or in this measurement suggests larger models have *proportionally*
lower functional rank, and the adversarial-workload result (11% of `d_in`,
the highest measured) points the other direction if anything.

---

## 7. Raw data

Full JSON: [phase0_real_results.json](phase0_real_results.json)
