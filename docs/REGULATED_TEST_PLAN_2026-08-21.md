# Hypothesis-aware regulated test plan

Date: 2026-08-21

Status: implemented as a machine-readable L0-L3 protocol registry in
`afterimage/protocols.py`, exposed by `afterimage test-plan` and the experiment
API. This document is the proposed execution order. Existing short result files
remain valid mechanism screens, but they are not retroactively promoted to
confirmatory evidence.

## Why one benchmark matrix is not a scientific test

A common prompt/token matrix measures whether configurations execute under one
wall-clock budget. It does not give every hypothesis the evidence it needs:

- residency is chosen offline and needs disjoint traces, several VRAM budgets,
  real-plan divergence and replay-transfer validation;
- online prefetch control needs burn-in, posterior/depth diagnostics and
  steady-state layer demands rather than first-token averages;
- learned speculation needs enough accepted/rejected positions to calibrate,
  a frozen held-out state and proof that it actually takes different actions;
- artifact/codec hypotheses need hundreds of bitwise tensor round trips before
  GPU latency is worth measuring;
- request-level RL needs many heterogeneous requests and an oracle-gap test, not
  a longer completion from one prompt.

The fundamental dense-model constraint is also important: every Qwen3-14B
decoder layer contributes to exact inference. Probability cannot justify
skipping a dense layer without changing the model. It can optimize uncertain
I/O timing, speculative stopping or an offline discrete placement landscape.

## Evidence levels

| Level | Meaning | Allowed conclusion |
|---|---|---|
| L0 | contract/invariant | implementation is safe enough to test |
| L1 | mechanism smoke | it runs and the intended action occurred; no performance support/falsification |
| L2 | regulated exploratory screen | advance, redesign, or stop for futility |
| L3 | predeclared fixed-stage confirmation | eligible for a performance claim |

All live latency comparisons use randomized paired blocks and the median paired
log ratio, `log(control seconds) - log(candidate seconds)`. L2 reports a 90%
descriptive bootstrap interval. L3 uses a fixed sample and a 95% interval; the
lower bound must be positive and the point effect must clear the practical gate.
Correctness, resource and environment gates override timing.

## Protocol by hypothesis family

| Family | Hypotheses | L1 prerequisite | Regulated L2/L3 evidence |
|---|---|---|---|
| Offline controller/RL | H0, H3, H8 | valid replay and disjoint split | 100-request screen, then 300 chronological requests; oracle gap, regret, MAPE and safety |
| Placement | H1, H10, H13 | frozen plan executes and differs from control | 4 prompt families, randomized cold-cache pairs, 1 then 3 budgets, replay MAPE and plan overlap |
| Adaptive prefetch | H4, H12 | >=80 posterior observations and bounded depth | burn-in excluded; posterior calibration, depth, inflight bytes, misses and exposed wait |
| Learned speculation | H2, H11 | >=200 calibration positions and >=10% action divergence | frozen state, distinct acceptance regimes, best calibrated fixed-k, latency plus distribution test |
| Certified search | H5 | adversarial certificate audit and >=70% predicted pruning | real-head pruning, fallbacks, index amortization and end-to-end latency |
| Artifact design | H6, H7 | representative bitwise round trips | cross-family/checkpoint coverage before any live GPU claim |
| RAM overlay | H9 | host can pin >=1.6 GB with no pageable fallback | matched-VRAM paired screen, then fixed-stage pinned confirmation |

Exact stage counts, diagnostics and decision rules are queryable with:

```bash
afterimage test-plan h12-bayesian-prefetch --json
afterimage test-plan h13-qubo-residency
```

## Current H11-H13 results under the new interpretation

The common four-family run was interrupted while H13 was executing. Its nine
completed methods were preserved verbatim in
`results/2026-08-21_h12_h13_qwen3-14b_rtx3080_interrupted1.json`; H13 was rerun
alone in `results/2026-08-21_h13_qubo_qwen3-14b_rtx3080_screen2.json`.

| Method | Peak VRAM | s/token | vs AirLLM | Evidence-aware verdict |
|---|---:|---:|---:|---|
| AirLLM 3.1.0 | 1.58 GB | 30.02 | 1.00x | external anchor |
| exact minimum-memory | 1.72 | 33.05 | 0.91x | compression alone remains slower |
| exact fixed prefetch, 4 GB | 3.93 | 20.04 | 1.50x | H12 control |
| PI prefetch | 3.93 | 21.25 | 1.41x | below fixed |
| MPC prefetch | 3.93 | 35.03 | 0.86x | controller chose depth zero and missed every demand |
| **H12 Bayesian probit** | 3.93 | **19.76** | **1.52x** | L1 mechanism works; effect not established |
| profiled knapsack | 3.93 | 19.55 | 1.54x | H13 control |
| critical path | 3.93 | **18.99** | **1.58x** | strongest contemporaneous exact non-spec row |
| replay CEM | 3.93 | 19.48 | 1.54x | parity; predicted only +0.32% over control |
| H13 QUBO, separate run | 3.93 | 18.38 | 1.63x* | no treatment: optimizer returned the profiled-control plan |

`*` Descriptive cross-run ratio, not a paired result. QUBO's event-DAG report
was exactly equal to its profiled-knapsack seed (`optimized_over_control=0`) in
both calibrations. Its 18.38 s/token therefore measures run-to-run variation of
the control plan, not a QUBO improvement. H13 stops at L1 until its optimizer
produces a genuinely different frozen plan.

For H12, the four paired cells give a median effect of **+2.23%**, a descriptive
90% interval of **[-6.26%, +7.50%]**, only 50% sign consistency, and +1.40% by
aggregate wall time. It selected depths 3-4 and accumulated 313 read and 312
lead-window observations, so the mechanism did act. But exposed wait was higher
than fixed depth and the point effect is below the 5% gate. This is L1 plus an
underpowered L2-shaped screen: redesign/extend, not support and not final
falsification.

The longer H11 screen (`results/2026-08-21_h11_neural_utility_qwen3-14b_rtx3080_screen3.json`)
used two prompt regimes and 16 tokens. Neural utility was 5.33 versus fixed-k
5.84 s/token (+9.5% aggregate direction), but both arms used identical target
sweeps and accepted-token counts. The neural policy recorded **zero stop
decisions**. It therefore failed the L1 action-divergence prerequisite; the
timing direction cannot be credited to the network.

## Proposed execution plan

1. **H13: stop now.** Do not buy more GPU repetitions for a plan identical to
   its control. Redesign the action space around storage extents/layout or add
   higher-order interactions; require nonzero plan divergence in replay first.
2. **H11: return to calibration, not latency.** Collect at least 200 draft
   positions across low/medium/high acceptance regimes and another draft/target
   pair. Freeze only a state that stops in >=10% of opportunities; otherwise
   kill the small network as fixed-k in disguise.
3. **H12: run one regulated L2 session.** Use two randomized paired blocks,
   four prompt families, four tokens after disjoint burn-in, fixed depth 2 as
   control, and log posterior calibration plus inflight bytes. Stop if the 90%
   upper interval is below +5% or wait remains worse; advance only on consistent
   throughput and >=10% lower exposed wait. Target: <=40 minutes.
4. **H9: keep environment-gated.** This WSL host exposes only 64 MB memlock.
   Do not treat pageable RAM as pinned H9; rerun only on a host that successfully
   pins the full 1.56 GB head.
5. **AirLLM: anchor L2/L3 sessions, not every microtest.** Rerun it once per
   regulated hardware session with identical model, BF16 greedy protocol, EOS
   handling and peak-VRAM counter. Compare matched-memory and extra-memory rows
   separately.
6. **Highest-value next engineering hypothesis:** co-design storage extents and
   residency, then fuse exact decode with consumption. Current evidence says
   policies are searching over a runtime whose dominant cost is still material
   movement/decode; another controller cannot manufacture a large oracle gap.

## Literature boundary

Analytic prefetch models already establish that fixed request cost, layout,
queueing and compute/I/O overlap jointly determine benefit
([Shriver et al., 1999](https://www.usenix.org/legacy/publications/library/proceedings/usenix99/full_papers/shriver/shriver_html/index.html)).
Distribution-driven probabilistic timing constraints are established in
[memory-latency regulation](https://drops.dagstuhl.de/opus/volltexte/2023/18033/pdf/LIPIcs-ECRTS-2023-4.pdf),
and probabilistic runtime scheduling appears in
[ALERT](https://www.usenix.org/conference/atc20/presentation/wan). QUBO plus
classical annealing for discrete placement is also established
([Dury and Di Matteo, 2020](https://arxiv.org/abs/2009.00140)). H12/H13 are
candidate domain-specific compositions, not new probability or quantum theory.
Their current measurements do not support a standalone novelty claim.
