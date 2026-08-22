# New methods: hypotheses, boundaries, and tests

> Historical first screen. H9 has since run with genuinely pinned memory on
> Qwen3-0.6B, and H6-H8 now have artifact/replay evidence. See the controlling
> [all-hypotheses report](ALL_HYPOTHESES_AND_BASELINES.md).

Date: 2026-08-21

Status: H9-H13 are implemented as opt-in research configurations. They do not
replace the existing H0-H8 methods and they do not yet have confirmatory GPU
results. “Novel” below means a candidate combination not found in the reviewed
sources, not a priority or publication claim. H14-H15 (storage-extent
residency) are documented in
[RESEARCH_METHODS.md](RESEARCH_METHODS.md#h14-h15----storage-layout-as-a-residency-action)
rather than here. [HYPOTHESIS_LINEAGE.md](HYPOTHESIS_LINEAGE.md) is the single
table of every hypothesis's source, adaptation and current verdict, H0-H15.

## Current exploratory screens

These are one-prompt mechanism screens, not the five-repeat tests below:

| Method | Candidate | Control | Direction | Gate status |
|---|---:|---:|---:|---|
| H9 pageable RAM overlay, 2 tokens | 31.99 s/token | 32.62 | +1.9% throughput | below 10% |
| H10 replay-CEM, 1 token | 18.37 s/token | 18.75 critical-path | +2.1% | below 8% |
| H11 neural utility, 8 tokens | 4.39 s/token | 4.59 fixed-k | +4.4% | inconclusive; same sweeps/accepts |

H9 preserved both token IDs, lowered peak allocated VRAM from 1.689 to 1.683
GB, and removed 1.065 GB of disk reads per token. The host's hard memlock limit
is only 64 MB, so the intended 1.56 GB pinned allocation degraded explicitly to
pageable RAM. At the time of this screen, pinned H9 was therefore untested;
the later genuine-pinned 0.6B result is in the controlling report. Pageable H9 is not supported by
the screen.

H10's replay predicted only 0.6% improvement over the best seeded calibration
plan. Its real 2.1% direction is below the gate and within single-run noise. H11
used two target sweeps and accepted seven draft tokens in both arms; without an
observed action difference, its 4.4% direction cannot be credited to the tiny
network. The raw immutable results are in `results/2026-08-21_h9_*`,
`results/2026-08-21_h10_*`, and `results/2026-08-21_h11_*`.

## The fundamental reframing

Afterimage is not primarily compute-limited. A target sweep is a recurring
material-handling job: weights move through storage, host memory and VRAM, with
setup costs and overlapping resources. This suggests three transfers from other
fields:

1. compiler liveness: allocate memory by *when* a weight is live, not merely by
   whether it is resident;
2. industrial digital twins/model-based RL: search complete schedules safely in
   replay, then freeze the policy before touching production;
3. survival analysis and cascade retrieval: a speculative chain reveals a
   prefix up to the first rejection and censors the tail.

## H9 — liveness-guided output-head overlay (implemented)

Configuration: `lm_head_policy="ram_overlay"` with decoded RAM and explicit
VRAM/RAM budgets.

Method: keep the exact decoded `lm_head.weight` in pinned host RAM. Transformer
layers and the output head have non-overlapping live ranges: decoder-layer
weights are dead before vocabulary projection begins. Copy the head into VRAM
only around its forward hook, run the ordinary full matrix multiplication, then
return the parameter to a meta placeholder. This replaces disk read + entropy
decode with a host-to-device copy without making the head permanently resident.

Why it may work: on Qwen3-14B the head is about 1.56 GB and its compressed form
still costs about 1.1 GB of storage traffic per token. PCIe transfer from pinned
RAM should be materially cheaper than cold NVMe read plus GPU entropy decode,
while peak VRAM remains set by the larger of the layer and head phases rather
than their sum.

Hypothesis: at matched peak VRAM (within 5%), H9 improves committed tokens/s by
at least 10% over exact disk-streaming and emits identical greedy token IDs.

Quick screen:

```bash
python scripts/run_bounded_suite.py \
  --methods exact-min,ram-overlay-head --max-new-tokens 4 \
  --time-budget-minutes 20 --out results/H9-screen.json
```

Confirmatory test: four prompt families, 16+ tokens, five randomized paired
cold-cache repeats. Record wall time, token IDs, peak allocated VRAM, disk bytes,
host RAM, and host-to-device bytes. Kill on any token mismatch, >5% peak-VRAM
increase, pinned-memory failure, or <10% median gain.

Literature boundary: liveness analysis and memory reuse are established in
[SuperNeurons](https://web.eecs.umich.edu/~mosharaf/Readings/SuperNeurons.pdf)
and graph-level layout optimization in
[ROAM](https://arxiv.org/abs/2310.19295). Host-memory offload is established in
[FlexGen](https://arxiv.org/abs/2303.06865),
[Select-N](https://arxiv.org/abs/2502.08182), and
[ATSInfer](https://arxiv.org/abs/2607.10183). The reviewed work did not show this
specific exact, late-live output-head overlay for losslessly compressed
NVMe-streamed inference. The novelty candidate is that narrow composition, not
liveness or offloading themselves.

## H10 — digital-twin whole-set residency search (implemented)

Configuration: `placement_policy="replay_cem"` plus a frozen
`replay_plan_state` generated by `afterimage research optimize-residency`.

Method: treat a complete resident tensor set as the action. A measured event DAG
is the digital twin; zeroing preparation spans for resident tensors gives the
simulated reward (negative makespan). Cross-entropy method (CEM) policy search
samples budget-feasible sets, retains elites, and updates Bernoulli placement
probabilities. The live engine never explores: it validates the manifest hash,
budget, headroom and selected bytes, then executes the frozen plan.

Why it may work: H1 valued every tensor independently and improved only 1.6%.
Critical paths switch when several spans disappear; therefore the value of A can
depend on whether B is resident. Direct whole-set replay captures these
interactions without assigning 441 live RL actions one scalar reward.

Hypothesis: H10 improves held-out committed tokens/s by at least 8% over the
best of traffic-density, measured-knapsack and independent critical-path plans
at the same VRAM budget, while replay prediction error stays below 10%.

Build and test:

```bash
afterimage research optimize-residency traces/calibration-*.json \
  --manifest STORE/manifest.json --vram-budget-gb 4 \
  --iterations 12 --population 64 --out plans/qwen14b-cem-4gb.json

afterimage run MODEL PROMPT --store STORE --vram-budget-gb 4 \
  --placement-policy replay_cem \
  --replay-plan-state plans/qwen14b-cem-4gb.json
```

Use disjoint prompt traces for calibration and generation for evaluation. Test
at no fewer than three budgets, five paired cold-cache repeats, and compare all
three existing planners. Report replay makespan, real makespan, search
evaluations/time, plan overlap, peak VRAM and token equality. Kill if simulator
MAPE exceeds 10%, rankings fail to transfer across prompt families, or the real
gain is below 8%.

Literature boundary: CEM is established for combinatorial optimization
([Rubinstein, 1999](https://doi.org/10.1023/A:1010091220143)) and model-based
planning ([Bharadhwaj et al., 2020](https://proceedings.mlr.press/v120/bharadhwaj20a.html)).
Digital-twin-assisted RL is established for task scheduling
([Wang et al., 2022](https://arxiv.org/abs/2208.01781)). H10's candidate novelty
is using exact event-DAG counterfactual replay as a safe world model for
lossless tensor-residency *set* search. CEM and digital twins are not new.

## H11 — tiny censored-survival utility controller (implemented)

Configuration: `spec_k_policy="neural_utility"` with a separately calibrated,
then frozen `spec_policy_state`.

Method: a six-hidden-unit MLP predicts conditional acceptance from draft
confidence, entropy, position and their interaction. Training uses the accepted
prefix and first rejection only; the unseen tail is censored. At each position,
the controller compares predicted expected tokens/s for stopping now versus
adding the candidate. It therefore learns a utility, not an accuracy proxy. The
ordinary speculative verifier remains unchanged, so bad predictions affect
latency but not the target distribution.

Why it may work: H2's position × confidence table was sparse and collapsed to
fixed `k=8`. Pooling related positions with a tiny nonlinear model should need
fewer samples, while the explicit utility includes the unusually high target
sweep cost that generic confidence thresholds ignore.

Hypothesis: after calibration on disjoint prompts, H11 improves committed
tokens/s by at least 8% over the best successively-halved fixed `k`, with no
distributional-correctness regression.

Quick screen:

```bash
python scripts/run_bounded_suite.py \
  --methods spec-fixed,spec-neural --max-new-tokens 8 \
  --time-budget-minutes 30 --out results/H11-screen.json
```

Confirmatory test: calibrate on at least 200 observed draft positions spanning
prompt families and acceptance regimes; freeze the state; evaluate 128+ tokens,
five paired repeats, temperature 0 and 1. Report Brier score/calibration error,
chosen chain length, accepted tokens/sweep, draft/target time, throughput and a
two-sample distributional test at temperature 1. Kill on poor held-out
calibration, a non-positive paired lower confidence bound, or regression in a
major prompt family.

Literature boundary: learned discrete hazards are established
([Gensheimer and Narasimhan, 2019](https://arxiv.org/abs/1805.00917)); adaptive
speculation is established by
[SpecDec++](https://arxiv.org/abs/2405.19715) and
[BanditSpec](https://arxiv.org/abs/2505.15141). The candidate novelty is the
combination of censored prefix training, a very small pooled survival network,
and an offloaded-target throughput objective. This is the least defensible of
the three as a standalone novelty claim and must be presented as a new
composition unless a broader search establishes otherwise.

## Priority and expected value

1. Re-test H9 only on a host that can pin at least 1.6 GB; otherwise kill it.
2. Keep H10 as a cheap offline planner, but do not fund confirmatory runs unless
   replay predicts at least 8% over its best seeded control on held-out traces.
3. Expand H11 calibration across acceptance regimes and require logged stop
   decisions before another speed comparison; otherwise kill it as fixed-k in
   disguise.

Two additional ideas remain proposals, not implementations: phase-color the
resident target and draft model so they time-share VRAM across draft/verify
phases; and make storage layout an action so the replay agent jointly chooses
resident tensors and contiguous read extents. Both need deeper runtime changes
and should be attempted only if H9/H10 pass their gates.

## H12-H13 follow-on

Two further opt-in exact methods are now implemented:

- H12 `prefetch_policy="bayes_probit"` models log read latency and observed
  per-layer lead windows with normal-inverse-gamma posteriors. A probit chance
  constraint chooses the smallest prefetch depth predicted to be ready on time.
- H13 `placement_policy="replay_qubo"` derives linear and pairwise residency
  coefficients from event-DAG counterfactuals and applies a classical simulated
  annealer to the capacity-penalized binary energy.

The first screens do not establish either performance hypothesis. H12 was 1.4%
faster by aggregate wall time (2.23% paired median, 90% descriptive interval
[-6.26%, +7.50%]) but increased exposed wait and missed its 5% gate. H13's QUBO
returned exactly its profiled-knapsack seed, so a faster separate run is
run-to-run variation rather than a treatment effect. The longer H11 rerun also
made zero neural stop decisions despite a favorable timing direction.

These outcomes prompted a hypothesis-aware L0-L3 testing system rather than
another universal short matrix. See
[REGULATED_TEST_PLAN_2026-08-21.md](REGULATED_TEST_PLAN_2026-08-21.md) for the
protocols, current H11-H13 analysis and proposed next execution order.
