# Configurable research program: H0-H18

This document turns the research plan into falsifiable, opt-in experiments. The
existing engine stays the control. Every candidate is a named method profile,
every result is written once with its configuration and environment, and every
method has a kill criterion. A method being implemented does **not** mean it is
faster.

H0-H8 are the original program. H9-H18 extend it without changing any of them.
For measured outcomes and the ranking that follows from them, read
[ALL_HYPOTHESES_AND_BASELINES.md](ALL_HYPOTHESES_AND_BASELINES.md); for where
each idea came from and what was actually borrowed, read
[HYPOTHESIS_LINEAGE.md](HYPOTHESIS_LINEAGE.md). Survey cutoff: 21 August 2026.

## 1. Start from the actual objective

For an offloaded dense model, the useful first-order cost is

```text
time / committed token
    = critical-path time per target sweep / committed tokens per sweep
```

This differs from adding I/O, decode, transfer, and compute timers. Those stages
overlap. It also explains why optimizing a locally large timer can have zero
end-to-end value when that timer has slack.

The research program therefore has two levels:

1. identify whether workload-dependent choice has enough oracle headroom (H0);
2. test mechanisms that shorten the critical path or amortize a target sweep
   (H1-H8).

If H0 fails, do not build a larger adaptive controller. Use the best global
profile.

## 2. What is implemented

| ID | Candidate | Candidate profile | Control profile | Runtime status | Exactness contract |
|---|---|---|---|---|---|
| H0 | joint semantic/system oracle gap | `contextual-linucb-v1` | `exact-streaming-v1` | dataset runner | reference execution equivalent |
| H1 | critical-path residency planner | `critical-path-v1` | `exact-streaming-v1` | live, opt-in | reference execution equivalent |
| H2 | cost-aware rejection hazard | `hazard-cost-v1` | `tuned-fixed-spec-v1` | live, opt-in | target-distribution exact |
| H3 | baseline-guarded contextual profile bandit | `contextual-linucb-v1` | `exact-streaming-v1` | offline replay | reference execution equivalent |
| H4 | feedback-controlled prefetch | `pi-prefetch-v1` | `fixed-prefetch-v1` | live, opt-in | reference execution equivalent |
| H5 | certified greedy LM-head search | `certified-mips-v1` | `exact-streaming-v1` | live, greedy only | greedy-token exact |
| H6 | per-tensor exact physical representations | `per-tensor-representation-v1` | `exact-streaming-v1` | artifact planner | reference execution equivalent |
| H7 | expert-local lossless reference coding | `xor-reference-v1` | `exact-streaming-v1` | artifact codec/audit | weight exact |
| H8 | shadow model-based joint controller | `model-based-rl-v1` | `contextual-linucb-v1` | shadow/offline only | reference execution equivalent |
| H9 | liveness-guided RAM output-head overlay | `ram-overlay-head-v1` | `exact-streaming-v1` | live, opt-in | reference execution equivalent |
| H10 | digital-twin whole-set residency search | `replay-cem-v1` | `profiled-knapsack-v1` | offline planner, frozen plan | reference execution equivalent |
| H11 | tiny censored-survival utility controller | `neural-utility-spec-v1` | `tuned-fixed-spec-v1` | live, opt-in | target-distribution exact |
| H12 | Bayesian chance-constrained prefetch | `bayes-probit-prefetch-v1` | `fixed-prefetch-v1` | live, opt-in | reference execution equivalent |
| H13 | event-interference QUBO residency | `replay-qubo-v1` | `profiled-knapsack-v1` | offline planner, frozen plan | reference execution equivalent |
| H14 | bounded contiguous storage reads | `coalesced-storage-v1` | `exact-streaming-v1` | live, opt-in | reference execution equivalent |
| H15 | physical-extent QUBO residency | `extent-qubo-v1` | `profiled-knapsack-v1` | offline planner, frozen plan | reference execution equivalent |
| H16 | speculation-conditioned critical-path residency | `spec-critical-path-v1` | `tuned-fixed-spec-v1` | live, opt-in | target-distribution exact |
| H17 | tensor-scoped overlap-preserving micro-extents | `tensor-extents-v1` | `exact-streaming-v1` | live, opt-in | reference execution equivalent |
| H18 | rollback-cached target verification | `rollback-cached-spec-v1` | `tuned-fixed-spec-v1` | live, opt-in | target-distribution exact |

Every candidate above is selectable by profile name and reproducible from the
registry: `afterimage research experiments --json` prints it, and
`afterimage research test-plan HYPOTHESIS --json` prints the protocol that
governs it.

The `execution_policy`, `representation_policy`, and `expert_codec` fields are
experiment-profile markers. The v2 dense engine rejects them when their required
request-level controller or artifact store is absent. This is a capability gate,
not a fallback.

## 3. Literature boundary and the genuinely new part

The mechanisms deliberately borrow from mature fields:

- SpecDec++ proves a threshold structure for adaptive speculative stopping and
  reports an acceptance predictor approach. AdaEDL derives the entropy criterion
  used in the implementation. BanditSpec establishes training-free bandits over
  speculative configurations. [SpecDec++](https://arxiv.org/abs/2405.19715),
  [AdaEDL](https://arxiv.org/abs/2410.18351),
  [BanditSpec](https://arxiv.org/abs/2505.15141)
- Critical-path DAG analysis is established for heterogeneous CPU/GPU traces;
  PyTorch's Holistic Trace Analysis also builds a weighted DAG with CPU/GPU
  synchronization edges. The proposed contribution is using counterfactual
  critical-path value as the objective for *compressed weight residency*, not
  inventing DAG analysis. [Hybrid MPI-CUDA critical-path analysis](https://doi.org/10.1177/1094342016661865),
  [HTA critical-path documentation](https://hta.readthedocs.io/en/latest/source/features/lightweight_critical_path_analysis.html)
- Pythia demonstrates feedback-aware RL prefetching and Sibyl applies online RL
  to hybrid-storage placement. Their dense per-access rewards motivate H4, while
  also showing why Afterimage's hundreds-of-items/one-wall-time placement problem
  has poor online credit assignment. [Pythia](https://arxiv.org/abs/2109.12021),
  [Sibyl](https://arxiv.org/abs/2205.07394)
- Conservative Linear UCB motivates retaining a baseline. The current H3 guard
  is a practical lower-confidence check, **not** CLUCB's cumulative
  high-probability guarantee. [Conservative Contextual Linear Bandits](https://arxiv.org/abs/1611.06426)
- Exact branch-and-bound MIPS is established. H5 adds explicit floating-point
  intervals and a normal LM-head fallback because language-model token equality
  is stricter than approximate retrieval recall. [Tree-based MIPS](https://arxiv.org/abs/1202.6101)
- Multiple-choice knapsack for physical layout/mapping selection is established
  in design-space exploration. H6 applies it to exact alternatives on the
  compressed streaming path. [NicePIM](https://arxiv.org/abs/2305.19041)
- BitX already uses XORed redundancy for lossless delta compression between base
  and fine-tuned LLMs. H7 is therefore a **transfer test inside one MoE
  checkpoint**, not a claim to have invented XOR reference coding.
  [BitX/ZipLLM](https://arxiv.org/abs/2505.06252)
- Successive halving is the screening primitive implemented here. It follows the
  resource-allocation idea behind Hyperband; it is not presented as BOHB or as a
  new optimizer. [Hyperband](https://www.jmlr.org/papers/v18/16-558.html),
  [BOHB](https://proceedings.mlr.press/v80/falkner18a.html)

The strongest defensible novelty candidates are consequently:

1. a storage-cost-aware *survival* policy for speculative stopping under a very
   expensive streamed target sweep (H2);
2. counterfactual event-DAG value for exact compressed residency (H1);
3. roundoff-certified greedy LM-head pruning, including an exact execution
   fallback (H5);
4. only if H0 shows real joint headroom, conservative selection among complete
   exact system profiles rather than learning low-level lossy actions (H3/H8).

H6 and H7 are useful compositions, but should not be the paper's main novelty
unless measurement exposes an unexpected effect.

## 4. General experiment contract

All generation experiments use `afterimage.experiments.run_paired`:

- candidate and named control trials are randomly interleaved;
- repeat seeds are paired;
- the primary result is the median relative effect with a paired bootstrap
  interval;
- deterministic exactness contracts are invalidated by any output-token
  mismatch;
- distribution-exact speculation is not incorrectly required to produce the
  same random realization;
- results include method/config identifiers, config fingerprints, trial order,
  token IDs, bytes read, peak allocated VRAM, and relevant controller counters;
- completed JSON results are immutable and include Python, PyTorch, CUDA, GPU,
  and Git revision when available.

For publishable measurements, additionally record the cache regime, power mode,
storage device, driver, model revision, tokenizer revision, prompt dataset hash,
and warm-up policy. The current API cannot drop the operating-system page cache;
use the repository's Linux/WSL head-to-head scripts where cold-cache evidence is
required.

Never combine cold and warm cache observations in one confidence interval.

The web UI exposes every hypothesis under **Experiment Lab**. Definitions are
also available at `GET /api/experiments`, and runs start at
`POST /api/experiments/{id}/runs`.

## 5. Evidence levels

One prompt/token matrix measures whether configurations execute under a single
wall-clock budget. It does not give every hypothesis the evidence it needs.
Residency is chosen offline and needs disjoint traces, several VRAM budgets,
real plan divergence, and replay-transfer validation. Online prefetch control
needs burn-in, posterior diagnostics, and steady-state layer demands rather than
first-token averages. Learned speculation needs enough accepted and rejected
positions to calibrate, a frozen held-out state, and proof that it takes
different actions at all. Artifact and codec hypotheses need hundreds of bitwise
tensor round trips before GPU latency is worth measuring.

`afterimage/protocols.py` therefore maps every hypothesis to four levels, and
`afterimage research test-plan` exposes them:

| Level | Meaning | Allowed conclusion |
|---|---|---|
| L0 | contract or invariant | the implementation is safe enough to test |
| L1 | mechanism smoke | it runs and the intended action occurred, with no performance support or falsification |
| L2 | regulated exploratory screen | advance, redesign, or stop for futility |
| L3 | predeclared fixed-stage confirmation | eligible for a performance claim |

All live latency comparisons use randomized paired blocks and the median paired
log ratio, `log(control seconds) - log(candidate seconds)`. L2 reports a 90%
descriptive bootstrap interval. L3 uses a fixed sample and a 95% interval, the
lower bound must be positive, and the point effect must clear the practical
gate. Correctness, resource, and environment gates override timing.

What each family has to show before its numbers mean anything:

| Family | Hypotheses | L1 prerequisite | Regulated L2/L3 evidence |
|---|---|---|---|
| Offline controller/RL | H0, H3, H8 | valid replay and disjoint split | 100-request screen, then 300 chronological requests; oracle gap, regret, MAPE, and safety |
| Placement | H1, H10, H13 | frozen plan executes and differs from control | 4 prompt families, randomized cold-cache pairs, 1 then 3 budgets, replay MAPE and plan overlap |
| Adaptive prefetch | H4, H12 | at least 80 posterior observations and bounded depth | burn-in excluded; posterior calibration, depth, inflight bytes, misses, and exposed wait |
| Learned speculation | H2, H11 | at least 200 calibration positions and at least 10% action divergence | frozen state, distinct acceptance regimes, best calibrated fixed-k, latency plus distribution test |
| Certified search | H5 | adversarial certificate audit and at least 70% predicted pruning | real-head pruning, fallbacks, index amortization, and end-to-end latency |
| Artifact design | H6, H7 | representative bitwise round trips | cross-family and cross-checkpoint coverage before any live GPU claim |
| RAM overlay | H9 | host can pin at least 1.6 GB with no pageable fallback | matched-VRAM paired screen, then fixed-stage pinned confirmation |
| Storage request geometry | H14 | exact tokens, at least 50% fewer calls, at most 5% byte amplification | fixed-residency randomized pairs |

Exact stage counts, diagnostics, and decision rules are queryable per
hypothesis:

```bash
afterimage research test-plan h12-bayesian-prefetch --json
afterimage research test-plan h13-qubo-residency
```

One constraint bounds the whole program. Every decoder layer of a dense model
contributes to exact inference, so probability can never justify skipping one
without changing the model. It can only optimize uncertain I/O timing,
speculative stopping, or an offline discrete placement landscape.

## 6. H0 - joint oracle-gap diagnostic

Hypothesis: semantic context and system state jointly provide at least 12% more
throughput than the best global profile.

Input rows:

```json
{
  "profile": "fixed-prefetch-v1",
  "semantic_bucket": "code",
  "system_bucket": "cold-cache",
  "committed_tokens_per_second": 0.42
}
```

Test: balanced held-out rows across profile × semantic bucket × system bucket.
Report the best global, semantic-only, system-only, and joint oracles. Kill
adaptive profile selection if joint uplift is below 12%, or if the apparent
uplift vanishes under leave-one-prompt-family-out evaluation.

## 7. H1 - critical-path residency

Generate raw steady-state traces without changing the default planner:

```bash
afterimage run MODEL PROMPT --vram-budget-gb 4 \
  --trace-output traces/control-01.json --max-new-tokens 32
afterimage research profile-trace traces/control-*.json \
  --out profiles/MODEL-critical-path.json \
  --manifest STORE/manifest.json
```

Collect complementary control traces until the profile covers at least 90% of
placement-eligible tensors. A tensor already resident in one trace has no
per-sweep preparation observation; use additional baseline placements that make
it stream. The engine rejects lower coverage instead of filling the gap with the
old traffic proxy.

Then run H1 with common overrides:

```json
{
  "vram_budget_gb": 4.0,
  "critical_path_profile": "profiles/MODEL-critical-path.json"
}
```

The trace recorder orders operations on storage-reader and CUDA resources and
adds read/decode/transfer → layer-compute dependencies. For every tensor, the
profiler sets its preparation durations to zero and replays the DAG; the
placement value is the baseline makespan minus the replayed makespan, not raw
critical-path occupancy. This correctly accounts for a second path becoming critical. Profiling mode
synchronizes CUDA around preparation spans; it is intentionally intrusive, so
the profile must be learned on separate runs and evaluated with tracing off.

Primary gate: at least 8% paired throughput gain. Mechanism gates: held-out
predicted-versus-observed rank correlation ≥0.8 and no exact-token mismatch.
Evaluate at no fewer than three VRAM budgets; a ranking that wins at only one
hand-picked budget is not a general result.

## 8. H2 - censored rejection hazard plus cost

Each sweep reveals accepted positions up to the first rejection; later
positions are censored, not negative examples. `HazardCostPolicy` keeps Beta
posteriors by position and confidence bin. It drafts another token only when

```text
P(accept next | survived so far, position, confidence)
    × measured target seconds per committed token
    > measured marginal draft seconds
```

This is distinct from a confidence-only threshold: the same rejection risk can
justify more drafting when a streamed target sweep is expensive.

Protocol:

1. tune fixed `spec_k` with `successive_halving` on calibration prompts;
2. calibrate the hazard policy to its own state file on the same calibration
   split (`spec_policy_learn=true`);
3. evaluate both on disjoint prompts with
   `spec_policy_learn=false`; H2's API refuses a missing or learning state;
4. use temperature 1.0 and compare distributional correctness separately via
   the existing verifier tests;
5. stratify by draft/target pair, prompt family, and acceptance regime.

Primary gate: ≥8% median throughput improvement over the tuned fixed chain with
a positive paired lower confidence bound. Kill on regression in any major
prompt family or an effective sample size too small to calibrate the hazard.

## 9. H3 - contextual selection among complete profiles

Actions are complete, versioned profiles, not arbitrary knob combinations. The
context vector should include prompt length/task embedding plus current cache,
I/O bandwidth, prefetch hit rate, and memory pressure. Reward is committed
tokens/s minus explicit startup or memory penalties.

The disjoint calibration and chronological result datasets share this schema:

```json
{
  "context": [1.0, 0.2, 0.0],
  "rewards": {
    "exact-streaming-v1": 0.31,
    "pi-prefetch-v1": 0.38
  }
}
```

Calibration rows provide full feedback for every complete profile; evaluation
rows reveal only the selected reward to the learner. The replay reports
achieved/oracle reward. Use chronological evaluation, an explicit
baseline-violation count, and Page-Hinkley resets as an ablation for cache or
storage regime changes. A live deployment should load this offline calibration
rather than explore cold. Gate: ≥95% of joint oracle reward, and only if H0 first
shows ≥12% oracle headroom.

## 10. H4 - feedback-controlled prefetch

PI control observes whether the demanded layer was ready and adjusts bounded
queue depth. The `mpc-prefetch-v1` ablation estimates layer bytes, bandwidth,
compute time, and exposed wait, then minimizes predicted stall plus an overfetch
penalty. Both preserve exact bytes; H4's preregistered primary candidate is PI.

Test fixed depths `{0,1,2,4,8}` first; the control is the best fixed depth, not
the repository default. Replay bandwidth perturbations or run controlled I/O
contention. Report throughput, exposed wait, hit rate, and peak host memory.
Gate: ≥5% throughput and ≥10% exposed-stall reduction.

## 11. H5 - certified greedy output head

The index partitions output rows and stores coordinate bounds. A query evaluates
promising blocks and prunes a block only when its outward-rounded real-dot upper
bound plus a standard floating-point accumulation bound lies below the best
candidate's lower bound. Inconclusive cases execute the original full head.

Scope is deliberately narrow: batch size 1, greedy decoding, bias-free head,
resident `lm_head.weight`. Sampling always uses the full distribution.

Report certificate rate, rows evaluated, fallbacks, one-time index build time,
index RAM, throughput, and token equality. Any certified mismatch is a hard
failure. Kill if fewer than 70% of rows are pruned on at least 95% of tokens or
steady-state throughput fails to improve by 8%.

## 12. H6 - exact physical representation planning

`RepresentationOption` describes exact storage, RAM, VRAM, artifact, and
measured preparation cost for one tensor representation. The dynamic program
chooses exactly one option per tensor under resource budgets and rejects lossy
options. Artifact validation occurs before execution.

Input example:

```json
{
  "tensor_key": "model.layers.0.mlp.down_proj.weight",
  "name": "compressed_ram",
  "ram_bytes": 74239120,
  "storage_bytes": 74239120,
  "prepare_s": 0.018,
  "exact": true,
  "artifact": "representations/l0-down.bin"
}
```

First gate the predicted plan against every uniform option (≥10%). Only then
materialize artifacts and measure held-out traces. Prediction error above 10%
invalidates the cost model before it invalidates the physical-design idea.

## 13. H7 - expert-local lossless reference coding

The codec XORs same-shape/same-dtype tensors, compresses the delta, records the
base SHA-256 and target CRC32, and reconstructs through a byte view so NaNs,
signed zero, and every BF16 payload bit survive. Artifacts are self-describing
and atomically written.

Declare an acyclic set of `reference_bases`, then audit every compatible
base→expert pair before building a dependent store. Compare against
**independent Afterimage compression**, not raw BF16. The runner counts base
storage and serialized artifact metadata and chooses independent coding when a
delta is larger. Include dependency depth, random-access decode latency, and
corruption blast radius. Gate: ≥10% total expert-storage reduction after the
base and metadata, with bitwise round-trip for every audited tensor.

## 14. H8 - model-based control, conditionally

This phase is shadow-only until H0 and H3 pass. A trace simulator predicts
profile reward and is permitted to recommend, but not apply, actions. Calibration
data can contain simple `actual_s`/`predicted_s` pairs; that can only produce an
`inconclusive` prerequisite result. A full policy test uses:

```json
{
  "actual_rewards": {"base": 1.0, "candidate": 1.3},
  "predicted_rewards": {"base": 1.1, "candidate": 1.25},
  "baseline_profile": "base"
}
```

Required calibration: MAPE ≤10% and rank correlation ≥0.9. Required control
effect: ≥10% reward over H3. Otherwise stop at the contextual bandit; a larger RL
stack is not justified.

## 15. H9-H11 come from other fields

H0-H8 treat the engine as a compute pipeline. H9-H11 start from a different
reading: Afterimage is not primarily compute-limited, and a target sweep is a
recurring material-handling job. Weights move through storage, host memory, and
VRAM, with setup costs and overlapping resources. Three transfers follow from
that.

1. Compiler liveness: allocate memory by *when* a weight is live, not merely by
   whether it is resident.
2. Industrial digital twins and model-based RL: search complete schedules safely
   in replay, then freeze the policy before touching production.
3. Survival analysis: a speculative chain reveals a prefix up to the first
   rejection and censors the tail.

All three are implemented as opt-in configurations. None replaces an H0-H8
method, and none has confirmatory GPU evidence. Measured outcomes are in
[ALL_HYPOTHESES_AND_BASELINES.md](ALL_HYPOTHESES_AND_BASELINES.md).

## 16. H9 - liveness-guided output-head overlay

Configuration: `lm_head_policy="ram_overlay"` with decoded RAM and explicit
VRAM/RAM budgets.

Keep the exact decoded `lm_head.weight` in pinned host RAM. Transformer layers
and the output head have non-overlapping live ranges, because decoder-layer
weights are dead before vocabulary projection begins. Copy the head into VRAM
only around its forward hook, run the ordinary full matrix multiplication, then
return the parameter to a meta placeholder. That replaces a disk read plus
entropy decode with a host-to-device copy, without making the head permanently
resident.

Why it might work: on Qwen3-14B the head is about 1.56 GB, and its compressed
form still costs about 1.1 GB of storage traffic per token. PCIe transfer from
pinned RAM should be materially cheaper than a cold NVMe read plus GPU entropy
decode, while peak VRAM stays set by the larger of the layer and head phases
rather than their sum.

Hypothesis: at matched peak VRAM, within 5%, H9 improves committed tokens/s by
at least 10% over exact disk streaming and emits identical greedy token IDs.

```bash
python scripts/run_bounded_suite.py \
  --methods exact-min,ram-overlay-head --max-new-tokens 4 \
  --time-budget-minutes 20 --out results/H9-screen.json
```

Confirmatory test: four prompt families, 16 or more tokens, five randomized
paired cold-cache repeats. Record wall time, token IDs, peak allocated VRAM,
disk bytes, host RAM, and host-to-device bytes. Kill on any token mismatch, a
peak-VRAM increase above 5%, a pinned-memory failure, or a median gain below
10%.

Literature boundary: liveness analysis and memory reuse are established in
[SuperNeurons](https://web.eecs.umich.edu/~mosharaf/Readings/SuperNeurons.pdf),
and graph-level layout optimization in [ROAM](https://arxiv.org/abs/2310.19295).
Host-memory offload is established in
[FlexGen](https://arxiv.org/abs/2303.06865),
[Select-N](https://arxiv.org/abs/2502.08182), and
[ATSInfer](https://arxiv.org/abs/2607.10183). The reviewed work did not show
this specific exact, late-live output-head overlay for losslessly compressed
NVMe-streamed inference. The candidate is that narrow composition, not liveness
or offloading themselves.

## 17. H10 - digital-twin whole-set residency search

Configuration: `placement_policy="replay_cem"` plus a frozen `replay_plan_state`
generated by `afterimage research optimize-residency`.

Treat a complete resident tensor set as the action. A measured event DAG is the
digital twin, and zeroing preparation spans for resident tensors gives the
simulated reward, negative makespan. The cross-entropy method samples
budget-feasible sets, retains elites, and updates Bernoulli placement
probabilities. The live engine never explores: it validates the manifest hash,
budget, headroom, and selected bytes, then executes the frozen plan.

Why it might work: H1 valued every tensor independently and improved only 1.6%.
Critical paths switch when several spans disappear, so the value of A can depend
on whether B is resident. Direct whole-set replay captures those interactions
without assigning 441 live RL actions one scalar reward.

Hypothesis: H10 improves held-out committed tokens/s by at least 8% over the
best of traffic-density, measured-knapsack, and independent critical-path plans
at the same VRAM budget, while replay prediction error stays below 10%.

```bash
afterimage research optimize-residency traces/calibration-*.json \
  --manifest STORE/manifest.json --vram-budget-gb 4 \
  --iterations 12 --population 64 --out plans/qwen14b-cem-4gb.json

afterimage run MODEL PROMPT --store STORE --vram-budget-gb 4 \
  --placement-policy replay_cem \
  --replay-plan-state plans/qwen14b-cem-4gb.json
```

Use disjoint prompt traces for calibration and generation for evaluation. Test
at no fewer than three budgets and five paired cold-cache repeats, and compare
all three existing planners. Report replay makespan, real makespan, search
evaluations and time, plan overlap, peak VRAM, and token equality. Kill if
simulator MAPE exceeds 10%, if rankings fail to transfer across prompt families,
or if the real gain is below 8%.

Literature boundary: CEM is established for combinatorial optimization
([Rubinstein, 1999](https://doi.org/10.1023/A:1010091220143)) and model-based
planning ([Bharadhwaj et al., 2020](https://proceedings.mlr.press/v120/bharadhwaj20a.html)).
Digital-twin-assisted RL is established for task scheduling
([Wang et al., 2022](https://arxiv.org/abs/2208.01781)). The candidate here is
using exact event-DAG counterfactual replay as a safe world model for lossless
tensor-residency *set* search. CEM and digital twins are not new.

## 18. H11 - tiny censored-survival utility controller

Configuration: `spec_k_policy="neural_utility"` with a separately calibrated,
then frozen, `spec_policy_state`.

A six-hidden-unit MLP predicts conditional acceptance from draft confidence,
entropy, position, and their interaction. Training uses the accepted prefix and
first rejection only, so the unseen tail is censored. At each position the
controller compares predicted expected tokens/s for stopping now against adding
the candidate, which makes it a utility rather than an accuracy proxy. The
ordinary speculative verifier is unchanged, so bad predictions cost latency but
never the target distribution.

Why it might work: H2's position by confidence table was sparse and collapsed to
fixed `k=8`. Pooling related positions with a tiny nonlinear model should need
fewer samples, and the explicit utility includes the unusually high target sweep
cost that generic confidence thresholds ignore.

Hypothesis: after calibration on disjoint prompts, H11 improves committed
tokens/s by at least 8% over the best successively-halved fixed `k`, with no
distributional-correctness regression.

```bash
python scripts/run_bounded_suite.py \
  --methods spec-fixed,spec-neural --max-new-tokens 8 \
  --time-budget-minutes 30 --out results/H11-screen.json
```

Confirmatory test: calibrate on at least 200 observed draft positions spanning
prompt families and acceptance regimes, freeze the state, then evaluate 128 or
more tokens over five paired repeats at temperature 0 and 1. Report Brier score
and calibration error, chosen chain length, accepted tokens per sweep, draft and
target time, throughput, and a two-sample distributional test at temperature 1.
Kill on poor held-out calibration, a non-positive paired lower confidence bound,
or a regression in any major prompt family.

Literature boundary: learned discrete hazards are established
([Gensheimer and Narasimhan, 2019](https://arxiv.org/abs/1805.00917)), and
adaptive speculation by [SpecDec++](https://arxiv.org/abs/2405.19715) and
[BanditSpec](https://arxiv.org/abs/2505.15141). The candidate is the combination
of censored prefix training, a very small pooled survival network, and an
offloaded-target throughput objective. This is the weakest of the three as a
standalone novelty claim, and it should be presented as a new composition unless
a broader search says otherwise.

## 19. H12-H13 - probabilistic prefetch and QUBO residency

Two further opt-in exact methods. Neither skips a dense layer or changes a
weight; both act only on scheduling and placement.

- **H12** (`prefetch_policy="bayes_probit"`) models log read latency and
  observed per-layer lead windows with normal-inverse-gamma posteriors. A probit
  chance constraint picks the smallest prefetch depth predicted to be ready on
  time. It never predicts *which* layer is needed, because dense execution order
  is known exactly. Probability decides only how early an exact read starts.
- **H13** (`placement_policy="replay_qubo"`) derives linear and pairwise
  residency coefficients from event-DAG counterfactuals, then applies a
  classical simulated annealer to the capacity-penalized binary energy. Every
  candidate is repaired to the byte budget and rescored on the original DAG.
  Quantum-inspired, not quantum.

Neither screen established its performance hypothesis, and H13 is the more
instructive failure: its optimizer returned exactly its profiled-knapsack seed,
so there was no treatment to measure. Greedy budget refill was identified as one
confound and removed, and a fresh post-repair run still produced 0% gain and
100% plan overlap. H13 stops at the action-divergence gate until its optimizer
produces a genuinely different frozen plan.

## 20. H14-H15 - storage layout as a residency action

H14 (`storage_read_policy="coalesced_extents"`) merges physically adjacent
compressed blobs into bounded contiguous reads before decoding, cutting fixed
per-request storage overhead. H15 (`placement_policy="replay_extent_qubo"`)
extends H13's QUBO search so its binary variables are bounded storage extents
rather than individual tensors, coupling physical read geometry to residency.

Both are implemented and measured. H14's mechanism gate passed cleanly (read
calls fell 89%, byte amplification stayed at 0%) and its performance gate
failed hard: **27.7% slower**, because a large contiguous read serialises
against decode instead of overlapping with it -- a real negative result about
this engine's I/O/decode overlap, not an unfinished feature. H15's fresh extent
calibration evaluated 81 physical groups and 369 candidates but still returned
the profiled control with 100% overlap and 0% gain. H13 and H15 therefore remain
stopped at the action-divergence gate. Full derivation:
[HYPOTHESIS_LINEAGE.md](HYPOTHESIS_LINEAGE.md).

## 21. H16-H18 - interaction repairs after the full comparison

- **H16** composes the fixed `k=8` speculative control with a distinct H1
  critical-path resident set. The treatment changed, VRAM and tokens matched,
  but it was slower (8.836 versus 8.350 s/token; -2.75% paired median).
- **H17** restricts coalescing to one tensor and an 8 MiB extent. It cut calls
  57.05% with 0.54% byte amplification, but all four cells lost (-18.37%
  paired median). This rules out the Python `bytearray`/view path, not native
  registered-buffer asynchronous I/O.
- **H18** uses Transformers' exact `DynamicCache.crop` to roll target lookahead
  back to the accepted prefix. The randomized L2 screen exercised 16 crops and
  326 reused prefix tokens with identical outputs and +0.49% peak VRAM. Its
  paired effect was -0.59%, 90% interval [-4.62%, +1.09%], so the current
  general treatment stops for futility.

All three remain selectable configurations so their negative results are
reproducible; none replaces the stable fixed-speculation default. Sources and
raw artifacts are linked from
[ALL_HYPOTHESES_AND_BASELINES.md](ALL_HYPOTHESES_AND_BASELINES.md).

Run `afterimage research experiments --json` for the machine-readable H0-H18
registry, or `afterimage research test-plan HYPOTHESIS --json` for its regulated
protocol.

## 22. Repository map

```text
afterimage/experiments.py                 immutable registry + paired tests
afterimage/protocols.py                   L0-L3 protocol registry per hypothesis
afterimage/runtime/critical_path.py       event DAG, replay, measured profile
afterimage/runtime/spec_policy.py         AdaEDL + hazard/cost policy
afterimage/runtime/controllers.py         PI/MPC, LinUCB/TS, change detector
afterimage/runtime/representations.py     exact multiple-choice planner
afterimage/runtime/xor_reference.py       exact dependent artifact codec
afterimage/runtime/certified_mips.py      certified greedy search + fallback
afterimage/bench/multifidelity.py         successive-halving screen
afterimage/server/app.py                  experiment API/jobs/result store
afterimage/server/static/index.html       clickable Experiment Lab
results/README.md                         result schema and publication rules
```
