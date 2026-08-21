# Configurable Research Program: H0-H8

Status: implemented research harness. The first bounded hardware screen is
reported in [BOUNDED_RESEARCH_REPORT_2026-08-21.md](BOUNDED_RESEARCH_REPORT_2026-08-21.md);
it does not satisfy the confirmatory repeat counts below.
Survey cutoff: 21 August 2026.

This document turns the research plan into falsifiable, opt-in experiments. The
existing engine remains the control. Every candidate is a named method profile,
every result is written once with its configuration and environment, and every
method has a kill criterion. A method being implemented does **not** mean it is
faster.

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

| ID | Candidate | Control | Runtime status | Exactness contract |
|---|---|---|---|---|
| H0 | joint semantic/system oracle | best global profile | dataset runner | depends on logged profiles |
| H1 | event-DAG critical-path residency | traffic-density residency | live, opt-in | reference execution equivalent |
| H2 | censored rejection-hazard/cost stopping | tuned fixed draft length | live, opt-in | target-distribution exact |
| H3 | baseline-guarded LinUCB profile choice | fixed global profile | offline replay | profiles retain their own contract |
| H4 | PI prefetch depth (MPC ablation available) | fixed depth | live, opt-in | reference execution equivalent |
| H5 | roundoff-bounded MIPS with full fallback | full LM head | live, greedy only | greedy-token exact |
| H6 | per-tensor exact representation DP | uniform representation | artifact planner | weight exact |
| H7 | expert-local XOR reference artifact | independent coding | artifact codec/audit | bit exact |
| H8 | calibrated simulator profile choice | H3 contextual controller | shadow/offline only | profiles retain their own contract |

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

## 5. H0 — joint oracle-gap diagnostic

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

## 6. H1 — critical-path residency

Generate raw steady-state traces without changing the default planner:

```bash
afterimage run MODEL PROMPT --vram-budget-gb 4 \
  --trace-output traces/control-01.json --max-new-tokens 32
afterimage profile-trace traces/control-*.json \
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
baseline-minus-replayed makespan—not raw critical-path occupancy—is its placement
value. This correctly accounts for a second path becoming critical. Profiling mode
synchronizes CUDA around preparation spans; it is intentionally intrusive, so
the profile must be learned on separate runs and evaluated with tracing off.

Primary gate: at least 8% paired throughput gain. Mechanism gates: held-out
predicted-versus-observed rank correlation ≥0.8 and no exact-token mismatch.
Evaluate at no fewer than three VRAM budgets; a ranking that wins at only one
hand-picked budget is not a general result.

## 7. H2 — censored rejection hazard plus cost

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

## 8. H3 — contextual selection among complete profiles

Actions are complete, versioned profiles—not arbitrary knob combinations. The
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

## 9. H4 — feedback-controlled prefetch

PI control observes whether the demanded layer was ready and adjusts bounded
queue depth. The `mpc-prefetch-v1` ablation estimates layer bytes, bandwidth,
compute time, and exposed wait, then minimizes predicted stall plus an overfetch
penalty. Both preserve exact bytes; H4's preregistered primary candidate is PI.

Test fixed depths `{0,1,2,4,8}` first; the control is the best fixed depth, not
the repository default. Replay bandwidth perturbations or run controlled I/O
contention. Report throughput, exposed wait, hit rate, and peak host memory.
Gate: ≥5% throughput and ≥10% exposed-stall reduction.

## 10. H5 — certified greedy output head

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

## 11. H6 — exact physical representation planning

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

## 12. H7 — expert-local lossless reference coding

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

## 13. H8 — model-based control, conditionally

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

## 14. Repository map

```text
afterimage/experiments.py                 immutable registry + paired tests
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

## 15. H9-H11 — new adjacent-domain methods

H9-H11 extend this plan without changing H0-H8:

- H9 overlays the exact output head from pinned RAM only during its live range;
- H10 learns a frozen whole-set residency action with CEM in an event-DAG
  digital twin;
- H11 trains a six-hidden-unit censored-survival model to stop speculative
  drafting by expected throughput.

Their literature boundaries, falsifiable gates, commands, and ordering are in
[NOVEL_METHODS_2026-08-21.md](NOVEL_METHODS_2026-08-21.md). All three are
implemented but unconfirmed; they remain experimental until held-out paired GPU
measurements pass those gates.

Run `afterimage experiments --json` for the machine-readable H0-H11 registry.
