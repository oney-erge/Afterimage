# Implementation & Test Plan — Adaptive Speculation

Execution plan for [PROPOSAL_ADAPTIVE.md](PROPOSAL_ADAPTIVE.md). Results append
to [RESULTS_LOG.md](RESULTS_LOG.md).

---

## 0. The apples-to-apples rules

These are non-negotiable for every measurement below. They exist because an
earlier comparison in this project reported "~2x faster than AirLLM" while
quietly using 2.66 GB against AirLLM's 1.57 GB — which is not a result.

| # | Rule | Why |
|---|---|---|
| 1 | **Every arm gets the SAME total VRAM budget.** Arms differ only in how they *spend* it. | A faster arm that used more memory has proven nothing. |
| 2 | **Peak VRAM is measured, not assumed** — `torch.cuda.max_memory_allocated()`, same counter, reset between arms. Report it next to every speed number. | Budgets are requests; peaks are facts. |
| 3 | **Cold page cache before every timed run** (`drop_caches`). | Otherwise we measure RAM bandwidth. |
| 4 | **Identical prompt set, identical token count, identical seed** across arms. | |
| 5 | **Correctness asserted every run, not assumed.** See §3. | |
| 6 | **Full field dump** — every measured field to JSON, never a curated subset. | |
| 7 | **End-to-end wall clock decides.** Component benchmarks inform, never conclude. | The CPU-decode lever passed its isolated gate at 1.33 GB/s and was 0.52x end-to-end. |
| 8 | **Baseline = the most recent logged row before the change**, not the oldest. | |

**Rule 1 is the one that shapes the whole design.** The three drafting strategies
have different intrinsic memory costs, so a naive comparison is meaningless. At a
fixed 6.0 GB budget:

| Arm | Draft costs | Left for the residency planner |
|---|---|---|
| **A** plain greedy | 0 GB | 6.0 GB |
| **B** small draft model (today) | ~1.2 GB (Qwen3-0.6B resident) | ~4.8 GB |
| **C** self-draft, N layers pinned | ~N × 0.66 GB | 6.0 − (N × 0.66) GB |

Every arm answers the same question: **what is the best use of exactly 6.0 GB?**

---

## 1. Implementation

All additions are opt-in `EngineConfig` fields defaulting to today's behaviour.

### 1.1 `EngineConfig` (new fields)

```python
draft_mode: str = "none"          # "none" | "model" | "self"
draft_exit_layer: int | None = None   # self-draft: use layers [0, N)
spec_k: int = 8                       # fixed draft length
spec_k_policy: str = "fixed"      # "fixed" | "gamma" | "threshold"
pin_draft_layers: bool = False        # tell the planner draft layers are hot
spec_policy_state: str | None = None  # path to persist bandit state across runs
```

Validation: `draft_exit_layer` required and `1 <= N < n_layers` when
`draft_mode="self"`; `pin_draft_layers` requires `draft_mode="self"` and
`vram_budget_gb`.

### 1.2 `runtime/self_draft.py` (new)

Early-exit forward: run embeddings → layers `[0, N)` → `model.norm` → `lm_head`.
This reuses the model's existing final norm and output head as the exit head,
which is what LayerSkip does — no new parameters, nothing to train.

Two mechanical hazards to handle explicitly:

- **KV cache separation.** The draft's k sequential passes write KV for layers
  `[0, N)`. The verification pass must not consume draft-contaminated cache
  entries for rejected tokens. Simplest correct v1: **draft with `use_cache=False`**
  and accept the recompute cost, then optimise only if profiling says it matters.
- **Layer residency.** The draft runs layers `[0, N)` k times per sweep. Under
  `pin_draft_layers=False` these re-stream k times and the arm will be *slower* —
  that is the expected, informative failure, not a bug.

### 1.3 `runtime/spec_policy.py` (new)

```python
class SpecPolicy:
    def choose_k(self, context) -> int: ...
    def update(self, k_used, n_accepted, sweep_seconds) -> None: ...
    def save(path) / load(path)          # persist across runs
```

- `FixedPolicy` — returns `spec_k`. The control arm.
- `GammaTunePolicy` — EWMA over recent acceptance; expand k when acceptance is
  high, contract when low ([GammaTune](https://arxiv.org/pdf/2504.00030)).
- `ThresholdPolicy` — stop drafting when the draft's own confidence for the next
  token drops below a learned threshold. SpecDec++ proves the optimal policy has
  this shape; using the draft's max-prob as a free proxy for P(reject) avoids
  training an acceptance head.

**Persistence matters more here than in the papers.** A 20-word answer is ~6
sweeps — far too few for a bandit to converge within one run. State is written
per-model to `spec_policy_state` so learning accumulates across runs.

### 1.4 `vram_planner` — draft-layer reuse

Today's ranking assumes uniform use: `value_density = comp_bytes / orig_bytes`,
justified in the module docstring by *"every weight ... is used exactly once per
token."* Self-drafting makes that false. Change the numerator to expected uses:

```
uses(tensor) = k + 1  if tensor is in a draft layer and draft_mode == "self"
               1      otherwise
value_density = (uses * comp_bytes) / orig_bytes
```

Guard: `uses` depends on k, which the bandit now varies. Use the policy's current
mean k at plan time and re-plan only if it drifts >50%, rather than re-solving
the knapsack every sweep.

### 1.5 `scripts/adaptive_bench.py` (new)

One harness, one JSON per run, enforcing §0 mechanically: takes `--vram-total-gb`
and an arm name, subtracts the arm's draft cost, passes the remainder as
`vram_budget_gb`, drops caches, resets peak counters, runs, dumps every field.

---

## 2. Test matrix

Qwen3-14B, RTX 3080 Laptop, 8 prompts × 32 tokens, seed fixed, cold cache.

### T0 — Coupling check *(no new code; runs first)*

Grid `spec_k ∈ {2,4,8,12}` × `vram_budget_gb ∈ {2,4,6}`, existing small-draft mode.

**Question:** does the best k *shift* with budget?
**Kill:** if `argmax_k` is flat across budgets, the knobs are independent — tune
them separately, skip §1.4 entirely. **This is a simplifying result, not a
failure**, and it costs nothing to obtain.

### T1 — Matched-VRAM arm comparison *(the headline)*

Total VRAM held at **6.0 GB**, then repeated at **4.0 GB**.

| Arm | `draft_mode` | Draft VRAM | Planner budget |
|---|---|---|---|
| A | `none` (greedy) | 0 | 6.0 |
| B | `model` (Qwen3-0.6B) | 1.2 | 4.8 |
| C4 | `self`, N=4, pinned | 2.6 | 3.4 |
| C8 | `self`, N=8, pinned | 5.3 | 0.7 |

**Report per arm:** measured peak VRAM, s/token, words/sweep, acceptance %,
GB/token, io/decode/compute split, and the generated text.
**Pass:** some arm beats A by ≥2x at equal measured peak VRAM.
**Kill for C:** if no self-draft arm beats B at equal VRAM, the 1.2 GB small
model is the better buy — report that and stop.

### T2 — Adaptive vs best fixed k

Best arm from T1, same VRAM. `spec_k_policy ∈ {fixed@best_k, gamma, threshold}`.
Report mean **and variance across prompts** — GammaTune's variance reduction
matters as much as its mean at ~15 s/sweep.
**Kill:** if neither adaptive policy beats the best *tuned constant* by >5%, ship
the constant and record that adaptivity wasn't justified.

### T3 — Pinning ablation

Best self-draft arm, `pin_draft_layers` on vs off, same total VRAM.
**Expected:** off is *slower* (draft layers re-stream k times). This is the
mechanism check for §1.4 — if pinning doesn't matter, the coupling claim in
PROPOSAL_ADAPTIVE §3C is wrong and should be retracted.

### T4 — Final head-to-head vs AirLLM

Winning configuration vs AirLLM at **matched measured peak VRAM** (the 1.62 vs
1.57 GB protocol already used). Same prompt, same counter, worked examples shown
as prompt → answer → wall time.

---

## 3. Correctness — how the speculative arms stay apples-to-apples

Speculative decoding *samples*; greedy is deterministic. Comparing a sampled run
against a greedy run token-for-token is invalid, and this project has already
been careful about that distinction.

**The clean resolution: run every timed benchmark at temperature 0.**

At T=0 the target distribution is one-hot at its argmax. If the draft proposes
the argmax, `p/q = 1` and it is accepted. If it proposes anything else,
`p_target = 0`, so it is rejected and the residual `(p − q)₊` renormalises to
one-hot at the argmax. **Either way the emitted token is the target's argmax** —
so speculative decoding at T=0 provably reproduces plain greedy, exactly.

That gives, for free:
- **A real correctness assertion for every arm**: token-identical to plain greedy.
- **A genuinely like-for-like speed comparison** — same decoding mode, same
  output, different mechanism. Only speed differs.

Sampling behaviour keeps its existing separate validation
(`tests/test_verify.py`'s distributional tests); it is not what these benchmarks
measure.

**Additional gates:** `pytest` green before any timed run; self-draft logits
asserted bit-identical to a full forward over the same layer prefix.

---

## 4. Order, and what each step can kill

| # | Step | Effort | Kills what, if it fails |
|---|---|---|---|
| 1 | **T0 coupling grid** | S — no code | Kills §1.4 (planner change) |
| 2 | `self_draft.py` + T1 | M | Kills self-drafting; keep small draft model |
| 3 | T3 pinning ablation | S | Kills the coupling claim |
| 4 | `spec_policy.py` + T2 | M | Kills adaptivity; ship a tuned constant |
| 5 | T4 head-to-head | S | — |

Nothing after step 1 begins until step 1 answers. Each step's kill criterion is
written before it runs.

**One expected outcome worth pre-committing to:** at a 2 GB budget, 8 pinned
draft layers (~5.3 GB) cannot fit. **Self-speculation is probably unaffordable at
the smallest budgets**, and if T1's 4.0 GB repeat confirms that, the honest
report is "self-drafting is a large-budget technique; small budgets keep the
1.2 GB draft model" — not a tuned-around failure.
