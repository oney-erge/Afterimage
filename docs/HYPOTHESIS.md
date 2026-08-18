# Restriction — Workload-Adaptive Model Compression for Offloaded Inference

**Status:** pre-implementation hypothesis. Nothing measured. **v3** — revised
against classical theory; two of v2's load-bearing claims did not survive.

**Goal:** run a dense ~27B model interactively on 8 GB VRAM, on ordinary hardware.

> **One line:** don't cache weights — cache what the weights *did*.

---

## 1. Prior art (unchanged from v1/v2; still the binding constraint)

| Work | Contribution | Tier |
|---|---|---|
| AirLLM | Layer-at-a-time; minimal peak VRAM | Disk |
| FlexGen | Cost-model placement | RAM + disk |
| SpecExec (2024) | Tree speculation, ~20 tok/target pass | RAM **or SSD** |
| SpecOffload (2025) | Draft in idle VRAM; 2.54× | CPU RAM |
| SubSpec (NeurIPS 2025) | Low-bit substitute draft from offloaded layers; 9.1× @ 8 GB | CPU RAM |
| ATSInfer (2026) | Tensor-granularity placement; 3.29× decode | CPU RAM |
| Gated Subspace Inference (2026) | Subspace/residual split with a gate — **offline basis, VRAM bandwidth, lossy** | n/a |

The original four-pillar plan maps one-to-one onto this table. **SpecExec already
covers SSD** and already reports that longer offload paths need larger draft
budgets. **GPUDirect Storage does not exist on GeForce** — the path is necessarily
NVMe → pinned host buffer → PCIe → VRAM. The original plan is a re-implementation.

---

## 2. The reframing that survives

Everyone optimises **bytes per token**. Wrong denominator.

A weight matrix is a *linear operator*. What a fetch buys is not "one token" — it
is the map restricted to the activation directions pushed through it. Collinear
activations extract identical value from one fetch.

> **The cost unit is bytes per newly-explored activation direction.**

Speculative decoding amortises a fetch over `k` tokens, then **discards the
operator and re-reads it next sweep**. Over 500 tokens at `k=15`, the same 8.5 GB
is read ~33 times: 8.5 GB of information, 280 GB of traffic. Nobody retains a
fetch's value across sweeps.

**Mechanism.** Per offloaded linear layer keep resident a basis `U ∈ ℝ^{d_in×r}`
and its image `M = W U`. For activation `x`: `c = Uᵀx`, `x⊥ = x − Uc`. On a hit,
`y = Mc` with **zero I/O**. On a miss, fetch `W`, compute `y = Mc + W x⊥` exactly,
then install `u = x⊥/‖x⊥‖` into `U` and `W u` into `M`.

Three properties justify this shape:

- **Fills are free.** The one moment `W` is in VRAM is a miss, so extending the
  sketch costs one matvec and **zero extra I/O**. No calibration, no training.
- **It is information-theoretically optimal.** For a linear operator queried only
  within an `r`-dimensional subspace, `W U` is a minimal sufficient statistic.
  Transformers apply weights only linearly (attention's nonlinearity involves the
  KV cache, not weights), so this covers ~100% of weight bytes.
- **It is self-improving.** Miss rate falls monotonically. Quantization,
  sparsity, and speculation cost the same forever; this gets cheaper.

**Why it was missed:** Gated Subspace Inference has the decomposition but targets
VRAM bandwidth, where `W` is already resident and skipping the residual saves only
FLOPs. Offloaded, that residual costs a *disk fetch* — ~1000× more per byte. Same
gate, ~1000× the value. And GSI's stated weakness (offline calibration degrades
OOD) is what an online basis fixes: OOD is a miss, a miss computes exactly, and
the miss enlarges the basis. **Being wrong self-corrects instead of silently
degrading.**

---

## 3. What classical theory says — three findings, two of them negative

### 3.1 FATAL to v2's gate: the novelty ratio is the wrong metric

v2 gated on `ρ = ‖x⊥‖/‖x‖`. That is mis-specified, and the literature is
unambiguous about why. LLM hidden states are dominated by a few **massive
activations / rogue dimensions** whose variance exceeds the rest by orders of
magnitude — see *Massive Activations in LLMs* and, canonically, *All Bark and No
Bite: Rogue Dimensions in Transformer LMs Obscure Representational Quality*.

**`‖x‖` is dominated by directions carrying almost no functional information.** A
basis capturing the rogue dimensions scores `ρ ≈ 0` while missing everything that
distinguishes one token from another. The gate reads "hit" precisely when it is
functionally wrong. Any variance- or norm-based criterion fails here; this is the
same trap *Variance Is Not Importance* (arXiv:2604.20682) documents.

**Fix — gate in output space, cheaply.** Store per matrix a Johnson–Lindenstrauss
sketch `G W` with `G ∈ ℝ^{m×d_out}` Gaussian, `m ≈ 32`. Then `‖GWx⊥‖·√(d_out/m)`
estimates `‖W x⊥‖` to relative error `O(1/√m)` — textbook randomized NLA
(Halko–Martinsson–Tropp). Cost: `m·d_in` ≈ **0.3 MB per matrix**, negligible
against a rank-128 sketch.

Then allocate a *global* error budget. With `s_ℓ` a calibrated layer-to-logit
sensitivity, minimise I/O subject to `Σ_ℓ s_ℓ η_ℓ ≤ ε`. The Lagrangian gives the
optimal control law directly:

```
fetch layer ℓ  ⟺  s_ℓ · ‖W x⊥‖ > λ
```

**One global knob `λ`, provably the right allocation across layers** — water-
filling, not a per-layer heuristic. This is strictly better than v2 and it is
forced by the theory rather than invented.

### 3.2 PROVED CLOSED: the dictionary escape cannot work

The textbook escape from a slowly-decaying Kolmogorov `n`-width is to abandon
linear subspaces for a nonlinear or dictionary representation (nonlinear-manifold
ROM, quadratic manifolds, clustered POD). And the evidence says LLM activations
really are dictionary-structured rather than low-rank: **superposition** holds
that models represent more features than they have dimensions using near-
orthogonal directions, and SAE work recovers exactly that — overcomplete
dictionaries of size `16–64 × d` with `L0 ≈ 20–100`.

So the natural move is to cache `W D` for an overcomplete dictionary `D` and
sparse-code each activation against it. **This is impossible, and the argument is
two lines:**

```
storage(W D) = d_out × K      must satisfy  K < d_in   to beat storage(W)
sparse coding requires        K > d_in                 (overcomplete)
```

**The compression requirement and the sparse-coding requirement are
contradictory.** You cannot cache a linear operator's action on an overcomplete
dictionary more cheaply than storing the operator. The escape hatch is closed by
arithmetic, not by engineering difficulty.

**Consequence: the sketch is forced to be a genuine low-rank subspace, and the
Kolmogorov barrier applies at full strength.** This is the central risk, and
theory made it *worse*, not better, than v2 assumed.

### 3.3 CONSTRUCTIVE: clustering converts capacity into bandwidth

The remaining textbook escape is **piecewise/clustered subspaces** — `K` local
bases of rank `r`, selected by context. Total capacity `K·r` is unchanged, but a
token touches only one cluster, so **per-token bandwidth falls `K`-fold**.

That is the correct shape when the sketch lives in RAM or NVMe and bandwidth
binds — exactly the 8 GB case. It also matches the empirical structure:
activations cluster by topic, syntax role, and token type. Select the cluster by
nearest centroid, a `d·K` dot product, negligible.

### 3.4 The one piece of good news: the residual stream damps error

`x_{ℓ+1} = x_ℓ + F_ℓ(x_ℓ)`. Approximation errors enter **additively** into the
residual stream rather than compounding through a product of Jacobians. Standard
forward-error analysis then gives `√L ≈ 8` growth over 62 layers under
independence, instead of the catastrophic multiplicative blow-up that killed the
v1 interval-arithmetic certificate. **This is a transformer-specific structural
reason the approach is not doomed**, and it is why a per-layer tolerance around
`10⁻³` may suffice.

### 3.5 Batched fills — and the fix for the speculation conflict

Caching theory: when a fill is ~1000× cheaper than the fetch that enabled it,
**never install rank-1.** On a miss, `W` is momentarily in VRAM — install a
rank-`b` update.

Where do the extra directions come from? **The draft model's predicted future
hidden states.** A SubSpec-style substitute draft already produces approximate
activations for upcoming positions at no I/O cost.

> The drafter's job changes from *proposing tokens* to *proposing activation
> directions to prefetch.*

This resolves v2's worst structural worry. v2 feared that a `k`-wide draft tree
makes a layer fetch likely if *any* candidate misses, so speculation and sketching
would fight. Under batched fills they compose constructively: the tree's candidate
activations are precisely the directions worth installing during a miss. Evict by
**LFU, not LRU** — atom usage is heavy-tailed.

### 3.6 Roofline check

A 1.6–3 GB resident sketch, read once per token at ~270 GB/s VRAM, costs 6–11 ms
→ a **90–170 tok/s ceiling**. So the method converts an NVMe-bound problem into a
VRAM-bandwidth-bound one:

> **Steady state runs at the speed of a ~3B model, with 27B quality on the
> directions it has learned.**

Sketch arithmetic (verify against the real checkpoint before trusting):

| rank `r` | sketch/layer | model (~62 layers) | compression |
|---|---|---|---|
| 128 | ~26 MB | ~1.6 GB | ~10× |
| 256 | ~51 MB | ~3.2 GB | ~5× |

### 3.7 The precision corollary (where v1's bit-ladder belongs)

On a miss you compute `W x⊥` where `‖x⊥‖ = ρ‖x‖`, so relative error `ε` there
contributes only `ρε` to `y`:

```
b_eff = b_full − log₂(1/ρ)
```

At `ρ=0.1`, 3.3 fewer bits. **Even a miss is cheap** — fetch a coarse bit-plane,
not all of `W`. Store `W` on NVMe as a residual ladder (Any-Precision LLM,
AnyBCQ, RRQ all build this). v1's bit-ladder is not a second idea; it is *forced*
by the decomposition, under one principle: **allocate bits in proportion to a
term's contribution.**

---

## 4. The hypothesis in one equation

Define the **closed-loop online hit curve** `H_ℓ(r, λ)`: the fraction of (layer,
token) pairs served from a *running* rank-`r` basis under the output-space gate of
§3.1. Then:

```
GB per token = Σ_ℓ (1 − H_ℓ(r,λ)) · size(ℓ) · b_eff/b_full
```

Everything claimed is a statement about `H`, and `H` is measurable **with forward
hooks and numpy — no CUDA kernels, no inference engine, one afternoon.**

---

## 5. Experiments

**Experiment 0 — the n-width spectrum. Do this first; it is the kill switch.**
Hook every linear layer of Gemma-3-27B or Qwen3-32B. Run real generations. For
each layer compute the singular value spectrum of the activation matrix **and the
decay of the *functional* error `‖W x⊥‖`** as a function of rank. Report both; the
gap between them is the rogue-dimension effect of §3.1 and will be large.

- *Go:* functional error at `r = 128–256` is below the per-layer tolerance.
- *No-go:* it decays like `r^{-α}` with small `α` — the Kolmogorov barrier —
  in which case §6 applies.

**Experiment 1 — measure `H` closed-loop.** The sketch perturbs layer `ℓ`'s
output, changing layer `ℓ+1`'s *input* and therefore its decomposition. **`H` must
be measured with the approximation active and errors propagating.** Open-loop
measurement on a clean forward pass gives an optimistic number the runtime will
never reproduce. This is the easiest way to fool yourself here.

**Experiment 2 — calibrate `λ` functionally.** Against logit change, never
captured variance. Use conformal calibration on held-out data for a
distribution-free bound on token disagreement.

**Experiment 3 — clustering payoff.** Measure `H(K, r)` for `K ∈ {1,4,16}` at
fixed `K·r`. Does the union of local subspaces beat one global subspace of the
same total budget?

**Experiment 4 — batched fills.** Does installing the draft tree's predicted
directions on each miss raise `H` faster than rank-1 installs? Measure
tokens-to-steady-state.

**Experiment 5 — end-to-end.** Only after 0–4 pass.

### Success thresholds

| Metric | Target |
|---|---|
| Model / VRAM | dense ~27B Q4 / 8 GB physical |
| CPU transformer FLOPs | 0 |
| Steady-state `H` | ≥ 0.7 at `r ≤ 256` (or `K·r ≤ 4096` clustered) |
| Token disagreement vs. exact | ≤ 0.1%, conformally bounded, audited online |
| GB/accepted token vs. AirLLM baseline | ≥ 5× from sketching alone |
| Tokens to steady state | ≤ 200 |

---

## 6. Honest verdict — v4, after chasing the actual measurements

### 6.1 Correction to v3

v3 cited two claims supporting low-rank activations: that they "occupy subspaces
an order of magnitude smaller than model width," and that "under 10% of
dimensions carry nearly all variance." **Both came from search-engine summaries,
not from the papers.** Fetching arXiv:2511.21594 directly: it contains neither
claim. Its only concrete number is a *preprocessing choice* — LLaMA hidden states
reduced 4096 → 512 for visualization — with variance retention unreported.

Do not build on those numbers. They were not real.

### 6.2 What the real measurements say

| Source | Measurement | Implication |
|---|---|---|
| Dimensional Collapse in Attention Outputs (arXiv:2508.16929) | Attention outputs ≈ **60%** effective rank; **MLP outputs and residual streams ≈ 90%** | **Bad.** The residual stream is precisely the *input* to every linear layer |
| Mixtures of Subspaces (arXiv:2606.16384) | Power-law spectral decay; `r = d/4` captures "almost all relevant activation content" | ~**4×**, not 10–40× |
| Small Singular Values Matter (random-matrix analysis) | In LLaMA/Pythia the **smallest** singular values can be the *second most important decile* | Confirms §3.1 — variance ranking is not importance ranking |
| ASVD / IO-SVD | Activation-aware low-rank compression achieves only **10–30%** | The *composed* object `W·(activation distribution)` is also not very low-rank |

**The single worst finding:** ~90% effective rank for residual streams. That is
the object feeding every linear layer. A basis capturing it needs `r ≈ 0.9d`,
which is no compression at all.

**The mitigating argument, and it is real:** these are corpus-level measurements.
The cache is **per-session**. Within one conversation the activation rank is
bounded by sequence length and plausibly far lower than across a corpus. Nobody
has measured *within-session* effective rank — which is exactly why Experiment 0
is genuine new measurement rather than a re-derivation.

**The counter to the mitigation:** within a `T`-token session the rank is at most
`T` trivially, so the cache helps only if effective rank `≪ T`. If it is `~0.5T`
you miss half the time and gain almost nothing.

### 6.3 Revised expected payoff

| Lever | Status | Expected |
|---|---|---|
| Speculation (SubSpec-class) | Published, measured, reproducible | **9–12×** |
| Residency | Published | ~1.6× |
| Subspace cache (this doc) | Unmeasured; evidence now unfavourable | **1.3–4×** |
| Precision escalation (§3.7) | Unmeasured | 1.3–2× |

v2 and v3 implied 10×+ from sketching. **The evidence does not support that.**
Realistic expectation is 1.5–3×, and it could be ~1×.

**Graceful degradation is the one structural comfort.** Even a weak cache still
pays via §3.7: capturing 75% of energy gives `ρ=0.5` → 1 bit saved on the fetch;
94% gives `ρ=0.25` → 2 bits. The design does not fall off a cliff, it just
delivers less.

### 6.4 Recommendation

**Ship the known-good stack, and treat this as a cheap side bet.**

1. Build SubSpec + SpecExec + residency against an NVMe tier. That is the
   27B-on-8GB product. It is engineering, not research, fully supported by
   published results, and delivers the actual goal.
2. Run Experiment 0 in parallel — one afternoon, hooks and numpy. It measures
   within-session effective rank, which nobody has published. If it comes back
   favourable, the cache is a real contribution layered on a working system. If
   not, nothing was lost.

Do not invert this order. The novel idea is now the speculative part of the plan,
not the foundation.

**Remaining risks if Experiment 0 passes:** VRAM accounting is tight (sketch + KV
+ activations + buffers in 8 GB, likely forcing `r=128`); incremental
orthogonalisation is `O(d·r)` per miss and numerically delicate over thousands of
updates (modified Gram–Schmidt with reorthogonalisation, monitor `‖UᵀU − I‖`);
cluster selection may thrash on topic switches.

---

## 7. Positioning

> **A weight fetch's value should be retained across decode steps in compressed
> form, in the subspace the workload actually visits, gated in output space by a
> global error budget, with exact weights on disk as a backstop and each miss
> paying to enlarge the cache.**

Composes with SubSpec and SpecExec rather than competing, so the baseline is
known-good and the contribution is one isolated factor that can be switched off
and measured.

---

## 8. Sources

SpecExec arXiv:2406.02532 · SpecOffload arXiv:2505.10259 · SubSpec
arXiv:2509.18344 · ATSInfer arXiv:2607.10183 · Gated Subspace Inference
arXiv:2605.03109 · ASVD arXiv:2312.05821 · IO-SVD arXiv:2605.15626 · Variance Is
Not Importance arXiv:2604.20682 · Massive Activations in LLMs (ICML 2024) · All
Bark and No Bite (rogue dimensions) · RRQ arXiv:2608.04048 · AnyBCQ
arXiv:2510.10467 · MARS arXiv:2601.15498

Textbook grounding: Trefethen & Bau / Golub & Van Loan (low-rank, incremental
SVD, Gram–Schmidt stability); Halko–Martinsson–Tropp and Woodruff (randomized
sketching, JL norm estimation); Higham (forward error analysis); Cover & Thomas
(rate–distortion); Borodin & El-Yaniv (competitive caching, LFU vs LRU); Pinkus,
*n-Widths in Approximation Theory* (the Kolmogorov barrier); Benner/Rozza et al.
on clustered and nonlinear-manifold ROM.
