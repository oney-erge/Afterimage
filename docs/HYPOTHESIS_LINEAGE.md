# Hypothesis lineage

Where each Afterimage hypothesis came from, what was actually borrowed, and
what was changed. **None of these is claimed as a confirmed novel method.** Every
mechanism below adapts published work from speculative decoding, systems
optimisation, control theory, survival analysis, or compression; what is ours is
the specific transfer to lossless compressed weight streaming. The evidence is
mixed: some performance candidates are contradicted, some have only a small
positive direction, some are gated or inapplicable, and one passed its mechanism
gate while regressing end to end. H16-H18 extend the registry with tested
compositions; none has reached L3 confirmation.

Citation status is marked per row:
**[V]** = source independently re-checked against the publisher/arXiv record on
21 August 2026. **[R]** = carried from
[LITERATURE.md](LITERATURE.md) / [RESEARCH_METHODS.md](RESEARCH_METHODS.md)
without a fresh check in that pass.

Verdicts and measured numbers:
[ALL_HYPOTHESES_AND_BASELINES.md](ALL_HYPOTHESES_AND_BASELINES.md).

---

## The gate hypothesis

| ID | Hypothesis | Source | Afterimage adaptation |
|---|---|---|---|
| **H0** | Oracle gap: check first whether different prompts or system states actually *need* different configurations. If the theoretical upside is tiny, adaptive control is not worth building. | **[V]** [BanditSpec](https://arxiv.org/abs/2505.15141) (ICML 2025, PMLR 267:24045) evaluates adaptive speculative-decoding policies against an *oracle best hyperparameter* upper bound. | Used as a **research gate**, not a method. The measured gap is 2.56% against a 12% gate. H3/H8 were later executed as bounded replays anyway: H3 reproduced the baseline and H8 failed simulator calibration, confirming that the gate had identified little usable upside. |

This is the most useful thing in the whole research layer, and it cost one
afternoon: it prevented an entire RL stack from being built on a 2.5% ceiling.

---

## Placement and residency

| ID | Hypothesis | Source | Afterimage adaptation |
|---|---|---|---|
| **H1** | Critical-path residency: keep the weights whose residency actually shortens end-to-end runtime, not the ones that save the most bytes. | **[R]** PyTorch [Holistic Trace Analysis](https://hta.readthedocs.io/en/latest/source/features/lightweight_critical_path_analysis.html) builds CPU/GPU dependency DAGs to find the execution critical path. | Applied to *weight placement*. Each tensor's read/decode/transfer spans are counterfactually zeroed in the measured DAG and the DAG is replayed; its residency value is the makespan reduction, not raw critical-path occupancy. This correctly accounts for a second path becoming critical. |
| **H6** | Exact representation planning: choose a different exact storage/memory representation per tensor under RAM, VRAM and storage limits. | **[V]** [NicePIM](https://arxiv.org/abs/2305.19041) (IEEE TCAD 2024) explores hardware/mapping design spaces for DRAM-PIM DNN accelerators under resource constraints. | The **multiple-choice-knapsack/DP formulation is ours**, not NicePIM's: see the correction note below. Options become exact weight representations: compressed on disk, compressed in RAM, decoded in RAM, VRAM-resident. Lossy options are rejected by construction. |
| **H10** | Replay/CEM residency search: search *complete combinations* of resident tensors rather than scoring each tensor independently. | **[V]** [Cross-Entropy Method](https://doi.org/10.1023/A:1010091220143) (Rubinstein 1999) for combinatorial optimisation; **[R]** [digital-twin-assisted RL](https://arxiv.org/abs/2208.01781) for evaluating decisions inside a simulated system. | A real Afterimage event DAG is the world model. CEM samples budget-feasible *sets*, replays them, keeps elites. The live engine never explores: it validates and executes a frozen plan. Motivated by H1's 1.6%: if bottlenecks switch, the value of A depends on whether B is resident. |
| **H13** | Event-interference QUBO residency: fit a pairwise Hamiltonian over the same counterfactuals and solve it by classical annealing. | **[R]** [QUBO + classical annealing for discrete placement](https://arxiv.org/abs/2009.00140) (Dury & Di Matteo 2020). | Linear coefficients are single-tensor counterfactual makespan reductions; quadratic coefficients are residual pair benefit. Quantum-*inspired*, not quantum. Every candidate is repaired to the byte budget and rescored on the original DAG. |
| **H15** | Physical-extent QUBO residency: make the binary variable a bounded contiguous `weights.bin` span rather than a tensor. | Same as H13, plus the H14 observation that request geometry matters. | Selecting a variable makes every tensor in that span resident, coupling physical layout to residency while keeping the runtime's tensor-level plan artifact unchanged. |

H13 and H15 both returned **exactly their control plan** (`treatment_diverged =
false`). Greedy budget refill was identified as one confound and removed, but a
fresh post-repair run still produced 0% gain and 100% overlap after 730 tensor
and 369 extent evaluations. The current null result therefore survives that
correction.

---

## Speculation

| ID | Hypothesis | Source | Afterimage adaptation |
|---|---|---|---|
| **H2** | Cost-aware speculative stopping: decide whether one more draft token is worth it *before* paying for the large target model. | **[V]** [SpecDec++](https://arxiv.org/abs/2405.19715) (COLM 2025) formulates candidate length as an MDP, proves the optimal policy is a threshold policy, and trains an acceptance-prediction head. **[V]** [AdaEDL](https://arxiv.org/abs/2410.18351) (NeurIPS 2024 ENLSP) derives a training-free entropy lower bound on acceptance for early draft stopping. | Adds the **offload cost explicitly**. Accepted prefixes and first rejections are treated as censored survival data; another token is drafted only when its expected saved target-sweep seconds exceed the marginal draft cost. The distinction matters because a streamed target sweep costs ~14 s, not ~50 ms. |
| **H11** | Neural speculative stopping: *learn* whether proposing another token improves real generation speed. | **[V]** [Gensheimer & Narasimhan](https://arxiv.org/abs/1805.00917) (PeerJ 2019) scalable discrete-time survival models for neural networks; **[V]** SpecDec++ for learned speculative rejection; **[V]** BanditSpec for training-free adaptive configuration. | A six-hidden-unit censored-survival MLP pools confidence × entropy × position instead of maintaining sparse bins, and its decision rule includes the measured cost of another streamed target sweep. The exact verifier is untouched, so a bad prediction costs latency, never correctness. |
| **H3** | Contextual profile selection: choose among *complete* inference setups from the request and machine state. | **[R]** [Conservative Contextual Linear Bandits](https://arxiv.org/abs/1611.06426) selects contextual actions while protecting a known baseline; **[V]** BanditSpec applies bandits to speculative-decoding configurations. | Actions are **complete versioned Afterimage profiles** (residency, speculation, prefetch, representation), not individual decoding knobs. The guard is a practical lower-confidence check, explicitly *not* CLUCB's cumulative high-probability guarantee. |
| **H8** | Simulator-based profile control: evaluate configurations in a model of the system before a controller may choose them live. | **[R]** [Digital-twin-assisted RL for task scheduling](https://arxiv.org/abs/2208.01781); **[V]** [Sibyl](https://arxiv.org/abs/2205.07394) (ISCA 2022) learns data placement across hybrid storage online. | The simulator represents measured SSD → RAM → VRAM → GPU behaviour and the actions are complete exact inference profiles. Shadow-only: it may recommend, never apply, until H0 and H3 pass. They did not. |
| **H16** | Speculation-conditioned critical-path residency: compose the strongest speculative control with the H1 resident set. | [SpecExec](https://arxiv.org/abs/2406.02532) for offloaded speculation; HTA/H1 for critical-path placement. | No source was copied as this combined profile. The resident set changed at matched VRAM, but the four-family screen was slower, so the composition is contradicted. |
| **H18** | Rollback-cached target verification: reuse immutable target context between speculative sweeps. | Hugging Face Transformers [`DynamicCache.crop`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py) is the exact primitive; [SpecInfer](https://arxiv.org/abs/2305.09781) establishes speculative KV reuse. | Adds fail-closed crop-length invariants to the offloaded target. L2 exercised 16 crops and reused 326 prefix tokens with identical outputs, but the paired median was -0.59%; it stops for futility. |

---

## I/O and memory

| ID | Hypothesis | Source | Afterimage adaptation |
|---|---|---|---|
| **H4** | Adaptive prefetch: change how many future layers are loaded ahead based on whether storage is keeping up. | **[V]** [Pythia](https://arxiv.org/abs/2109.12021) (MICRO 2021) adapts hardware prefetch decisions with online RL and a bandwidth-aware reward. | PI / MPC control over prefetch *depth* for streamed compressed LLM layers. Byte-exact: only scheduling changes. Note Pythia's dense per-access reward is exactly what Afterimage lacks: hundreds of items, one wall-clock outcome. |
| **H12** | Bayesian chance-constrained prefetch: pick the smallest depth that is *probably* ready in time. | **[R]** Distribution-driven probabilistic timing constraints in [memory-latency regulation](https://drops.dagstuhl.de/opus/volltexte/2023/18033/pdf/LIPIcs-ECRTS-2023-4.pdf) and [ALERT](https://www.usenix.org/conference/atc20/presentation/wan). | Normal-inverse-gamma posteriors over log read latency and observed per-layer lead window; a probit constraint `P(R ≤ d·C) ≥ target` chooses depth. It never predicts *which* layer is needed: dense execution order is known exactly. Probability decides only how early an exact read starts. |
| **H9** | Liveness-guided lm_head overlay: keep the output head in host RAM and borrow VRAM for it only after transformer weights no longer need that memory. | **[R]** [SuperNeurons](https://web.eecs.umich.edu/~mosharaf/Readings/SuperNeurons.pdf) uses tensor liveness to reuse GPU memory; **[V]** [FlexGen](https://arxiv.org/abs/2303.06865) places LLM tensors across GPU, CPU RAM and disk. | Combines the two: the decoded head stays in pinned RAM, occupies the transformer's *released* VRAM during output projection, then leaves. Peak VRAM is set by the larger of the two phases, not their sum. No implementation of this exact composition was found in the reviewed work. |
| **H14** | Coalesced contiguous storage reads: merge physically adjacent compressed arrays into bounded extents. | **[R]** [Shriver et al. (1999)](https://www.usenix.org/legacy/publications/library/proceedings/usenix99/full_papers/shriver/shriver_html/index.html) on fixed request cost, layout, queueing and compute/I/O overlap. | `weights.bin` already preserves model order, so a layer's blobs are physically adjacent and can be merged with zero byte amplification. Cut read calls 89%. **Went 27.7% slower**, the classic overlap result, measured on this workload. |
| **H17** | Tensor-scoped micro-extents: retain H14's request amortization but never merge across a tensor consumer boundary. | Same [Shriver et al.](https://www.usenix.org/legacy/publications/library/proceedings/usenix99/full_papers/shriver/shriver_html/index.html) request-cost/overlap model. | The tensor boundary is the local repair, not a copied repository specification. Calls fell 57.05% with only 0.54% extra physical bytes, but all four cells lost and the paired median was -18.37%. |

---

## Compression

| ID | Hypothesis | Source | Afterimage adaptation |
|---|---|---|---|
| **H5** | Certified output-head pruning: during greedy decoding, skip vocabulary rows provably unable to win. | **[V]** [Ram & Gray (2012)](https://arxiv.org/abs/1202.6101) exact branch-and-bound MIPS over tree data structures. | Applied to the LLM output head, with **explicit floating-point accumulation bounds** and a full `lm_head` fallback whenever the winner cannot be certified exactly: token equality is stricter than retrieval recall. Sampling always uses the full distribution. |
| **H7** | Expert-to-expert XOR compression: store one MoE expert as a lossless difference from a similar expert. | **[V]** [BitX / ZipLLM](https://arxiv.org/abs/2505.06252) (NSDI 2026) XORs a fine-tuned model against its base and losslessly compresses the delta. | The XOR method is prior work; the **relation** is changed from base → fine-tune to expert → expert inside one checkpoint. Untestable on Qwen3-14B, which is dense. |

The base codec itself is not a hypothesis: bf16 exponent Huffman coding is
**[V]** [ZipNN](https://arxiv.org/abs/2411.05239)'s method, which reports ~33%
savings on BF16 models, consistent with Afterimage's measured 1.453x. And
speculative decoding over an *offloaded* target is **[V]**
[SpecExec](https://arxiv.org/abs/2406.02532) (NeurIPS 2024). Neither is claimed
here.

---

## Corrections found during verification

**H6 / NicePIM.** An earlier draft of this table said NicePIM "uses
multiple-choice knapsack / dynamic programming to choose one implementation
option per component." That is **not what the paper does.** NicePIM's
components are PIM-Tuner, PIM-Mapper, and a **integer-linear-programming**
data scheduler; multiple-choice knapsack is not its formulation. NicePIM is a
legitimate precedent for *constrained design-space exploration over per-component
implementation options*, and that is what it is cited for here. The MCKP/DP
formulation is Afterimage's own choice and must not be attributed to it.

**H9 numbering.** The 21 August bounded report closed by proposing an
*"H9, layout-aware residency"*. That proposal shipped as **H14/H15**, and the H9
slot went to the RAM overlay instead. Anything citing the older numbering is a
forward-looking note, not a description of the shipped H9.

**AdaEDL venue.** AdaEDL is a NeurIPS 2024 workshop paper (Efficient Natural
Language and Signal Processing, PMLR v262), not a main-track paper. SpecDec++
appeared at COLM 2025. Neither changes the argument; both are recorded here so
the citations are not overstated.

---

## What this lineage is for

It exists so that no reader has to take a novelty claim on trust, and so that a
failed hypothesis can be traced to the assumption it inherited. Three of the
failures are direct consequences of borrowing from a domain whose cost
structure does not match this one:

- **H4** inherited Pythia's dense per-access reward. Afterimage has hundreds
  of placement items and one wall-clock outcome per request.
- **H11** inherited a throughput objective where the marginal draft cost is
  ~28 ms against a ~14 s target sweep, making "draft one more" almost always
  correct. It never stopped once.
- **H14** inherited "fewer, larger requests are better" from storage systems
  where I/O is not overlapped with a GPU decode that consumes it.

Those are the useful outputs of this program. They are not results *about*
Afterimage; they are results about what does and does not transfer into an
offloaded, entropy-coded inference regime.
