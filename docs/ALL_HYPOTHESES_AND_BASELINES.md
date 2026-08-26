# All hypotheses, baselines, results, and ranking

**Last reevaluated:** 2026-08-22
**Primary hardware:** NVIDIA RTX 3080 Laptop GPU, 8 GB
**Primary model:** Qwen3-14B, original BF16 weights
**Cross-family checks:** Phi-4 Mini 3.8B and Mistral Small 24B
**Scope:** cold page cache, forced greedy decoding, four prompt families unless a row says otherwise

This is the controlling results document. It covers every registered hypothesis
H0-H18, the stable Afterimage configurations, AirLLM 3.2.0, and Hugging Face
Accelerate disk offload. A result is not called “failed” merely because an
upstream gate prevented a meaningless timing comparison.

The Qwen3-14B AirLLM anchor was refreshed from 3.1.0 to 3.2.0 on 2026-08-26
(28.861 to 26.828 s/token, AirLLM's own 7.04% speedup, no Afterimage number
changed); every ratio below reflects the current anchor. See
[RESULTS_LOG.md](RESULTS_LOG.md#2026-08-26-airllm-baseline-refresh-to-320)
for the refresh protocol and raw comparison. The Phi-4 Mini and Mistral Small
24B AirLLM numbers referenced later in this document are from the 2026-08-22
cross-family campaign and still use AirLLM 3.1.0; they have not been rerun.

## Answer in one paragraph

Afterimage is **not better at the exact low-VRAM floor**: it uses 1.723 GB and
runs at 0.83x AirLLM. At about 4 GB, exact residency reaches 1.55x AirLLM but is
0.82x the speed of Hugging Face Accelerate's 3.80 GB hybrid. Afterimage's best
configuration is fixed speculative decoding: **9.150 s/token, 2.93x AirLLM and
1.56x Accelerate at 3.813 GB**, with identical greedy token IDs in the tested
suite. Of the H0-H18 ideas, H9 is the strongest measured lead (+41.4%
throughput at matched VRAM on a smaller model), H1 is the strongest positive
14B live result (+1.61%), and H6 has the largest predicted 14B upside (38.56%)
but still lacks a held-out live execution. H16/H17 regressed; H18's exact KV
rollback worked but stopped for L2 futility at -0.59% paired throughput. No
H1-H18 candidate has L3 confirmatory superiority evidence.

The subsequent [cross-family and scale campaign](CROSS_MODEL_BENCHMARK_2026-08-22.md)
does not change the Qwen ranking. It shows where it transfers: the lossless
store stays at 1.45–1.49x across three families, H12/H14/H17 remain negative,
and certified MIPS becomes the strongest 24B Afterimage point (1.66x AirLLM,
9.52% behind Accelerate). AirLLM still wins the exact minimum-VRAM boundary.

## Stable systems and external baselines

These are complete four-family × four-token runs. “Versus” is throughput ratio,
so values above 1.00x are faster. Accelerate was allowed 4.0 GB and actually
peaked at 3.800 GB; its map put 2 modules on CUDA, 9 on CPU, and 33 on disk.

| Overall rank | System/configuration | Peak VRAM | Seconds/token | vs AirLLM | vs Accelerate | Output contract | What the result means |
|---:|---|---:|---:|---:|---:|---|---|
| **1** | **Afterimage + fixed `k=8` speculation** | 3.813 GB | **9.150** | **2.93x** | **1.56x** | Greedy-token exact at T=0; distribution-exact sampler at T>0 | Best measured operating point; the ~3x result is real and was never invalidated by H2/H11. |
| **2** | Hugging Face Accelerate BF16 GPU/CPU/disk offload | 3.800 GB | **14.318** | **1.87x** | 1.00x | Same BF16 checkpoint; same four greedy token IDs | Best non-speculative 4 GB result in this repository's runs. |
| **3** | Afterimage exact + 4 GB residency | 3.934 GB | 17.360 | **1.55x** | 0.82x | Reference-execution equivalent | Faster than AirLLM, slower than Accelerate at roughly matched VRAM. |
| **4** | AirLLM 3.2.0 | **1.583 GB** | 26.828 | 1.00x | 0.53x | Same BF16 checkpoint; corrected forced-greedy run | Best measured exact low-VRAM external baseline. |
| **5** | Afterimage chunked head | **0.901 GB** | 29.606 | 0.91x | 0.48x | **Approximate** matmul; tested token IDs agreed | Lowest VRAM and near AirLLM speed, but not an exact method. |
| **6** | Afterimage exact minimum-memory | 1.723 GB | 32.514 | 0.83x | 0.44x | Reference-execution equivalent | Compression saves storage, not enough runtime to beat AirLLM at the floor. |

The exact minimum cannot honestly be advertised below AirLLM's measured VRAM:
Qwen3-14B's full BF16 output head is 1.556 GB before transient scratch. Computing
the head in blocks reaches 0.901 GB, but changes BF16 matmul rounding. That is
why “just match lower VRAM” is not a missing flag.

## Source and lineage for every hypothesis

“Adapted” means the cited mechanism is established but its use in this exact
compressed/offloaded path is Afterimage's composition. “Afterimage formulation”
means no cited paper is being credited with the specific formulation. These are
closest-prior links, not novelty or priority claims.

| ID | Closest paper / repository source | What was taken, and what was not |
|---|---|---|
| H0 | [BanditSpec](https://arxiv.org/abs/2505.15141) | Borrowed the oracle-best-hyperparameter comparison; Afterimage uses it as a stop-before-RL research gate. |
| H1 | [PyTorch Holistic Trace Analysis critical path](https://hta.readthedocs.io/en/latest/source/features/lightweight_critical_path_analysis.html) | Borrowed event-DAG critical-path analysis; counterfactual tensor-residency value is the Afterimage adaptation. |
| H2 | [SpecDec++](https://arxiv.org/abs/2405.19715), [AdaEDL](https://arxiv.org/abs/2410.18351) | Borrowed threshold/acceptance-aware stopping; added censored survival and the measured cost of a streamed target sweep. |
| H3 | [Conservative contextual linear bandits](https://arxiv.org/abs/1611.06426), [BanditSpec](https://arxiv.org/abs/2505.15141) | Borrowed safe contextual selection; actions are complete versioned Afterimage profiles. |
| H4 | [Pythia](https://arxiv.org/abs/2109.12021) | Borrowed feedback-aware prefetch control; PI/MPC control of compressed-layer lookahead is the transfer. |
| H5 | [Exact tree-based MIPS](https://arxiv.org/abs/1202.6101) | Borrowed branch-and-bound MIPS; floating-point certificates plus an exact full-head fallback are Afterimage-specific. |
| H6 | [NicePIM](https://arxiv.org/abs/2305.19041) | Constrained per-component physical design is precedent; the multiple-choice exact-representation DP is an **Afterimage formulation**, not NicePIM's algorithm. |
| H7 | [BitX / ZipLLM](https://arxiv.org/abs/2505.06252) | XOR-reference coding is prior work; expert-to-expert coding inside one MoE checkpoint is the transfer. |
| H8 | [Digital-twin-assisted RL](https://arxiv.org/abs/2208.01781), [Sibyl](https://arxiv.org/abs/2205.07394) | Borrowed simulated/shadow policy evaluation for system actions; complete exact inference profiles are the Afterimage action space. |
| H9 | [SuperNeurons](https://web.eecs.umich.edu/~mosharaf/Readings/SuperNeurons.pdf), [FlexGen](https://arxiv.org/abs/2303.06865) | Borrowed liveness reuse and tiered LLM placement; the late-live exact pinned output-head overlay is the candidate composition. |
| H10 | [Cross-Entropy Method](https://doi.org/10.1023/A:1010091220143) | CEM is prior work; searching complete resident sets in an exact measured event-DAG replay is the adaptation. |
| H11 | [Discrete-time neural survival](https://arxiv.org/abs/1805.00917), [SpecDec++](https://arxiv.org/abs/2405.19715) | Borrowed survival modelling and learned speculative stopping; the tiny cost-aware pooled model is the transfer. |
| H12 | [ALERT](https://www.usenix.org/conference/atc20/presentation/wan), [probabilistic memory-latency regulation](https://drops.dagstuhl.de/opus/volltexte/2023/18033/pdf/LIPIcs-ECRTS-2023-4.pdf) | Borrowed chance-constrained timing; the posterior/probit prefetch-depth controller is the adaptation. |
| H13 | [QUBO placement with classical annealing](https://arxiv.org/abs/2009.00140) | Borrowed the QUBO/annealing form; event-DAG counterfactual interaction coefficients are Afterimage's formulation. |
| H14 | [Shriver et al. on prefetching, request cost, and overlap](https://www.usenix.org/legacy/publications/library/proceedings/usenix99/full_papers/shriver/shriver_html/index.html) | Borrowed contiguous-request amortization; the `weights.bin` layer-wide implementation is local and was contradicted. |
| H15 | [QUBO placement](https://arxiv.org/abs/2009.00140), [H14 storage model](https://www.usenix.org/legacy/publications/library/proceedings/usenix99/full_papers/shriver/shriver_html/index.html) | Combines prior QUBO search with physical extents; the extent variable and exact repair/replay are local. |
| H16 | [SpecExec](https://arxiv.org/abs/2406.02532), [HTA critical path](https://hta.readthedocs.io/en/latest/source/features/lightweight_critical_path_analysis.html) | Both ingredients are prior work; their frozen same-budget composition is Afterimage's test, not copied from another repository. |
| H17 | [Shriver et al.](https://www.usenix.org/legacy/publications/library/proceedings/usenix99/full_papers/shriver/shriver_html/index.html) | Request batching/overlap is prior work; making the tensor the hard coalescing boundary is Afterimage's H14 repair. |
| H18 | [Transformers `DynamicCache.crop`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py), [SpecInfer](https://arxiv.org/abs/2305.09781) | The runtime uses Transformers' exact cache-crop primitive and established speculative KV reuse; the offloaded-target rollback integration is local. |

The fuller correction notes, including the earlier NicePIM over-attribution, are
in [HYPOTHESIS_LINEAGE.md](HYPOTHESIS_LINEAGE.md).

## One large H0-H18 comparison table

“vs Air/HF” is only populated for a real end-to-end generation timing. Offline
or mechanism rows are compared with their registered control instead of being
forced into a meaningless external speed ratio.

| Research rank | ID | Method / hypothesis | Candidate and registered control | Evidence actually run | Candidate result | Control result | Effect versus control | vs AirLLM | vs HF Accelerate | VRAM / exactness | Final decision |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| **1** | **H9** | Pinned-RAM output-head overlay | Pinned decoded head vs disk/decode head | L1, four families × eight tokens on Qwen3-0.6B; real pinned allocation | **1.279 s/tok** | 1.809 s/tok | **+41.4% throughput** | **3.33x** on the same 0.6B run | N/A | 0.419 vs 0.419 GB; identical IDs; no pageable fallback | **Positive L1.** Best new measured candidate. The 1.556 GB 14B head still exceeds this WSL2 host's ~1 GB pinned ceiling. |
| **2** | **H1** | Critical-path residency | Event-DAG criticality vs traffic-density residency | L1, real 14B CUDA screen; 441-tensor trace coverage | **17.084 s/tok** | 17.360 s/tok | **+1.61%** | 1.69x in its screen; ~1.52x vs corrected Air run | ~0.84x descriptively | 3.934 GB; exact | Positive direction, below the registered +8% gate. Best live 14B research candidate. |
| **3** | **H6** | Per-tensor exact representation design | Mixed disk/compressed-RAM/decoded-RAM/VRAM DP vs uniform disk | L1 offline prediction on all 441 real tensors; 64 MiB DP quantum; measured 6.669 GB/s pinned H2D floor | **15.011 s predicted prepare** | 24.421 s predicted prepare | **38.56% predicted reduction** | N/A | N/A | 3.959 GB VRAM + 7.986 GB RAM; exact options only | **Prediction gate passed.** Do not rank as a live speedup until the frozen plan executes on held-out traces. |
| **4** | **H10** | Replay/CEM whole-set residency | CEM plan vs profiled knapsack | L1 real 14B common screen plus one-token pilot | **19.480 s/tok** | 19.554 s/tok | **+0.38%**; pilot +2.1% | 1.54x in common screen | ~0.74x descriptively | 3.934 GB; exact | Real but parity-band result; below +8% gate. |
| **5** | **H0** | Joint semantic/system oracle gap | Per-context oracle vs best global policy | L1 replay of fixed vs hazard speculation across four held-out families | 0.13537 tok/s oracle | 0.13199 tok/s global | **+2.56% ceiling** | N/A | N/A | Diagnostic; no memory change | Gate did its job: below 12%, so contextual/RL control has too little upside. |
| **6** | **H3** | Baseline-guarded contextual bandit | LinUCB selection vs fixed global profile | Four-fold leave-one-family-out replay, run despite H0 | 0.52794 total reward | 0.52794 total reward | **0.00%**; 97.50% of oracle | N/A | N/A | Offline controller; exact profiles | Controller ran successfully but learned the baseline. Passes oracle-fidelity metric only because the oracle gap is tiny. Do not deploy. |
| **7** | **H8** | Shadow model-based RL controller | Digital-twin choice vs contextual baseline | L1 shadow replay on two real held-out CEM/control timing pairs | 0.10932 reward | 0.10897 reward | **+0.33%**, but MAPE 11.87%, rank correlation -0.80 | N/A | N/A | Offline only | **Calibration gate failed** (requires MAPE ≤10%, rank ≥0.90). The RL simulator does not rank actions reliably. |
| **8** | **H18** | Rollback-cached target verification | Exact target KV crop/reuse vs full-prefix fixed speculation | **L2**, eight randomized pairs × eight tokens | 4.640 s/tok | 4.574 s/tok | **-0.59% paired**; 90% interval [-4.62%, +1.09%] | N/A | N/A | 3.834 vs 3.815 GB; identical IDs; 16 crops/326 prefix tokens | Mechanism passed, performance stopped for futility. Only 3/8 pairs won; fixed speculation often finished in one sweep. |
| **9** | **H16** | Speculation-conditioned critical-path residency | Distinct critical resident set + fixed speculation vs fixed speculation | L1, four families × four tokens | 8.836 s/tok | 8.350 s/tok | **-2.75% paired median**; 1/4 cells won | ~3.27x descriptive | ~1.62x descriptive | 3.814 GB matched; identical IDs; distinct plan | Combination did not compound. Stop current composition. |
| **10** | **H7** | Expert-local XOR reference coding | One base + seven XOR deltas vs independent compression | L1 artifact screen on eight real Qwen1.5-MoE-A2.7B layer-0 experts | 37.734 MB forced XOR | 36.909 MB independent | **-2.24% storage**; safe chooser saved 0% | N/A | N/A | Real BF16 tensors round-tripped bit-for-bit | Exact mechanism works; compression hypothesis is contradicted on this expert set. |
| **11** | **H13** | Tensor QUBO residency | Pairwise QUBO vs profiled knapsack | L1 replay after greedy-refill repair; 730 candidates | 20.479 s replay | 20.479 s replay | 0%; 100% plan overlap | N/A | N/A | Exact frozen plan | No distinct treatment exists to time. Stop before GPU by design. |
| **12** | **H15** | Physical-extent QUBO residency | Extent QUBO vs profiled knapsack | L1 replay; 369 candidates over 81 real extents | 20.479 s replay | 20.479 s replay | 0%; 100% overlap | N/A | N/A | Exact frozen plan | Same null action-space result as H13. |
| **13** | **H11** | Tiny neural survival-utility stopping | Neural stopping vs tuned fixed `k=8` | L1 train/held-out action gate; 200 observations, 47 opportunities | 0 early stops | fixed `k=8` | No action divergence; apparent +9.5% timing is noise | N/A | N/A | Distribution exact | Current network learned the fixed policy. Do not use its timing as evidence. |
| **14** | **H2** | Hazard-cost speculative stopping | Frozen hazard policy vs tuned fixed `k=8` | L1 four-family CUDA screen | **9.773 s/tok** | **9.150 s/tok** | **-6.4% throughput** | 2.95x | 1.47x | 3.814 GB; greedy IDs exact | Fixed speculation wins. This does **not** invalidate fixed speculation. |
| **15** | **H12** | Bayesian probit prefetch | Chance-constrained depth vs fixed depth 2 | L2, eight randomized pairs, 780 posterior observations | 20.271 s/tok | 19.531 s/tok | **-5.89% paired**; 90% interval [-8.01%, +5.55%] | 1.49x same-session | ~0.71x descriptive | 3.934 GB; exact | Do not advance: wait rose 28.2% and only 3/8 pairs won. |
| **16** | **H4** | Feedback prefetch (PI/MPC) | Adaptive depth vs tuned fixed depth | Two real CUDA screens | PI 21.253; MPC 35.031 s/tok | 20.040 s/tok | PI **-5.7%**; MPC **-42.8%** throughput | PI 1.41x; MPC 0.86x in screen | Both slower descriptively | 3.934 GB; exact | Direction was negative in both screens. Earlier PI run was -35.7%; magnitude changed, conclusion did not. |
| **17** | **H5** | Certified greedy MIPS head | Certified branch-and-bound vs resident full head | L1 real 14B screen; 6.296 GB index | 20.065 s/tok | 13.949 s/tok | **-30.5% throughput**; only 0.084% rows pruned | 1.45x in matched cells | ~0.71x descriptive | 2.669 GB GPU + 6.296 GB CPU index; greedy exact | Kill current bounds/index. Certification overhead dominates. |
| **18** | **H17** | Tensor-scoped micro-extents | Per-tensor 8 MiB extents vs per-blob reads | L1, four families × four tokens | 19.747 s/tok | 16.252 s/tok | **-18.37% paired median**; 0/4 cells won | ~1.46x descriptive | ~0.73x descriptive | 3.934 GB; exact; 57.05% fewer calls; +0.54% bytes | The narrower boundary did not fix the Python buffer/view cost. Stop current path. |
| **19** | **H14** | Coalesced storage reads | Adjacent extent reads vs per-blob reads | L1, four randomized pairs | 26.527 s/tok | 19.578 s/tok | **-27.73% throughput** | ~1.13x descriptive | ~0.54x descriptive | 3.934 GB; exact | Mechanism passed (89.07% fewer calls, 0% extra bytes) but destroyed overlap. Stop this implementation. |

## Ranking: what is actually worth using or pursuing

### Use now

1. **Fixed speculation** is Afterimage's strongest result: 2.93x AirLLM and
   1.56x Accelerate. H2 and H11 are adaptive alternatives to this control, not
   renamings of it.
2. **Hugging Face Accelerate** is the best exact non-speculative 4 GB runtime in
   these measurements. Afterimage should not claim otherwise.
3. **Afterimage exact residency** remains useful when the compressed store,
   explicit memory dial, or Afterimage API/server matters, but it loses to
   Accelerate on raw non-speculative latency at 4 GB.
4. **AirLLM** remains the best measured exact low-VRAM point. Afterimage's exact
   minimum is 17.5% slower (26.828 vs 32.514 s/token, throughput-deficit
   convention: `1 - candidate_throughput/AirLLM_throughput`) and uses 0.140 GB
   more VRAM.

### Research next

1. **H9**: retest the full 14B pinned head on native Linux or hardware that can
   pin at least 1.6 GB. Its smaller-model matched-VRAM effect is large enough to
   justify that run.
2. **H1**: run the registered paired L3 protocol at several VRAM budgets. It is
   small, but it is the cleanest positive 14B live signal.
3. **H6**: materialize and execute the frozen mixed-representation plan. Its
   38.56% figure is a cost-model prediction, not a benchmark.
4. **Do not iterate further on Python `bytearray` coalescing.** H17 narrowed the
   boundary to one tensor, cut calls 57.05%, and was still 18.37% slower by the
   paired median. A new storage treatment must use reusable registered buffers
   plus native `preadv`, `io_uring`, or mmap page-fault scheduling and preserve
   consumer-ready completion order.
5. **Do not advance H16 or H18 as general defaults.** H16 shows that two
   individually sensible optimizations need not compound. H18 may be revisited
   only for a separately preregistered long-context/many-sweep stratum; its
   general eight-token L2 screen stopped for futility.

H9's native-Linux scale test and H6's pinned 8 GB mixed plan cannot be executed
faithfully on this WSL2 host: this is an environment requirement, not a missing
Python package. H1's registered three-budget L3 requires more than the stated
one-hour bounded-run ceiling at 14B latency. They remain explicit external-host
or longer-run work, not silently relabeled as complete.

Everything else is below gate, action-identical to its control, or contradicted
by current evidence.

## Why the earlier conclusions appeared to keep changing

- The original “3x” row is **fixed speculation**, a stable core configuration.
  Later H2/H11 results tested whether adaptive policies beat that row; they did
  not. They never removed the fixed result.
- The first AirLLM run inherited Qwen generation/EOS behavior that did not match
  the forced-greedy protocol. The corrected AirLLM run is 28.861 s/token with
  the same token IDs. Same-session screens can show about 30.0 s/token; that is
  normal run/session variation, not a renamed method.
- On 2026-08-26 the AirLLM anchor was refreshed from the corrected 3.1.0 run
  above to AirLLM 3.2.0 (26.828 s/token), a real upstream speedup, not a
  correction of a previous mistake. Every “vs AirLLM” ratio in this document
  moved by the same factor because no Afterimage number changed. See
  [RESULTS_LOG.md](RESULTS_LOG.md#2026-08-26-airllm-baseline-refresh-to-320).
- H9's old 14B measurements silently used pageable RAM and therefore were not a
  test of pinned memory. Raising memlock exposed the real WSL2 allocation limit;
  the new 0.6B run executes the intended pinned mechanism without fallback.
- H11, H13, and H15 deliberately stop when candidate actions equal control
  actions. Timing identical code paths and calling the noise a speedup would be
  statistically invalid.
- “L1”, “L2”, and “L3” are evidence levels, not names for pass/fail. Most of the
  current work is a mechanism screen because a full 5×16 or 5×128-token protocol
  would take hours on a 20–30 second/token streamed target.

## Novelty assessment

There is a useful system here, but the evidence does not support a broad claim
that every component is new:

- Exact field-split compression is closely related to
  [ZipNN](https://arxiv.org/abs/2411.05239); layer streaming is the space
  [AirLLM](https://github.com/lyogavin/airllm) already occupies.
- Speculative execution over an offloaded target has direct prior art in
  [SpecExec](https://arxiv.org/abs/2406.02532). Afterimage's 2.93x is a valuable
  implementation result, not a claim to invent speculation.
- H3/H8 adapt contextual control and digital-twin RL. They ran, but the observed
  2.56% oracle ceiling and failed simulator ranking remove the practical premise.
- H6's multiple-choice exact physical-design formulation is a plausible
  Afterimage-specific contribution, although dynamic programming and tiered
  placement are standard. It needs a live validation before a novelty claim.
- H13/H15 are quantum-**inspired** QUBO formulations solved classically. They
  returned the control plan exactly, so there is no quantum advantage result.
- H9's liveness-specific pinned output-head overlay is the most interesting new
  systems lead found here. The current result is too small-scale for a claim of
  general 14B superiority.
- H16 is a new composition of two established ideas, but the distinct combined
  treatment regressed; integration alone is not a performance contribution.
- H17's tensor-scoped coalescing boundary was not found as an exact copied
  Afterimage design, but its performance is contradicted and request batching
  itself is old systems work.
- H18's implementation calls Transformers' established `DynamicCache.crop` and
  speculative runtimes already roll back KV state. The local contribution is
  the fail-closed offloaded-target integration and evidence, not invention of
  KV rollback; the L2 result is negative.

The defensible contribution today is the integration: a lossless compressed
store, explicit and fail-closed memory tiers, exact/approximate contracts,
speculation, serving interfaces, and an evidence system that records negative
and gated results instead of hiding them.

## Why other repositories are not in the numeric direct table

- [FlexGen](https://github.com/Relaxed-System-Lab/FlexGen) is designed for
  throughput-oriented large batches and OPT-family experiments; its published
  numbers are not interactive BF16 Qwen3-14B latency.
- [DeepSpeed ZeRO-Inference](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/inference/huggingface/zero_inference)
  is likewise throughput/batch focused and often changes quantization or
  hardware assumptions.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) would require a GGUF
  conversion and is normally compared with integer quantization and substantial
  CPU execution. That is useful in practice but not the same weight/runtime
  contract.
- AirLLM wrappers and forks are not independent execution baselines.
- MoE-specific systems address a different model family than dense Qwen3-14B.

Hugging Face Accelerate is included because its official
[big-model inference](https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling)
supports the original checkpoint, BF16, and GPU/CPU/disk placement without a
format conversion.

## Raw evidence

- [Corrected AirLLM four-family run](../results/2026-08-21_airllm_noeos_qwen3-14b_rtx3080_run2.json)
- [Afterimage common bounded matrix](../results/2026-08-21_bounded_qwen3-14b_rtx3080_run1.json)
- [Accelerate four-family GPU/CPU/disk run](../results/2026-08-22_hf_accelerate_gpu_disk_qwen3-14b_rtx3080_full_l1.json)
- [H9 genuine pinned overlay run](../results/2026-08-22_h9_pinned_overlay_qwen3-0.6b_rtx3080_l1.json)
- [H0/H3/H6/H7/H8 offline and artifact runs](../results/2026-08-22_offline_h0_h3_h6_h7_h8.json)
- [Pinned H2D bandwidth](../results/2026-08-22_pinned_h2d_rtx3080.json)
- [H1 critical-path screen](../results/2026-08-21_h1_critical_path_qwen3-14b_rtx3080_run1.json)
- [H10 CEM screen](../results/2026-08-21_h10_replay_cem_qwen3-14b_rtx3080_screen2.json)
- [H11 action-divergence gate](../results/2026-08-21_h11_neural_utility_qwen3-14b_rtx3080_calibration_gate4.json)
- [H12 regulated pairs](../results/2026-08-21_h12_bayesian_prefetch_qwen3-14b_rtx3080_l2.json)
- [H13/H15 post-repair replay](../results/2026-08-21_h13_h15_qubo_qwen3-14b_rtx3080_postrepair_gate2.json)
- [H14 coalesced-I/O screen](../results/2026-08-21_h14_coalesced_storage_qwen3-14b_rtx3080_l1_screen2.json)
- [H16/H17/H18 common 14B screen](../results/2026-08-22_h16_h17_h18_qwen3-14b_rtx3080_l1.json)
- [H18 randomized L2 screen](../results/2026-08-22_h18_rollback_cached_spec_qwen3-14b_rtx3080_l2.json)

## Reproduce the new runs

Repository verification as of the cross-family campaign (2026-08-22):

- WSL/CUDA: **317 passed, 61 deselected** in 112.25 seconds; one upstream
  Starlette/httpx deprecation warning;
- focused campaign/CLI regression suite: **12 passed** in 3.08 seconds;
- changed-scope Ruff, `compileall`, and `git diff --check` passed.

Current verification (2026-08-26, after the AirLLM 3.2.0 refresh and the
repository-hygiene pass): **366 passed, 61 deselected** on WSL2/CUDA with
zero skips, **300 passed / 66 skipped** on native Windows CPU (Triton has no
Windows wheel, so the GPU decode tests cannot run there), Ruff clean across
`afterimage`, `scripts`, and `tests`. See
[REPRODUCE.md](REPRODUCE.md).

```bash
# Real pinned H9 mechanism test; systemd removes the shell's 64 MiB memlock cap.
systemd-run --wait --pipe -p LimitMEMLOCK=infinity /bin/bash -lc '
  source /root/.venv/bin/activate
  python scripts/run_bounded_suite.py \
    --model Qwen/Qwen3-0.6B \
    --store /root/afterimage/store_qwen3_0.6b \
    --methods airllm,exact-min,ram-overlay-head \
    --max-new-tokens 8 \
    --ram-overlay-vram-budget-gb 0.5 \
    --ram-overlay-host-budget-gb 0.35 \
    --out results/H9.json'

python scripts/run_offline_hypotheses.py \
  --manifest /root/afterimage/store_14b/manifest.json \
  --moe-shard /path/to/Qwen1.5-MoE/model-00001-of-00008.safetensors \
  --out results/OFFLINE.json

pip install -e ".[bench]"
python scripts/run_hf_offload_baseline.py \
  --gpu-memory 4000MB --cpu-memory 8GB --max-new-tokens 4 \
  --out results/HF_ACCELERATE.json

# H16/H17/H18 common screen, then H18's randomized L2 follow-up.
python scripts/run_bounded_suite.py \
  --methods exact-resident,tensor-extents,spec-fixed,spec-critical,spec-cached \
  --max-new-tokens 4 --time-budget-minutes 58 --out results/H16_H17_H18.json
python scripts/run_regulated_pair.py \
  --hypothesis H18 --blocks 2 --max-new-tokens 8 --skip-airllm \
  --time-budget-minutes 40 --out results/H18_L2.json
```

NVIDIA documents that pinned system memory availability is limited under WSL2:
[CUDA on WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
That is why the 14B H9 scale test requires native Linux or a different host,
not another Python package.
