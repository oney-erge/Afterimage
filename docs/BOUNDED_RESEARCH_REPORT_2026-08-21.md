# Bounded research report — 21 August 2026

## Outcome

The richer benchmark changes the headline. On four held-out prompt types,
Afterimage's minimum-memory exact path is **11.2% slower than AirLLM**, not
faster. Spending 3.93 GB gives a reproducible **1.66x** speedup, and fixed-k
speculation gives the largest exact result, **3.15x** at 3.81 GB. The old
12.5x number was real for one unusually high-agreement prompt, but it does not
generalise: fixed speculation ranged from 1.88x to 5.36x by prompt and averaged
3.15x here.

None of H0-H8 is presently a validated new research result. H1, H2, H4, and H5
missed their gates; H3 and H8 were correctly stopped by H0; H6 and H7 need
artifacts this dense checkpoint does not contain. The implementation is useful
as a configurable falsification platform, but implementation is not novelty.

## Protocol

- Model: Qwen3-14B BF16, revision
  `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- Store: 29.536 GB original, 20.328 GB exact compressed, 1.453x ratio.
- Hardware: RTX 3080 Laptop GPU, 8 GB; WSL2; PyTorch 2.6.0+cu124;
  Transformers 5.12.1; AirLLM 3.1.0.
- Four evaluation prompts: factual answer, arithmetic, Python completion, and
  long-context retrieval. Two disjoint prompts were reserved for calibration
  (`prompt_suite.py` version `bounded-chat-v1`, in effect for this run only;
  it was later expanded to eight calibration prompts as `bounded-chat-v2` for
  the H9-H15 screens -- the four evaluation prompts are unchanged).
- Four forced greedy tokens per evaluation prompt. Linux page cache was dropped
  successfully before every timed cell. Every system used the same
  `torch.cuda.max_memory_allocated()` counter.
- AirLLM EOS stopping was disabled with `eos_token_id=[]`, rather than using
  `min_new_tokens`, so exactly four tokens were generated without suppressing
  EOS in the logits. This corrected run is the comparison denominator.
- Each cell was run once. This is a 45.9-minute multi-fidelity screen, not the
  five-repeat confirmatory protocol. Historical measurements put the noise band
  near ±4%; small effects are directional only.

All methods returned the expected answer on all evaluated cells. In the core
four-prompt matrix, the corrected AirLLM run and every exact/approximate
Afterimage row emitted the same four token IDs on every shared prompt. This
does not turn the chunked head into an exact method; it only reports empirical
argmax agreement.

## Main comparison

The speed denominator is the corrected AirLLM aggregate, 28.861 s/token and
1.583 GB. “Exact” refers to the execution contract, not merely answer quality.

| Method | Prompts × tokens | Peak VRAM | s/token | vs AirLLM | Contract | Result |
|---|---:|---:|---:|---:|---|---|
| AirLLM 3.1.0 | 4 × 4 | 1.583 GB | 28.861 | 1.00x | BF16 greedy baseline | 4/4 answers, 4/4 token sequences |
| Afterimage minimum-memory | 4 × 4 | 1.723 GB | 32.514 | **0.89x** | exact | 11.2% slower, 8.8% more VRAM |
| Afterimage 4 GB residency | 4 × 4 | 3.934 GB | 17.360 | **1.66x** | exact | robust gain, bought with 2.48x VRAM |
| Chunked output head | 4 × 4 | **0.901 GB** | 29.606 | 0.97x | approximate | 43.1% less VRAM, parity-speed band |
| Fixed-k speculation | 4 × 4 | 3.813 GB | **9.150** | **3.15x** | greedy-token exact at T=0 | strongest full-suite row |
| Frozen hazard speculation | 4 × 4 | 3.814 GB | 9.773 | 2.95x | greedy-token exact at T=0 | no gain over fixed-k |
| Chunked head + speculation | 1 × 4 | 2.056 GB | 7.949 | 3.70x* | approximate | promising low-memory screen |

`*` One factual prompt only, compared with AirLLM's matching 29.45 s/token
cell. It is not a suite aggregate.

An additional two-prompt placement control kept the full output head resident:
2.681 GB and 13.949 s/token, or 2.09x the matching AirLLM cells. This was faster
than the 4 GB traffic-density plan despite reading slightly more bytes. It is a
strong clue that access layout and large contiguous reads matter more than the
current byte-count proxy captures.

## H0-H8 verdicts

| ID | Measured result | Verdict now |
|---|---|---|
| H0 joint oracle gap | Semantic oracle uplift **2.56%**, gate 12%; only one system bucket | **Gate closed.** Too little observed value for contextual/RL selection. |
| H1 critical-path residency | 17.084 vs 17.360 s/token, **1.61%** gain; gate 8% | **Below gate.** The 441-tensor profile passed coverage, but the effect is small/mixed. |
| H2 rejection-hazard stopping | 9.773 vs fixed 9.150 s/token, **6.4% lower throughput** | **Not supported.** Two calibration prompts were too sparse and the policy used the same target-sweep counts as fixed-k. |
| H3 conservative contextual bandit | Not run after H0 failed | **Correctly gated**, not a failure to execute. |
| H4 feedback prefetch | PI 27.009 vs fixed 17.360 s/token: **35.7% lower throughput** | **Rejected in current form.** Misses rose 47→98 and exposed wait 3.51→13.96 s. MPC also lost on its screen. |
| H5 certified MIPS head | 20.065 vs matched full head 13.949 s/token: **30.5% lower throughput** | **Killed with current bounds.** 6.296 GB index, 11.08 s build, only 0.084% rows pruned. |
| H6 exact representation DP | No alternative exact artifacts in the v2 store | **Untested.** Planner unit tests are not a hardware result. |
| H7 expert-local XOR reference | Qwen3-14B is dense; there are no expert tensors | **Not applicable on this checkpoint.** |
| H8 model-based RL | Not run after H0/H3 gates failed | **Correctly gated.** |

H1 calibration initially exposed a trace-lifecycle bug: discarded startup
events remained in dependency maps. The shared trace reset was fixed and
regression-tested before the reported rerun. H5 initially exposed a correct
capability gate at the wrong placement; it was rerun against a matched resident
full-head control.

## How this stands against nearby systems

AirLLM is the only installed baseline that executes the same Qwen3-14B BF16
checkpoint with interactive layer streaming, so it is the only external system
given a direct number. The installed 3.1.0 code streams one module at a time and
prefetches the next; current upstream releases are evolving, so the result must
not be generalised to every AirLLM configuration or future release.

Other repositories were reviewed but not forced into an invalid one-hour table:

- [FlexGen](https://arxiv.org/abs/2303.06865) and
  [ZeRO-Inference](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/inference/huggingface/zero_inference)
  are throughput-oriented, usually compare large batches, and commonly use
  quantisation/KV offload. Their published numbers are not interactive BF16
  Qwen3-14B latency numbers.
- [SpecExec](https://arxiv.org/abs/2406.02532) is the closest conceptual prior:
  it explicitly amortises an offloaded target iteration over many speculative
  tokens. This means “speculation for offloaded models” is not new here.
- [PowerInfer](https://arxiv.org/abs/2312.12456) depends on activation sparsity,
  predictor weights, and supported ReLU-sparse models; dense Qwen3-14B is not a
  like-for-like input.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) would require a GGUF
  conversion and is usually evaluated with integer quantisation. That changes
  precision, storage format, and often CPU participation.
- ATSInfer performs tensor-granular placement and load-aware transfer on
  consumer devices, further narrowing any broad placement novelty claim.
  See [ATSInfer](https://arxiv.org/abs/2607.10183).

Installing or converting these systems would have exceeded the bounded run and
still not produced a matched model/precision/latency comparison. The honest
comparison is therefore one measured external baseline plus literature-level
positioning for the rest.

## Novelty assessment

There is **no defensible “this does not exist in the literature” claim yet**.

- Lossless BF16 compression is established by
  [ZipNN](https://arxiv.org/abs/2411.05239).
- Offloaded speculative execution is established by SpecExec.
- Adaptive draft length/confidence stopping is crowded: [AdaEDL](https://arxiv.org/abs/2410.18351),
  [PEARL](https://arxiv.org/abs/2408.11850), and
  [PACER](https://arxiv.org/abs/2602.01274) already cover entropy-, history-,
  parallel-, or predictor-based adaptation.
- Exact branch-and-bound MIPS predates this project; see
  [tree MIPS](https://arxiv.org/abs/1202.6101) and
  [MAXIMUS/OPTIMUS](https://arxiv.org/abs/1706.01449). The roundoff-safe LLM
  fallback is a narrow implementation twist, but the measured method is worse.
- XOR delta compression is already part of ZipNN; moving the reference relation
  inside one MoE checkpoint is an untested transfer hypothesis, not an invention.

The most defensible contribution today is an engineering artifact with an
honest Pareto map and negative results. That can be valuable, but it is not the
same as a novel algorithmic paper.

## What to pursue next, from fundamentals

Do not invest next in RL. H0 found only 2.56% oracle headroom, and both feedback
controllers regressed. The surprising full-head result suggests a better new
hypothesis:

> **Layout-aware residency (proposed here as "H9" at the time of writing --
> that slot was subsequently taken by the liveness-guided RAM overlay
> hypothesis; the layout idea itself shipped as H14/H15. Read this block as
> the forward-looking proposal that motivated H14/H15, not as a description
> of the current H9):** selecting contiguous on-disk segments and
> large execution phases, rather than independent tensors, reduces request
> fragmentation and preserves sequential reads; at the same VRAM it beats both
> traffic-density and noisy per-tensor critical-path placement.

This comes from the observation that 2.68 GB resident-full-head execution
outperformed the 3.93 GB tensor plan while reading more logical bytes. The test
should add `pread` count, request-size distribution, physical disk bytes, queue
depth, and segment boundaries; compare fixed tensor, full-head, segment-knapsack,
and segment+critical-path profiles at 2.68 and 4.0 GB; and require ≥8% held-out
gain with identical tokens. It remains a hypothesis until measured.

**Update, measured since:** H14 (coalesced contiguous storage reads) tested
exactly this mechanism and found the opposite of what this section predicted
-- 27.7% *slower*, because a large contiguous read serialises against decode
instead of overlapping with it. See
[RESEARCH_METHODS.md](RESEARCH_METHODS.md#h14-h15----storage-layout-as-a-residency-action)
and [HYPOTHESIS_LINEAGE.md](HYPOTHESIS_LINEAGE.md).

A second, higher-risk direction is fused decode-and-matmul over the exact
compressed representation. Minimum-memory Afterimage loses to AirLLM because
saved I/O is being exchanged for a separate decompression/materialisation pass.
Eliminating that pass attacks the observed root cost. It is more promising than
another policy layered over the current kernels.

## Raw evidence

- [`results/2026-08-21_airllm_noeos_qwen3-14b_rtx3080_run2.json`](../results/2026-08-21_airllm_noeos_qwen3-14b_rtx3080_run2.json)
  — corrected AirLLM denominator.
- [`results/2026-08-21_bounded_qwen3-14b_rtx3080_run1.json`](../results/2026-08-21_bounded_qwen3-14b_rtx3080_run1.json)
  — core matrix; its original AirLLM rows used the superseded
  `min_new_tokens` protocol.
- [`results/2026-08-21_h1_critical_path_qwen3-14b_rtx3080_run1.json`](../results/2026-08-21_h1_critical_path_qwen3-14b_rtx3080_run1.json)
- [`results/2026-08-21_h5_certified_mips_qwen3-14b_rtx3080_run1.json`](../results/2026-08-21_h5_certified_mips_qwen3-14b_rtx3080_run1.json)
- [`results/2026-08-21_ablation_screen_qwen3-14b_rtx3080_run1.json`](../results/2026-08-21_ablation_screen_qwen3-14b_rtx3080_run1.json)
