# All hypotheses, baselines, results, and ranking

**Last reevaluated:** 2026-08-22
**Primary hardware:** NVIDIA RTX 3080 Laptop GPU, 8 GB
**Primary model:** Qwen3-14B, original BF16 weights
**Scope:** cold page cache, forced greedy decoding, four prompt families unless a row says otherwise

This is the controlling results document. It covers every registered hypothesis
H0-H15, the stable Afterimage configurations, AirLLM 3.1.0, and Hugging Face
Accelerate disk offload. A result is not called “failed” merely because an
upstream gate prevented a meaningless timing comparison.

## Answer in one paragraph

Afterimage is **not better at the exact low-VRAM floor**: it uses 1.723 GB and
runs at 0.89x AirLLM. At about 4 GB, exact residency reaches 1.66x AirLLM but is
0.82x the speed of Hugging Face Accelerate's 3.80 GB hybrid. Afterimage's best
configuration is fixed speculative decoding: **9.150 s/token, 3.15x AirLLM and
1.56x Accelerate at 3.813 GB**, with identical greedy token IDs in the tested
suite. Of the new H0-H15 ideas, H9 is the strongest measured lead (+41.4%
throughput at matched VRAM on a smaller model), H1 is the strongest positive
14B live result (+1.61%), and H6 has the largest predicted 14B upside (38.56%)
but still lacks a held-out live execution. No H1-H15 candidate has L3
confirmatory superiority evidence.

## Stable systems and external baselines

These are complete four-family × four-token runs. “Versus” is throughput ratio,
so values above 1.00x are faster. Accelerate was allowed 4.0 GB and actually
peaked at 3.800 GB; its map put 2 modules on CUDA, 9 on CPU, and 33 on disk.

| Overall rank | System/configuration | Peak VRAM | Seconds/token | vs AirLLM | vs Accelerate | Output contract | What the result means |
|---:|---|---:|---:|---:|---:|---|---|
| **1** | **Afterimage + fixed `k=8` speculation** | 3.813 GB | **9.150** | **3.15x** | **1.56x** | Greedy-token exact at T=0; distribution-exact sampler at T>0 | Best measured operating point; the 3x result is real and was never invalidated by H2/H11. |
| **2** | Hugging Face Accelerate BF16 GPU/CPU/disk offload | 3.800 GB | **14.318** | **2.02x** | 1.00x | Same BF16 checkpoint; same four greedy token IDs | Best non-speculative 4 GB result in this repository's runs. |
| **3** | Afterimage exact + 4 GB residency | 3.934 GB | 17.360 | **1.66x** | 0.82x | Reference-execution equivalent | Faster than AirLLM, slower than Accelerate at roughly matched VRAM. |
| **4** | AirLLM 3.1.0 | **1.583 GB** | 28.861 | 1.00x | 0.50x | Same BF16 checkpoint; corrected forced-greedy run | Best measured exact low-VRAM external baseline. |
| **5** | Afterimage chunked head | **0.901 GB** | 29.606 | 0.97x | 0.48x | **Approximate** matmul; tested token IDs agreed | Lowest VRAM and near AirLLM speed, but not an exact method. |
| **6** | Afterimage exact minimum-memory | 1.723 GB | 32.514 | 0.89x | 0.44x | Reference-execution equivalent | Compression saves storage, not enough runtime to beat AirLLM at the floor. |

The exact minimum cannot honestly be advertised below AirLLM's measured VRAM:
Qwen3-14B's full BF16 output head is 1.556 GB before transient scratch. Computing
the head in blocks reaches 0.901 GB, but changes BF16 matmul rounding. That is
why “just match lower VRAM” is not a missing flag.

## One large H0-H15 comparison table

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
| **8** | **H7** | Expert-local XOR reference coding | One base + seven XOR deltas vs independent compression | L1 artifact screen on eight real Qwen1.5-MoE-A2.7B layer-0 experts | 37.734 MB forced XOR | 36.909 MB independent | **-2.24% storage**; safe chooser saved 0% | N/A | N/A | Real BF16 tensors round-tripped bit-for-bit | Exact mechanism works; compression hypothesis is contradicted on this expert set. |
| **9** | **H13** | Tensor QUBO residency | Pairwise QUBO vs profiled knapsack | L1 replay after greedy-refill repair; 730 candidates | 20.479 s replay | 20.479 s replay | 0%; 100% plan overlap | N/A | N/A | Exact frozen plan | No distinct treatment exists to time. Stop before GPU by design. |
| **10** | **H15** | Physical-extent QUBO residency | Extent QUBO vs profiled knapsack | L1 replay; 369 candidates over 81 real extents | 20.479 s replay | 20.479 s replay | 0%; 100% overlap | N/A | N/A | Exact frozen plan | Same null action-space result as H13. |
| **11** | **H11** | Tiny neural survival-utility stopping | Neural stopping vs tuned fixed `k=8` | L1 train/held-out action gate; 200 observations, 47 opportunities | 0 early stops | fixed `k=8` | No action divergence; apparent +9.5% timing is noise | N/A | N/A | Distribution exact | Current network learned the fixed policy. Do not use its timing as evidence. |
| **12** | **H2** | Hazard-cost speculative stopping | Frozen hazard policy vs tuned fixed `k=8` | L1 four-family CUDA screen | **9.773 s/tok** | **9.150 s/tok** | **-6.4% throughput** | 2.95x | 1.47x | 3.814 GB; greedy IDs exact | Fixed speculation wins. This does **not** invalidate fixed speculation. |
| **13** | **H12** | Bayesian probit prefetch | Chance-constrained depth vs fixed depth 2 | L2, eight randomized pairs, 780 posterior observations | 20.271 s/tok | 19.531 s/tok | **-5.89% paired**; 90% interval [-8.01%, +5.55%] | 1.49x same-session | ~0.71x descriptive | 3.934 GB; exact | Do not advance: wait rose 28.2% and only 3/8 pairs won. |
| **14** | **H4** | Feedback prefetch (PI/MPC) | Adaptive depth vs tuned fixed depth | Two real CUDA screens | PI 21.253; MPC 35.031 s/tok | 20.040 s/tok | PI **-5.7%**; MPC **-42.8%** throughput | PI 1.41x; MPC 0.86x in screen | Both slower descriptively | 3.934 GB; exact | Direction was negative in both screens. Earlier PI run was -35.7%; magnitude changed, conclusion did not. |
| **15** | **H5** | Certified greedy MIPS head | Certified branch-and-bound vs resident full head | L1 real 14B screen; 6.296 GB index | 20.065 s/tok | 13.949 s/tok | **-30.5% throughput**; only 0.084% rows pruned | 1.45x in matched cells | ~0.71x descriptive | 2.669 GB GPU + 6.296 GB CPU index; greedy exact | Kill current bounds/index. Certification overhead dominates. |
| **16** | **H14** | Coalesced storage reads | Adjacent extent reads vs per-blob reads | L1, four randomized pairs | 26.527 s/tok | 19.578 s/tok | **-27.73% throughput** | ~1.13x descriptive | ~0.54x descriptive | 3.934 GB; exact | Mechanism passed—89.07% fewer calls, 0% extra bytes—but destroyed overlap. Stop this implementation. |

## Ranking: what is actually worth using or pursuing

### Use now

1. **Fixed speculation** is Afterimage's strongest result: 3.15x AirLLM and
   1.56x Accelerate. H2 and H11 are adaptive alternatives to this control, not
   renamings of it.
2. **Hugging Face Accelerate** is the best exact non-speculative 4 GB runtime in
   these measurements. Afterimage should not claim otherwise.
3. **Afterimage exact residency** remains useful when the compressed store,
   explicit memory dial, or Afterimage API/server matters, but it loses to
   Accelerate on raw non-speculative latency at 4 GB.
4. **AirLLM** remains the best measured exact low-VRAM point. Afterimage's exact
   minimum is 8.8% slower and uses 0.140 GB more VRAM.

### Research next

1. **H9**: retest the full 14B pinned head on native Linux or hardware that can
   pin at least 1.6 GB. Its smaller-model matched-VRAM effect is large enough to
   justify that run.
2. **H1**: run the registered paired L3 protocol at several VRAM budgets. It is
   small, but it is the cleanest positive 14B live signal.
3. **H6**: materialize and execute the frozen mixed-representation plan. Its
   38.56% figure is a cost-model prediction, not a benchmark.
4. For H14, use reusable registered buffers plus `preadv`, `io_uring`, or mmap
   page-fault scheduling. Repeating the current large Python buffer is not useful.

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
  [SpecExec](https://arxiv.org/abs/2406.02532). Afterimage's 3.15x is a valuable
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

## Reproduce the new runs

Final repository verification after the reevaluation:

- WSL/CUDA: **301 passed, 61 deselected** in 101.51 seconds; one upstream
  Starlette/httpx deprecation warning;
- Windows/CPU: **231 passed, 65 skipped, 61 deselected** in 39.66 seconds;
- `python -m ruff check afterimage`, `compileall`, and `git diff --check` passed.

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
```

NVIDIA documents that pinned system memory availability is limited under WSL2:
[CUDA on WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
That is why the 14B H9 scale test requires native Linux or a different host,
not another Python package.
