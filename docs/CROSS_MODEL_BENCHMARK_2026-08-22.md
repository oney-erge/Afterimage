# Cross-family and scale benchmark, 2026-08-22

**Status:** hardware campaign complete, as run on this date -- see note below
**Host:** RTX 3080 Laptop GPU (8 GB), WSL2, BF16, cold page cache
**Question:** do Afterimage's Qwen3-14B conclusions transfer to a smaller
checkpoint and a different, substantially larger model family?

All AirLLM numbers in this document, including the Qwen3-14B row below, are
AirLLM 3.1.0, as run on 2026-08-22. The Qwen3-14B AirLLM anchor used
elsewhere in this repository (README.md, ALL_HYPOTHESES_AND_BASELINES.md,
HOW_IT_WORKS.md) was refreshed to AirLLM 3.2.0 on 2026-08-26; this campaign
report is left as the historical record of what actually ran that day and
has not been reopened. The Phi-4 Mini and Mistral Small 24B rows have not
been rerun on 3.2.0. See
[RESULTS_LOG.md](RESULTS_LOG.md#2026-08-26-airllm-baseline-refresh-to-320).

## Result in one paragraph

The answer is mixed and useful. The lossless store transfers cleanly: three
families compress to a very stable 1.45–1.49x. Runtime winners depend on scale
and memory. Hugging Face Accelerate is fastest on Phi-4 Mini 3.8B and on
Mistral Small 24B. Afterimage fixed speculation remains the Qwen3-14B winner.
On Mistral 24B, Afterimage certified MIPS is the best Afterimage method at
27.539 s/token and 2.915 GB VRAM: **1.66x AirLLM**, but **9.52% slower than
Accelerate**. At the minimum-memory floor AirLLM still wins: 45.737 s/token at
1.367 GB versus Afterimage exact-min at 48.477 s/token and 1.709 GB. H12,
H14, and H17 all fail their paired speed gates on the larger family. The novel
result worth developing is therefore certified output-head search on untied,
large-vocabulary checkpoints, not more prefetch-controller complexity and not
the current coalescing implementations.

## Why these models

The small target is [Microsoft Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct):
3.8B parameters, 32 layers, tied embeddings, revision
`cfbefacb99257ffa30c83adab238a50856ac3083`. The requested approximately-27B
target was first attempted with
[Gemma 2 27B](https://huggingface.co/google/gemma-2-27b-it), but its checkpoint
was license-gated and returned HTTP 401 in this unauthenticated environment.
The closest open, text-only substitute was
[Mistral Small 24B Instruct 2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501):
40 layers, untied output head, Apache-2.0, revision
`9527884be6e5616bdd54de542f9ae13384489724`. Mistral's
[official model documentation](https://docs.mistral.ai/models/mistral-small-3-0-25-01)
describes the same 24B release.

The external controls are
[AirLLM 3.1.0](https://github.com/lyogavin/airllm/releases/tag/v3.1.0) and
[Hugging Face Big Model Inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling).
Accelerate was configured for 4 GB GPU and 12 GB CPU placement with disk
offload; AirLLM used its original BF16 streaming path.

## Protocol

| Stratum | Prompts × forced tokens | Purpose | Runtime |
|---|---:|---|---:|
| Phi-4 Mini broad L1 | 4 families × 4 | cheap cross-family ranking | 15.52 min |
| Mistral 24B broad L1 | 2 families × 2 | scale screen of all applicable methods | 31.02 min |
| Mistral AirLLM | 2 families × 2 | isolated external baseline | 3.13 min after one-time split |
| Mistral H12 paired L1 | 2 randomized pairs plus disjoint burn-in | causal mechanism check | 8.19 min |
| Mistral H14 paired L1 | 2 randomized pairs plus disjoint burn-in | causal mechanism check | 7.19 min |
| Mistral H17 paired L1 | 2 randomized pairs plus disjoint burn-in | causal mechanism check | 7.25 min |

Every timed cell drops the Linux page cache and records raw token IDs,
per-case wall time, peak CUDA allocation, resolved configuration, environment,
model/tokenizer revision, and mechanism counters. Results are checkpointed to
`*.json.partial` after every completed case. The campaign runner now supports
`--resume`; it archives an interrupted checkpoint rather than deleting or
overwriting it.

`expected_match_rate` is a semantic-prefix completion score. It is **not** an
exactness metric: two Mistral tokens produce `73`, not the full expected
`7319`. Execution exactness is `token_agreement_vs_exact_min`, which is 1.0
for every measured Afterimage and AirLLM method in the tables below, and the
same token IDs also match Accelerate. The future runner records these two
metrics separately during interim checkpoints.

## One cross-scale comparison table

`vs Air` and `vs HF` are throughput ratios; values above 1.00x are faster.
Rows are directly comparable only within the same model and protocol. MIPS's
5.432 GB host index is called out because VRAM alone would hide that cost.

| Scale / model | System or configuration | Peak VRAM | s/token | vs Air | vs HF | Output contract | Reading |
|---|---|---:|---:|---:|---:|---|---|
| 3.8B Phi | **HF Accelerate** | 3.770 GB | **1.327** | **6.02x** | 1.00x | BF16/token exact | Clear speed winner when the model is small enough for substantial GPU/CPU residency. |
| 3.8B Phi | Afterimage replay CEM | 2.897 GB | 2.812 | 2.84x | 0.47x | Reference-execution equivalent | Best Afterimage point, but CEM only barely beats simpler critical-path placement. |
| 3.8B Phi | Afterimage exact resident | 2.878 GB | 2.958 | 2.70x | 0.45x | Reference-execution equivalent | Most of the gain is capacity, not search complexity. |
| 3.8B Phi | AirLLM 3.1.0 | **1.458 GB** | 7.984 | 1.00x | 0.17x | Same token IDs | Low VRAM, far slower than resident execution. |
| 3.8B Phi | Afterimage exact minimum | 1.705 GB | 9.775 | 0.82x | 0.14x | Reference-execution equivalent | AirLLM wins the floor. |
| 14B Qwen3 | **Afterimage fixed `k=8` speculation** | 3.813 GB | **9.150** | **3.15x** | **1.56x** | Greedy-token exact at T=0 | Overall Qwen winner; speculation amortizes streamed target passes. |
| 14B Qwen3 | HF Accelerate | 3.800 GB | 14.318 | 2.02x | 1.00x | BF16/token exact | Best non-speculative 4 GB point. |
| 14B Qwen3 | Afterimage exact resident | 3.934 GB | 17.360 | 1.66x | 0.82x | Reference-execution equivalent | Beats AirLLM, trails HF. |
| 14B Qwen3 | AirLLM 3.1.0 | **1.583 GB** | 28.861 | 1.00x | 0.50x | Same token IDs | Low-VRAM external control. |
| 14B Qwen3 | Afterimage exact minimum | 1.723 GB | 32.514 | 0.89x | 0.44x | Reference-execution equivalent | AirLLM wins the floor. |
| 24B Mistral | **HF Accelerate** | 3.821 GB | **25.146** | **1.82x** | 1.00x | BF16/token exact | Fastest 24B result. |
| 24B Mistral | **Afterimage certified MIPS** | 2.915 GB | **27.539** | **1.66x** | **0.91x** | Certified greedy exact; zero fallbacks | Best Afterimage result; 5.432 GB host index and 4.47 s one-time index build. |
| 24B Mistral | Afterimage profiled residency | 3.921 GB | 29.015 | 1.58x | 0.87x | Reference-execution equivalent | Only 2.0% faster than simple residency. |
| 24B Mistral | Afterimage exact resident | 3.920 GB | 29.616 | 1.54x | 0.85x | Reference-execution equivalent | Strong simple control; most placement value is capacity. |
| 24B Mistral | AirLLM 3.1.0 | **1.367 GB** | 45.737 | 1.00x | 0.55x | Same token IDs | Best exact low-VRAM point. |
| 24B Mistral | Afterimage chunked head | **1.186 GB** | 47.885 | 0.95x | 0.53x | Approximate BF16 reduction order; same tested IDs | Lowest VRAM, but neither exact execution nor faster. |
| 24B Mistral | Afterimage exact minimum | 1.709 GB | 48.477 | 0.94x | 0.52x | Reference-execution equivalent | Slightly slower and larger than AirLLM at the floor. |

## Storage transfer result

| Checkpoint | Original Transformers weights | Afterimage store | Lossless ratio |
|---|---:|---:|---:|
| Phi-4 Mini 3.8B | 7.672 GB | 5.166 GB | 1.485x |
| Qwen3-14B | 29.536 GB | 20.328 GB | 1.453x |
| Mistral Small 24B | 47.145 GB | 31.789 GB | 1.483x |

This is the strongest cross-family generalization result in the campaign: the
codec's storage benefit is essentially scale- and family-stable. It does not by
itself imply a runtime win; the fastest runtime depends on where the avoided
bytes sit on the critical path.

## Paired hypothesis results

| Hypothesis | Phi-4 Mini result | Mistral 24B result | Mechanism | Decision |
|---|---|---|---|---|
| H12 Bayesian/probit prefetch | −3.73%; 8 pairs; 90% interval [−10.17%, +6.55%] | −3.27%; 2 pairs; 90% interval [−7.29%, +0.92%] | Failed on both: exposed wait did not fall enough | Stop the current controller. Apparent broad-run wins are residency/noise, not Bayesian causality. |
| H14 layer coalescing | −33.76%; 8 pairs; interval [−36.26%, −32.31%] | −26.08%; 2 pairs; interval [−26.53%, −25.63%] | Passed: Mistral read calls −83.64%, bytes +0.56% | Mechanism real, wall time contradicted. Do not tune this implementation. |
| H17 tensor micro-extents | −12.35%; 8 pairs; interval [−17.19%, −3.92%] | −26.42%; 2 pairs; interval [−26.55%, −26.29%] | Passed: Mistral read calls −54.22%, bytes +0.56% | The proposed H14 repair regresses more at scale. Stop. |

The Mistral rows are L1 scale screens with only two pairs, not L3 claims. Their
effects are nevertheless directionally decisive and reproduce the broader
screens. No H12/H14/H17 candidate advances.

## What the campaign changed

- The compressor now follows `model.safetensors.index.json` and ignores a
  repository's duplicate `consolidated*.safetensors` export. Disk preflight
  uses the same exact shard set instead of double-counting Mistral as 94.3 GB.
- Compression's exponent/sign extraction is bounded-chunk work. Mistral's
  1.34-billion-element output head stays near 0.8 GB worker RSS instead of
  creating multiple full `int32` temporaries and exhausting WSL memory.
- Tied embeddings are streamed again for the disjoint output-head live range.
  This fixed Phi's pre-patch wrong token (`85123`) to the HF-matching token
  `33626` without permanently holding the full tied head in VRAM.
- AirLLM is isolated from Afterimage stages. Its Mistral tokenizer is replaced
  with the benchmark's `fix_mistral_regex=True` tokenizer, and fixed-length
  generation supplies an explicit pad ID for Transformers 5.x.
- Every stage writes durable interim JSON, logs its environment and revisions,
  and can resume without overwriting the interrupted checkpoint.

## Infrastructure incident, preserved but excluded

AirLLM's first two split attempts drove Windows C: to exactly zero free bytes.
The thin-provisioned WSL VHD still reported hundreds of GB internally, then
ext4 remounted read-only when the host VHD could no longer grow. Those rows are
not timings. The 245.45 GB Ubuntu distribution was moved with `wsl --manage
Ubuntu-24.04 --move` to `D:\WSL\Ubuntu-24.04`, restoring 250.93 GB on C: and
leaving over 1 TB free on D:. The failed split, runner checkpoint, and the
separate empty-EOS compatibility failure were preserved. The final AirLLM run
started from a complete fresh split and is the only one used numerically.

## Research conclusion and next tests

1. **Develop H5/certified MIPS, not H12/H14/H17.** On an untied 131k-vocabulary
   head it closes most of the HF gap while using about 1 GB less VRAM. The next
   treatment should compress or lazily stage the 5.432 GB host index and report
   total host RAM, certification rate, rows pruned, and index build amortization.
2. **Use capacity as the placement control.** CEM, critical-path, Bayesian, and
   profiled policies cluster within a few percent of simple residency on both
   new families. Any learned/RL policy must first demonstrate an offline oracle
   gap above 10% and a distinct action rate before another GPU timing.
3. **Keep speculation family-specific.** Qwen's gain is real, but this campaign
   did not invent a tokenizer-compatible resident draft for Phi or Mistral.
   Cross-family speculation is excluded rather than silently using incompatible
   token IDs.
4. **Run confirmation only where a mechanism and effect coexist.** Repeat MIPS
   across at least four prompt families and longer 8–16-token cells. Do not
   spend L2/L3 budget on the contradicted coalescing candidates.

## Raw evidence and reproduction

- [Campaign configuration](../configs/cross_model_benchmark_v1.json)
- [Phi broad matrix](../results/cross_model_2026-08-22_full_v2/phi4-mini-3.8b/broad-l1.json)
- [Phi HF baseline](../results/cross_model_2026-08-22_full_v2/phi4-mini-3.8b/hf-accelerate.json)
- [Phi H12](../results/cross_model_2026-08-22_full_v2/phi4-mini-3.8b/h12-l2.json),
  [H14](../results/cross_model_2026-08-22_full_v2/phi4-mini-3.8b/h14-l2.json),
  and [H17](../results/cross_model_2026-08-22_full_v2/phi4-mini-3.8b/h17-l2.json)
- [Mistral broad Afterimage matrix](../results/2026-08-22_mistral-small-24b_afterimage-broad-l1.json)
- [Mistral AirLLM](../results/2026-08-22_mistral-small-24b_airllm-3.1.0-l1.json)
- [Mistral HF baseline](../results/cross_model_2026-08-22_large_chunked_w1/mistral-small-24b/hf-accelerate.json)
- [Mistral H12](../results/2026-08-22_h12_mistral-small-24b_rtx3080_scale-l1.json),
  [H14](../results/2026-08-22_h14_mistral-small-24b_rtx3080_scale-l1.json), and
  [H17](../results/2026-08-22_h17_mistral-small-24b_rtx3080_scale-l1.json)
- [Invalid but preserved AirLLM empty-EOS failure](../results/2026-08-22_mistral-small-24b_airllm-3.1.0-empty-eos-failed.json)
- [Invalid but preserved AirLLM host-disk exhaustion checkpoint](../results/2026-08-22_mistral-small-24b_airllm-3.1.0-disk-full-failed.json)

Re-run the full staged campaign:

```bash
python scripts/run_cross_model_campaign.py \
  --config configs/cross_model_benchmark_v1.json \
  --out-dir results/cross_model_YYYY-MM-DD \
  --workers 1

# After an interruption, preserve the runner checkpoint and continue.
python scripts/run_cross_model_campaign.py \
  --config configs/cross_model_benchmark_v1.json \
  --out-dir results/cross_model_YYYY-MM-DD \
  --workers 1 --resume
```

One worker is intentional for 24B laptop runs. More compressor workers increase
throughput on small checkpoints but multiply per-worker tensor scratch and can
exhaust the 19 GB WSL memory limit.
