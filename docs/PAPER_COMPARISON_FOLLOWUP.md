# Paper comparison follow-up

**Audit date:** 2026-08-27  
**Purpose:** define the next clean campaign after the current exploratory run.

## Do not cite the current `qwen3-14b-paper-gen` files

The existing files under `results/paper-comparison/` were produced while the
runner and instrumentation were being repaired:

- `1tok` is not paper-eligible and contains the earlier zero-VRAM Accelerate
  measurement plus three DFloat11 OOM cells.
- `32tok` has one usable block and 13 missing or failed cells.
- `128tok` stopped as a partial file after four worker failures.

Keep them as debugging evidence only. Do not merge them with a new run, fill
their missing cells by hand, or use them in a paper figure.

## Headline comparison

Use three external systems that can execute the Qwen3 checkpoint on the same
host:

1. [AirLLM](https://github.com/lyogavin/airllm), the established low-VRAM
   layer-streaming baseline.
2. [Hugging Face Accelerate Big Model Inference](https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling),
   which dispatches weights across GPU, CPU, and disk.
3. [DeepSpeed ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html),
   configured for ZeRO-3 NVMe parameter offload because the 29.5 GB BF16
   checkpoint is larger than the reference host's 19 GiB RAM.

Compare those with `exact-min`, `exact-resident`, and `spec-fixed`. DFloat11
remains available as an opt-in capacity appendix. Its OOM on this 8 GB GPU is
an applicability boundary, not a latency baseline. Do not place an empty
"failed" DFloat11 row in the headline performance table.

SpecExec is relevant prior work for tree-based speculative decoding, but it is
not a drop-in package baseline for this matrix. A valid comparison requires a
SpecExec implementation using the same target, draft model, prompts, greedy
decoding contract, output lengths, and hardware. Until that path exists, cite
the method and compare the speculative mechanics, not an invented runtime row.

## Clean restart

After the current source changes are committed and the benchmark environment
contains `.[bench]`, run:

```bash
bash paper_benchmark.sh
```

The wrapper now performs two restartable campaigns:

| Suite | Output lengths | Purpose |
|---|---:|---|
| `evaluation` | 1, 4 | Literal TTFT and short cold-start behavior on exact-answer prompts |
| `paper_generation` | 1, 32, 128 | Paired TTFT, marginal decode rate, and sustained generation on prompts that require long answers |

Every token length uses randomized method order, isolated worker processes,
untimed warmup, cache-drop records, thermal monitoring, external VRAM sampling,
host-RSS high-water marks, token IDs, initialization time, process read bytes,
and inclusive energy estimates based on actual monotonic sample spacing.
`--require-complete` prevents a partial matrix from becoming an immutable
headline result. `--resume` reruns only incomplete non-capacity cells.

## Additional runs after the clean pilot

Do not start these until the three-block pilot finishes and its variance is
known.

1. **Confirmatory blocks.** Use the pilot block log-ratios with
   `scripts/power_analysis.py`; expect 8 to 12 randomized blocks, but set the
   final count from the observed variance before looking at the confirmation.
2. **Matched memory.** Add `--vram-budgets 2,3,4` so Afterimage and Accelerate
   are compared at explicit memory budgets. Plot unmatched systems only on the
   measured VRAM-latency Pareto frontier.
3. **Input-length TTFT.** Add a separate 1-token run at controlled prompt
   lengths such as 128, 512, and 2,048 input tokens. This remains a specific
   follow-up because the current four prompt families are semantically varied,
   not token-length controlled.
4. **Speculation detail.** Report accepted tokens per target pass, acceptance
   by position, draft time, target verification time, and end-to-end time for
   `spec-fixed`. A speed number without these mechanism measurements is hard to
   interpret.
5. **Ablation.** Present compression-only (`exact-min`), added residency
   (`exact-resident`), and added speculation (`spec-fixed`) as a cumulative
   mechanism table. This separates what each component contributes.
6. **Model transfer.** Repeat the frozen protocol on at least one smaller model
   and one second architecture. Do not retune policies on the reported prompts.

## Figures and tables

The final paper should lead with measurements, not architecture decoration:

- measured peak VRAM versus seconds per output token, with the Pareto frontier;
- TTFT and marginal decode tokens per second from paired 1-token and 128-token
  runs on the same generation prompts;
- speed versus output length for 1, 32, and 128 tokens;
- compression ratio and exact round-trip checks by model family;
- speculation acceptance and stage-time breakdown;
- median with one point per randomized block, plus raw block values;
- initialization time, host RAM, process read bytes, thermal integrity, and
  inclusive energy in a secondary systems table.

Do not draw a headline point when peak VRAM is unknown, any required block is
missing, the run throttled, or a method with an exactness contract diverged from
the control.
