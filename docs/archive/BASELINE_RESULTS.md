# Baseline Results — Measured, Not Estimated

First real rows of the [VALIDATION_PLAN.md](VALIDATION_PLAN.md) table, on the
actual target hardware (RTX 3080 Laptop 8 GB, WSL2/CUDA).

**Status:** measurement harness validated end to end. Baseline rows filled for
Qwen2.5-1.5B. Larger models and the Afterimage/offload rows are the next step.

---

## Qwen2.5-1.5B-Instruct

Measured 2026-08-17. 12 prompts, 32 new tokens each, greedy (`do_sample=False`),
perplexity over 8 held-out texts. `nvidia-smi delta` subtracts the ~1.07 GB the
Windows desktop holds, so it reflects this workload's own cost.

| Config | Checkpoint | torch peak VRAM | nvidia-smi delta | Host RSS peak | Perplexity | tok/s |
|---|---|---|---|---|---|---|
| **fp32 (reference)** | 3.10 GB | 7.82 GB | 6.90 GB | 10.04 GB | 6.1545 | 33.1 |
| **fp16** | 3.10 GB | 4.46 GB | 4.72 GB | 6.96 GB | 6.1510 | 36.5 |

**fp16 vs fp32 reference:**

| Metric | Result |
|---|---|
| VRAM reduction | **−31.6%** (6.90 → 4.72 GB) |
| Perplexity change | **−0.058%** (slightly *better*, i.e. noise) |
| **Token identity** | **100.000%** (360/360 tokens, 12/12 prompts) |
| Speed | **1.10x** (33.1 → 36.5 tok/s) |

---

## What this establishes

**1. The instruments work and are calibrated.** fp16 is the textbook case of a
change that saves real memory with no meaningful quality cost, and all three
instruments agree on that independently: token identity says byte-exact,
perplexity moves 0.058% (well inside noise), memory drops 31.6%. If the harness
had reported degradation here, the harness would be wrong.

**2. Token identity is the sharp instrument, as designed.** 100.000% on 360
tokens is a far stronger statement than "perplexity moved 0.058%" — it says the
two configs produced *literally the same output*. This is the test that Track A
(lossless offloading) must pass, and it will catch any bug that a task-accuracy
suite would miss entirely.

**3. Checkpoint size is not the memory story.** Both rows load the *same* 3.10 GB
checkpoint, yet peak VRAM differs by 3.36 GB — because dtype conversion,
activations, and KV cache dominate. Any claim of the form "the model is N GB"
that quotes checkpoint size alone is not describing what has to fit on the GPU.
This is why the table reports three separate memory columns.

**4. Host RSS moves with VRAM here, and will move *against* it later.** Both
configs load through host RAM, so RSS tracks VRAM. When the offloading rows land,
this column should go *up* while VRAM goes *down* — that is the actual trade
offloading makes, and the table is built to show it rather than hide it.

---

## Reproducing

```bash
# in WSL, with the venv active
bash scripts/baseline_run.sh Qwen/Qwen2.5-1.5B-Instruct

# compare any two runs (the S2 token-identity test)
python scripts/compare_runs.py \
    ~/afterimage/results/baseline_..._fp32.json \
    ~/afterimage/results/baseline_..._fp16.json \
    --require-lossless          # exits non-zero unless token identity is 100%
```

`--require-lossless` makes this usable as a CI gate for Track A: any change that
breaks byte-exactness fails the build rather than silently degrading output.

---

## Next rows to fill

| Row | Blocked on |
|---|---|
| Q8 / Q4_K_M quantized | `bitsandbytes` install, or GGUF via Ollama through LocalDeploy |
| 4B and 8B models | download only — harness is model-agnostic |
| Offload + speculation (Track A) | real-model integration of `runtime/` (currently toy-model only) |
| Afterimage cache (Track B) | same, plus the cache wiring |

The harness itself needs no further work for any of these — it takes
`--model` and `--dtype` and is agnostic to what produced the logits.
