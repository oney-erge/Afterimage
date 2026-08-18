# Capacity Results — How Big a Model Fits, Measured

**Question answered:** on your actual RTX 3080 Laptop (8 GB VRAM), using
proven quantization (GGUF via Ollama/llama.cpp — not the experimental
Afterimage cache, which Phase 0 already ruled out), what is the largest
model that runs entirely on-GPU, and what happens past that point?

Measured 2026-08-17, two independent runs, same result both times.

---

## The answer

| Resident size (weights + KV @ 8k ctx) | Placement | tok/s |
|---|---|---|
| 3.0 GB (4B) | 100% GPU | 76 |
| 4.2 GB (4B) | 100% GPU | 80 |
| 5.1 GB (7B) | 100% GPU | 62 |
| 5.8 GB (8B) | 100% GPU | 44 |
| **6.2 GB (8B)** | **100% GPU — the largest that fully fits** | **43** |
| **6.9 GB (8B)** | **20% CPU / 80% GPU — spillover begins** | **22 (2.0x slower)** |
| 8.9 GB (12B) | 41% CPU / 59% GPU | 8 (5.4x slower) |

**The boundary is between 6.2 GB and 6.9 GB resident.** Below it, 100% GPU.
Above it, Ollama's own placement (not an estimate — this is llama.cpp
deciding where each layer goes) starts pushing layers to CPU, and speed
drops off a cliff: crossing that line costs 2x immediately, and by 8.9 GB
you're at roughly a fifth of on-GPU speed.

**Practical answer: an 8B model at Q4-class quantization is the largest
model this GPU runs at full speed. A 12B model runs, but at CPU-offload
speeds too slow for interactive use (single-digit tok/s).**

---

## Why this table is trustworthy

`ollama ps`'s `PROCESSOR` column is Ollama/llama.cpp's own placement
decision, not something derived from a timing-sensitive VRAM reading —
**it was byte-identical across two independent sweeps**, run minutes apart,
in a different order of prior GPU state. That reproducibility is the actual
evidence here, not a single number.

**Caveat, stated honestly:** the per-model VRAM *deltas* in the raw JSON are
not clean — `ollama stop` doesn't release CUDA memory as fast as the script's
first version assumed, and a "settle" fix (`scripts/ollama_capacity_test.ps1`)
only partially helped because Ollama's keep-alive left a model resident
across script invocations, contaminating even the "baseline" reading. The
`PROCESSOR` and `SIZE` columns are unaffected by this — they come from
Ollama's internal accounting, not from racing nvidia-smi against driver
cleanup — which is why they're what this table reports.

---

## One data quality note, reported rather than hidden

`qwen3:8b`'s test returned an **empty answer** despite using its full
80-token budget (`eval_count: 80`). This is very likely because Qwen3 emits
a `<think>...</think>` reasoning block by default, and 80 tokens were spent
entirely on that block with none left for the final answer — a token-budget
artifact of the test script, not a quantization quality failure. Every other
model answered the factual check correctly. If Track A/B testing later needs
Qwen3-class reasoning models, raise `num_predict` well past 80 or strip
thinking tags (LocalDeploy's `benchmark.py` already has
`strip_thinking_tags()` for exactly this).

---

## How this relates to the rest of the project

- **This is proven, existing technology** (llama.cpp GGUF quantization via
  Ollama), not the Afterimage cache — Phase 0 already found the cache
  doesn't have enough exploitable structure to matter
  ([PHASE0_RESULTS.md](PHASE0_RESULTS.md)).
- **This table is the bar** [VALIDATION_PLAN.md](VALIDATION_PLAN.md) #2 asks
  any offloading/caching method to beat. An offloaded fp16 8B model
  (Track A) is worth building only if it beats **62 tok/s at 5.1 GB** (7B,
  Q4) or **43 tok/s at 6.2 GB** (8B, Q4) on accuracy at equal-or-lower VRAM —
  otherwise plain quantization already wins and there's nothing to build.
- **Not yet measured:** task accuracy for each of these rows (this test only
  checked one factual question per model, to confirm nothing was obviously
  broken). VALIDATION_PLAN.md #4 specifies the real instrument (paired
  accuracy, n≥400, token-identity, perplexity) — this table establishes
  *what fits*, not yet *what's lost by fitting it*.

---

## Reproducing

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ollama_capacity_test.ps1
```

Swap `-Models` for any tags already pulled (`ollama list`) or new ones
(`ollama pull <tag>` first). Raw per-model JSON:
[ollama_capacity_results.json](ollama_capacity_results.json).
