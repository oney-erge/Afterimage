# Validation Plan — Memory vs. Accuracy, Measured

**Purpose:** produce one table you can look at and say "this works" or "this
doesn't," without needing to interpret functional-error curves.

## 0. The answer, up front

> **Correction to an earlier version of this document:** it originally claimed
> Phase 0 answered a mechanism question but not the memory-vs-accuracy
> question. That was wrong — Phase 0's rank and functional-error numbers
> convert directly into memory and accuracy, and the table below is that
> conversion, computed from the same real measurement on Qwen2.5-1.5B
> ([PHASE0_RESULTS.md](PHASE0_RESULTS.md)).

**One real layer (`down_proj`), before and after Afterimage's compression:**

| Compression | Memory per layer | Output error (6 layers cached together, end to end) |
|---|---|---|
| None (full weight) | 27.5 MB | 0% — reference |
| Afterimage, **164× smaller** (rank 8) | 0.17 MB | **76–96%** |
| Afterimage, **41× smaller** (rank 32) | 0.67 MB | **69–73%** |
| Afterimage, **10× smaller** (rank 128) | 2.7 MB | **60–71%** |

**Accuracy loss does not scale with compression ratio — it is catastrophic at
every ratio tested, including the gentlest (10×).** There is no dial that
trades a little accuracy for a lot of memory; it is a cliff, not a slope, and
the lightest compression tested is already past it. An "8B model shrunk to
feel like 2B" does not degrade gracefully — 59–96% output error means most
generated output is simply wrong, not "slightly less sharp."

**Scope of this specific number:** one layer type, 6 of 28 layers cached
simultaneously, 1.5B model. Not yet run on a 4B/8B model end to end — that
requires the full offloading runtime, which this NO-GO result means is not
worth building (see [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md)).
Nothing in the measurement suggests larger models compress better; the
adversarial-workload result in Phase 0 points the other way if anything.

The rest of this document is the plan for iso-memory comparisons against
quantization — useful for validating the fallback (residency + speculation,
no cache), but **the cache itself is already answered by the table above.**

---

## 1. The reframe that makes the table decision-legible

"Method X saves memory with only a small accuracy loss" is a meaningless claim
in isolation, because **quantization already does that, for free, today.** If
Afterimage gives 2x memory reduction at 5% accuracy loss but Q4 gives 3.5x at
1% loss, Afterimage is worthless despite sounding good in isolation.

So every experiment below is **iso-memory**: fix the VRAM budget, then ask
which method delivers the best accuracy and speed inside it.

> **The question: at a fixed ~6.5 GB VRAM budget on your RTX 3080, what is the
> most accurate way to run an 8B model — and does any offloading/caching method
> beat plain Q4 quantization?**

If nothing beats Q4, that is a complete and useful answer, and the project ends
with a defensible negative result rather than an open question.

---

## 2. The output table (this is the deliverable)

One of these per model. Every cell measured, none estimated.

### `Qwen3-8B` @ 6.5 GB VRAM budget

| # | Config | Weights on disk | **Peak VRAM** | Task acc. (n=400) | Δ acc vs ref | Token-identity vs ref | PPL | tok/s |
|---|---|---|---|---|---|---|---|---|
| R | fp16 reference (no budget cap) | 16.4 GB | ~17.5 GB | **baseline** | — | 100% by definition | — | — |
| 1 | Q8_0 | 8.2 GB | ~9.0 GB | | | | | |
| 2 | **Q4_K_M (the bar to beat)** | 4.7 GB | ~5.5 GB | | | | | |
| 3 | Q4 + offload + speculation (Track A) | 4.7 GB | | | **must be 0.00** | **must be 100%** | | |
| 4 | fp16 + offload + speculation (Track A) | 16.4 GB | | | **must be 0.00** | **must be 100%** | | |
| 5 | Afterimage cache r=256 (Track B) | | | | | | | |
| 6 | Afterimage cache r=1024 (Track B) | | | | | | | |

**How to read it:**

- **Row 2 is the bar.** Anything that doesn't beat Q4_K_M on accuracy *at the
  same or lower peak VRAM* is not worth shipping.
- **Rows 3–4 are lossless by construction.** Exact weights, exact speculative
  sampling. Token-identity **must** be 100% and Δacc **must** be 0.00. Anything
  else is a bug in our implementation, not a property of the method. This makes
  them a self-checking correctness test, not just a benchmark.
- **Row 4 is the actual value proposition:** fp16-quality 8B inside a 6.5 GB
  budget, which Q4 cannot do at any speed. The cost is tok/s. If row 4 lands at
  a usable speed with 0.00 accuracy delta, that is a real, shippable win *even
  though Phase 0 failed*, because it doesn't depend on the cache at all.
- **Rows 5–6 are the cache's last fair chance.** Phase 0 predicts they fail
  badly. Running them costs little now that the harness exists, and a measured
  accuracy number is far more convincing to a skeptic than a functional-error
  curve.

Repeat for `Qwen3-4B` and `Gemma-3-4B` (4B class), where fp16 is ~8 GB and the
budget pressure is milder — useful as a control showing the method's behaviour
when the model *almost* fits.

---

## 3. Models

| Model | Params | fp16 | Q8 | Q4_K_M | Role |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | 3.1 GB | 1.5 GB | 1.0 GB | fast iteration; already downloaded |
| Gemma-3-4B | 4.3B | 8.6 GB | 4.3 GB | 2.6 GB | 4B class; already a LocalDeploy default profile |
| Qwen3-4B | 4.0B | 8.0 GB | 4.0 GB | 2.4 GB | 4B class, second family (guards against one-model artifacts) |
| Qwen3-8B | 8.2B | 16.4 GB | 8.2 GB | 4.7 GB | **the headline case** |
| Llama-3.1-8B-Instruct | 8.0B | 16.0 GB | 8.0 GB | 4.6 GB | 8B class, second family |

Size columns are **arithmetic estimates to be replaced by measured file sizes.**
Two model families per size class is deliberate: a result that holds on Qwen but
not Llama is an artifact, not a finding.

---

## 4. Measuring accuracy honestly — the part that's easy to get wrong

You asked to prove "no deterioration." Task accuracy alone **cannot** prove that
at small sample sizes, and this trips up most published comparisons.

### 4.1 Statistical power

> **Correction:** an earlier draft of this section quoted noise floors roughly
> 2x too optimistic. The numbers below are the corrected ones, verified in code
> against the textbook reference point (detecting 5 pp at p=0.5, 80% power
> requires n ≈ 1570 per group) and guarded by
> `tests/test_accuracy_instruments.py` so the doc and code cannot drift apart
> again.

**Unpaired** (each system measured on a different sample) — the pessimistic bound:

| n (questions) | smallest detectable Δ accuracy (80% power) |
|---|---|
| 50 | **~28 pp** |
| 100 | ~20 pp |
| 400 | ~10 pp |
| 1000 | ~6 pp |
| 1570 | ~5 pp |

**A 50-question suite showing "72% vs 70%" is indistinguishable from noise** —
the noise floor there is 28 pp. Reporting that as "only 2% loss" would be
actively misleading.

**Paired** (both systems answer the *same* questions) — dramatically better,
and the reason `n = 400` is a workable budget rather than an inadequate one:

McNemar's test looks only at *discordant* pairs. When the candidate is an
approximation of the reference, it should only ever get things **wrong** that
the reference got right, never the reverse — so all discordant pairs point the
same direction, and the statistic clears significance at roughly **6 discordant
pairs**. Detecting a regression affecting fraction `δ` of items therefore needs
`δ·n ≳ 6`:

| n | smallest detectable one-sided regression |
|---|---|
| 400 | **~1.5 pp** |
| 1000 | ~0.6 pp |

So the rules:

- **Always run paired** — same questions, same order, both configs. Never
  compare two independently-sampled accuracy rates.
- **n ≥ 400** for any headline accuracy claim (paired: ~1.5 pp sensitivity).
- **Check `cand_only` before trusting the paired figure.** If the candidate is
  also *improving* on items the reference got wrong, discordant pairs no longer
  point one way, power collapses back toward the unpaired bound, and you need
  the larger `n`. `PairedAccuracyResult.cand_only` reports this directly.
- State the confidence interval in every cell. `72.8% ± 4.4` is honest;
  `72.8%` alone is not.

### 4.2 Three instruments, in order of sensitivity

Task accuracy is what you *care* about but it is the *least* sensitive detector.
Use all three:

1. **Token-identity rate** (most sensitive). Greedy decode (`temperature=0`),
   same prompts, compare output token-by-token against the reference. For any
   lossless config this must be **100.000%** — a single differing token is a
   real bug. Detects deviations far too small for task accuracy to see, on a few
   hundred prompts.
2. **Perplexity** on held-out text (continuous, sensitive, cheap). Catches
   degradation that hasn't yet flipped any answer.
3. **Task accuracy** via LocalDeploy's graded suite (what you actually care
   about, noisiest). Report with CIs.

A method that keeps task accuracy but wrecks perplexity is degrading and will
fail on harder prompts later. Reporting all three prevents that from hiding.

### 4.3 Determinism

Everything at `temperature=0`, fixed seeds, fixed prompt set committed to the
repo. Speculative decoding is only *distributionally* exact at `T>0`, so also
run a `T=0.7` pass and compare output **distributions** (KL over ~10k positions),
not individual strings — at `T>0` strings legitimately differ.

---

## 5. Measuring memory honestly

Report **three separate numbers**; conflating them is the usual way this gets
overstated.

| Number | Meaning | How |
|---|---|---|
| Weights on disk | what the checkpoint costs to store | `du -b` on the model file |
| **Peak VRAM** | what actually has to fit in the GPU — *the number that decides whether it runs* | `torch.cuda.max_memory_allocated()` **and** `nvidia_smi_used_mb()` sampled at 100 ms during the run, report the max of the sampled series |
| Host RAM high-water | what offloading pushes onto the CPU side | RSS high-water |

**Offloading does not reduce total bytes — it reduces peak VRAM by moving bytes
to RAM/NVMe.** Rows 3–4 will show low peak VRAM and *high* host RAM. That is the
honest trade and the table must show both columns or it is misleading.

Also pin: same context length, same `max_output_tokens`, same batch size across
every row. KV cache scales with context and will otherwise silently dominate the
VRAM column and invalidate the comparison.

---

## 6. LocalDeploy integration — use it, don't rebuild it

LocalDeploy already has everything the accuracy half of this needs, verified by
reading the source:

- `localdeploy/backends/openai_compatible.py` speaks standard
  `POST /v1/chat/completions` — so **any runtime exposing that endpoint is
  benchmarkable with zero new harness code.**
- `localdeploy/benchmark.py` provides graded test cases with real scoring
  functions (`builtin_test_cases()`, `build_test_cases()` for custom sets),
  `iter_run()` to sweep profiles, `category_summary()`, and `write_reports()`.
- `nvidia_smi_used_mb()` already captures VRAM, and `gpu_info()` captures the
  hardware context.
- Profiles are plain JSON in `config.json` with `backend`, `model_id`,
  `base_url`, `context_limit`, `max_output_tokens`, `temperature`.

### 6.1 What to build (small)

**One FastAPI shim** exposing `/v1/chat/completions` in front of the Afterimage
runtime. That is the entire integration. Then register profiles:

```jsonc
{
  "afterimage_qwen3_8b_offload": {
    "name": "afterimage_qwen3_8b_offload",
    "backend": "openai_compatible",
    "model_id": "Qwen3-8B",
    "base_url": "http://127.0.0.1:8djust",   // the Afterimage shim
    "enabled": true,
    "context_limit": 4096,
    "max_output_tokens": 768,
    "temperature": 0.0,
    "timeout_seconds": 600                    // offloading is slow; don't let
                                              // the harness score a timeout as
                                              // a wrong answer
  },
  "ollama_qwen3_8b_q4": {
    "backend": "ollama",
    "model_id": "qwen3:8b",
    "base_url": "http://127.0.0.1:11434",
    "context_limit": 4096,
    "max_output_tokens": 768,
    "temperature": 0.0
  }
}
```

Then every row of the table is a profile, and LocalDeploy sweeps them and writes
the comparison report. **Do not rebuild accuracy grading, VRAM sampling, or
reporting** — it exists and is already exercised.

### 6.2 What LocalDeploy does *not* give you

- **GB transferred per accepted token** — Afterimage's own
  `bench/iocount.py` covers this; LocalDeploy is throughput-oriented and
  doesn't track I/O volume.
- **Token-identity rate** — needs a direct comparison harness (§4.2), roughly
  40 lines against two endpoints.
- Set `timeout_seconds` generously for offloaded rows. LocalDeploy's
  `_is_oom()`/timeout handling will otherwise mark a slow-but-correct config as
  failed and silently deflate its accuracy.

---

## 7. Execution order, with kill gates

Ordered so the cheapest experiment that can falsify the most is first.

| Step | What | Effort | Gate |
|---|---|---|---|
| **S1** | Build the OpenAI-compatible shim; register one Afterimage profile and one Ollama profile for the *same* 1.5B model; run LocalDeploy end to end | 1 day | Harness produces a populated table. Nothing measured yet — this only proves the pipeline works |
| **S2** | **Token-identity test on Track A at 1.5B.** Q4 weights, offload + speculation, greedy, 200 prompts vs. the un-offloaded reference | 1 day | **Must be 100.000%.** If not, we have a correctness bug and every later number is meaningless. This is the single highest-value test in the plan |
| **S3** | Fill the full table for **Qwen3-4B** (rows R, 1, 2, 3, 4) | 2 days | Does row 4 (fp16 offloaded) fit the budget with 0.00 Δacc? If yes, the product works |
| **S4** | Fill the full table for **Qwen3-8B** + **Llama-3.1-8B** | 2 days | Does the row-4 result hold across families and at 8B? Is tok/s usable (≥3 tok/s)? |
| **S5** | **Track B final test:** rows 5–6, cache at r=256 and r=1024, on 4B | 2 days | Does it beat Q4_K_M on accuracy at equal-or-lower peak VRAM? Phase 0 says no. **If no → close Track B permanently and delete the cache from the runtime** |
| **S6** | Scale-up + write-up | 2 days | — |

**Total ~10 days.** S2 alone is worth doing even if nothing else happens — it
either validates or invalidates the entire lossless-offloading claim in a day.

---

## 8. Decision rules, written before the data

Committing to these now so the result can't be rationalised afterward.

**Track A (offloading + speculation) ships if:**
- Token-identity = 100.000% at greedy on ≥200 prompts, **and**
- Peak VRAM for fp16-weights-offloaded ≤ Q4's peak VRAM, **and**
- Task accuracy ≥ Q4's accuracy (it should *exceed* it — fp16 weights vs Q4
  weights — and this is the whole point), **and**
- ≥ 3 tok/s sustained on the 8B at 4k context.

**Track B (Afterimage cache) is closed permanently unless:**
- At equal peak VRAM to Q4_K_M, task accuracy is **within 2 pp** of Q4 with
  n ≥ 400, **and**
- Perplexity degradation < 5%.

Anything less and Q4 simply wins, and the honest conclusion is that the cache is
a worse tool than quantization for the same job.

**If both tracks fail**, the finding is: *on consumer 8 GB hardware, Q4
quantization is already at the efficient frontier for 4B–8B models, and neither
activation-subspace caching nor weight offloading improves on it.* That is a
genuinely useful negative result — it tells you to stop looking here — and the
measurement infrastructure built along the way is reusable for the next idea.

---

## 9. Expected outcome, stated honestly in advance

- **Track A: likely passes.** It's built from published, reproduced methods
  (SubSpec/SpecExec-class), it's lossless by construction, and the only real
  risk is that tok/s is too slow to be pleasant. Expect the win to be
  *quality-at-fixed-VRAM* (fp16 8B in 6.5 GB), not raw speed.
- **Track B: likely fails,** and Phase 0 already showed why. It's in the plan
  because it's cheap once the harness exists, and because a measured
  accuracy-vs-Q4 number is a far more durable answer than a rank curve.

The plan is designed so Track A's success does not depend on Track B, and so
the whole thing produces a defensible answer either way.
