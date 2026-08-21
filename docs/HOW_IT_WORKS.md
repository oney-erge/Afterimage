# What this is, method by method, next to AirLLM

Qwen3-14B is **29.5 GB**. The GPU has **8 GB**. The model does not fit.

AirLLM and this engine solve that the same way: **stream the model through
the GPU one layer at a time.** Nothing big is ever held.

The cost: to produce **one token** you read the **whole model** off disk.
That is why a token takes ~27 seconds, not milliseconds.

```
time per token  ≈  bytes read per token / disk speed  +  decode time
```

Every method below attacks one term. Same prompt, same 8 tokens, cold cache,
peak VRAM from the same counter for every row including AirLLM's.

---

## The whole result in one table

| # | Method | VRAM | s/token | Lossless | vs AirLLM | VRAM vs AirLLM |
|---|---|---|---|---|---|---|
| — | **AirLLM (baseline)** | 1.568 GB | 27.37 | yes | 1.00x | — |
| 1 | Compression only | 1.677 GB | 28.01 | yes | **0.98x** | +7% |
| 2 | + residency @2.1 GB | 1.990 GB | 17.20 | yes | 1.59x | +27% |
| 2 | + residency @4.0 GB | 3.888 GB | 15.22 | yes | 1.80x | +148% |
| 3 | + chunked head | **0.855 GB** | 26.38 | **NO** | 1.04x | **−45%** |
| 4 | + speculation @4 GB | 3.784 GB | **2.19** | yes | **12.5x** | +141% |
| 5 | chunked head + speculation | 2.051 GB | 3.28 | **NO** | 8.3x | +31% |

Every row answered the prompt identically:
*"What is the capital of France?"* → *" The capital of France is Paris. It"*

---

## Method 1 — Compress the weights (lossless)

**What:** Huffman-code the bf16 exponent field. 29.5 GB → 20.3 GB (1.453x).

**Why it should help:** fewer bytes to read per token.

**What it costs:** the GPU must decode the bytes back — work AirLLM never
does, because AirLLM reads raw weights.

**Result: 0.98x. Parity. It does not beat AirLLM.**

Disk reads really do halve (14.7 vs 29.2 GB/token, measured). The saving is
entirely spent on decode. This is the single most important number here and
the easiest one to overclaim, so it is stated plainly: **compression alone,
at the same memory, is a wash.**

---

## Method 2 — Spend spare VRAM on residency

**What:** with more VRAM than the minimum, keep some tensors on the GPU
permanently so they are never re-read.

**Which ones:** rank by `compressed_bytes / original_bytes`. A tensor that
compresses *badly* costs the most disk traffic per byte of VRAM it occupies,
so pin those first. Counterintuitive, and only visible once compression is
on the streaming path.

**Result: 1.59x at 2.0 GB, 1.80x at 3.9 GB.**

**This is where compression finally pays off** — with headroom, decode runs
in big slices and some weights stop being re-read at all.

**Honest framing:** this is not a better algorithm, it is more memory.
Note the diminishing return — doubling 2.0→3.9 GB buys only 13%.

---

## Method 3 — Chunked output head (LOSSY)

**What:** `lm_head` is 1.556 GB — the single largest tensor, and the floor
*both* engines are stuck on. But logits are a concatenation over vocabulary
rows with no interaction between blocks:

```
logits[..., a:b] = x @ W[a:b].T
```

So compute it a block at a time and never hold the whole thing.

**Result: 0.855 GB — 45% less VRAM than AirLLM — at parity speed (1.04x).**
The only row that beats AirLLM on memory.

**What it costs — and this is the catch:** it is **not bit-exact.** Blocking
changes the matmul's reduction order, and cuBLAS picks a different kernel per
output shape, so bf16 rounding diverges (1-2 absolute in logits).

I tried to fix this by padding blocks to a fixed size. **It does not work:**

| rows/block | seq=1 | seq=5 | seq=9 |
|---|---|---|---|
| 75968 (half the vocab) | exact | exact | exact |
| 9496 | differs | differs | differs |
| 1024 | **exact** | differs | differs |

`N=1024` is exact at sequence length 1 and wrong at 5. Kernel choice depends
on the whole (rows, block, hidden) triple unpredictably — and sequence length
changes on every speculative sweep. Only "half the vocab" is reliably exact,
which saves just 0.78 GB.

**So this is a real limit, not an unfinished task.** It ships opt-in and
flips `is_lossless` to False, exactly like `q8` already does.

---

## Method 4 — Speculative decoding (lossless)

**What:** a small model (Qwen3-0.6B) cheaply guesses the next k tokens; the
big model checks **all k in one pass**. Rejection-sampling math guarantees
the output is exactly what the big model alone would produce — the draft's
quality affects only *speed*, never correctness.

**Why it wins:** one 20 GB read yields several tokens instead of one. It
attacks the *numerator* — reads per token — which no other method here does.

**What it costs:** 1.3 GB of VRAM for the draft model, permanently.

**Result: 2.19 s/token — 12.5x vs AirLLM, fully lossless.** By far the
largest legitimate win in this project.

It is also the least novel part — standard draft/verify from the literature.
The engineering contribution is that it *composes* with everything else.

---

## Method 5 — Both levers together (LOSSY)

The chunked head frees ~0.8 GB, which is roughly what the draft model needs.
So speculation now fits in **2.05 GB** instead of the ~3 GB floor it had
before — a configuration that did not exist prior to this pass.

**Result: 3.28 s/token at 2.051 GB — 8.3x vs AirLLM at +31% VRAM**, versus
12.5x at +141%. On an 8 GB card this is arguably the most useful operating
point, *if* the bit-exactness requirement can be relaxed.

---

## Does the lossy path actually change the answer?

**No — not once, in this test.** Every method, including both LOSSY rows,
produced **token-identical output** to the lossless greedy path.

That is a real practical result and it is **not** a correctness proof: one
prompt, 8 tokens. It shows the deviation is small relative to typical gaps
between competing tokens, not that it can never flip one. The config still
declares itself lossy, which is correct.

---

## How much of this is noise?

The matched-VRAM pair was measured three separate times:

| run | AirLLM | Method 1 | ratio |
|---|---|---|---|
| 1 | 29.10 | 30.71 | 0.95x |
| 2 | 27.71 | 27.05 | 1.02x |
| 3 | 27.37 | 28.01 | 0.98x |

Mean 0.98x, spread ±4%. **At 8 tokens on a single run, anything under ~10%
is not resolvable.** So the 1.59x, 1.80x, 8.3x and 12.5x results are real
(far outside the band); the 0.98x and 1.04x rows mean *parity*, and should
not be reported as a win in either direction.

---

## Tried and failed — recorded, not hidden

| Method | Idea | Result |
|---|---|---|
| **Self-speculation** | Draft using the big model's *own* first 4-8 layers, freeing the draft model's 1.3 GB | **0% acceptance** at both depths. An untrained checkpoint's early layers don't predict well enough. LayerSkip's 2.16x needs a model *trained* for early exit. |
| **Bandit-tuned k** | Learn how many tokens to guess, live | Never beat a tuned constant. ~7 sweeps per answer is far too few to learn from. |
| **CPU/GPU split decode** | Use the idle 16-core CPU to decode | Passed its isolated gate at 1.33 GB/s, then made the engine **0.52x** — CPU decode ran inside the same thread as the I/O it was meant to overlap. Removed. |

---

## The honest bottom line

1. **At equal memory and equal losslessness, we are at parity with AirLLM
   (0.98x).** Compression alone buys nothing in wall-clock; it trades disk
   time for decode time.
2. **Our real lossless advantage is that we can spend memory AirLLM cannot**
   — 1.6-1.8x from residency, and **12.5x from speculative decoding.**
3. **Our real memory advantage needs the lossy head** — 45% below AirLLM's
   footprint at parity speed.
4. **The best all-round operating point is method 5** — 8.3x at 2.05 GB —
   available only if bit-exactness can be relaxed.

What this engine actually offers over AirLLM is not "faster at the same
memory." It is **a dial**: spend more memory for large speedups, or far less
memory for the same speed. AirLLM has neither end of that dial.

---

*Every number is a real run on an RTX 3080 Laptop (8 GB), WSL2, NVMe, cold
caches, measured — not projected. Raw JSON in `results/`, full history
including failures and retractions in [RESULTS_LOG.md](RESULTS_LOG.md).*

*AirLLM is run with `do_sample=False`. Without it, HuggingFace `generate()`
honours Qwen3's sampling config and the baseline answers a different
question than our greedy path — an earlier version of this comparison had
that bug and the transcripts were not comparable.*
