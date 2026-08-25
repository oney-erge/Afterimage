# What this is, method by method, next to AirLLM

System and evidence-flow diagrams are collected in
[ARCHITECTURE.md](ARCHITECTURE.md).

Qwen3-14B is **29.5 GB**. The GPU has **8 GB**. The model does not fit.

AirLLM and this engine solve that the same way: **stream the model through
the GPU one layer at a time.** Nothing big is ever held.

The cost: to produce **one token** you read the **whole model** off disk.
That is why a token takes tens of seconds, not milliseconds.

```
time per token  ≈  bytes read per token / disk speed  +  decode time
```

Every method below attacks one term. All numbers on this page are the current
bounded, multi-prompt evidence: Qwen3-14B, RTX 3080 Laptop (8 GB), WSL2/CUDA,
cold page cache, **four held-out prompt families × four forced greedy tokens**,
peak VRAM from the same `torch.cuda.max_memory_allocated()` counter for every
row including AirLLM's. Full protocol, per-hypothesis verdicts and raw files:
[ALL_HYPOTHESES_AND_BASELINES.md](ALL_HYPOTHESES_AND_BASELINES.md).

---

## The whole result in one table

| Configuration | Peak VRAM | s/token | Contract | vs AirLLM |
|---|---:|---:|---|---:|
| **AirLLM 3.1.0** (baseline) | 1.583 GB | 28.861 | BF16 greedy | 1.00x |
| Minimum-memory exact | 1.723 GB | 32.514 | exact | **0.89x** |
| + 4 GB residency | 3.934 GB | 17.360 | exact | **1.66x** |
| Chunked output head | **0.901 GB** | 29.606 | **approximate** | 0.97x |
| + fixed-k speculation | 3.813 GB | **9.150** | greedy-token exact at T=0 | **3.15x** |
| + frozen hazard speculation | 3.814 GB | 9.773 | greedy-token exact at T=0 | 2.95x |
| Chunked head + speculation† | 2.056 GB | 7.949 | approximate | 3.70x |

† One factual prompt only, not the 4-prompt suite: a promising low-memory
screen, not a confirmed result.

Every row answered every shared prompt with the same token IDs as the
corrected AirLLM baseline (see "Does the lossy path change the answer?"
below).

---

## Method 1: Compress the weights (lossless)

**What:** Huffman-code the bf16 exponent field. 29.5 GB → 20.3 GB (1.453x,
measured; the proven ceiling for bf16 is ~1.51x).

**Why it should help:** fewer bytes to read per token.

**What it costs:** the GPU must decode the bytes back, work AirLLM never
does, because AirLLM reads raw weights.

**Result: 0.89x at the memory floor, 11% *slower* than AirLLM.**

The saving is spent entirely on decode. This is the single most important
number here and the easiest one to overclaim, so it is stated plainly:
**compression alone, at the same memory, is a loss, not a wash.** The
richer four-prompt suite is less forgiving than the historical one-prompt
screen, which had shown roughly parity (0.89x–1.02x across three repeats;
see the noise table below).

---

## Method 2: Spend spare VRAM on residency

**What:** with more VRAM than the minimum, keep some tensors on the GPU
permanently so they are never re-read.

**Which ones:** rank by `compressed_bytes / original_bytes`. A tensor that
compresses *badly* costs the most disk traffic per byte of VRAM it occupies,
so pin those first. Counterintuitive, and only visible once compression is
on the streaming path.

**Result: 1.66x at 3.93 GB** (2.48x the memory of the minimum-memory row).

**This is where compression finally pays off.** With headroom, decode runs
in big slices and some weights stop being re-read at all. It is not a better
algorithm, it is more memory; the honest framing throughout this project.

A two-prompt placement control that instead kept the *full output head*
resident measured 2.09x at only 2.68 GB, faster than the 4 GB traffic-density
plan despite reading slightly more bytes. That is the clue behind H14/H15
(coalesced/extent-aware residency, see [HYPOTHESIS_LINEAGE.md](HYPOTHESIS_LINEAGE.md)).
Measured since, that clue did not survive: coalescing storage reads made
the engine 27.7% *slower*, because a large contiguous read serialises against
decode instead of overlapping with it.

---

## Method 3: Chunked output head (approximate)

**What:** `lm_head` is 1.556 GB, the single largest tensor, and the floor
*both* engines are stuck on. But logits are a concatenation over vocabulary
rows with no interaction between blocks:

```
logits[..., a:b] = x @ W[a:b].T
```

So compute it a block at a time and never hold the whole thing.

**Result: 0.901 GB, 43% less VRAM than AirLLM's floor, at parity speed
(0.97x).** The only row that beats AirLLM on memory.

**What it costs, and this is the catch:** it is **not bit-exact.** Blocking
changes the matmul's reduction order, and cuBLAS picks a different kernel per
output shape, so bf16 rounding diverges (1–2 absolute in logits).

Padding blocks to a fixed size does **not** fix this:

| rows/block | seq=1 | seq=5 | seq=9 |
|---|---|---|---|
| 75968 (half the vocab) | exact | exact | exact |
| 9496 | differs | differs | differs |
| 1024 | **exact** | differs | differs |

`N=1024` is exact at sequence length 1 and wrong at 5. Kernel choice depends
on the whole (rows, block, hidden) triple unpredictably, and sequence length
changes on every speculative sweep. Only "half the vocab" is reliably exact,
which saves just 0.78 GB.

**So this is a real limit, not an unfinished task.** It ships opt-in and
flips `is_lossless` to `False`, exactly like `q8` already does.

---

## Method 4: Speculative decoding (exact at T=0)

**What:** a small model (Qwen3-0.6B) cheaply guesses the next tokens; the big
model checks all of them **in one pass**. Rejection-sampling math guarantees
the output is exactly what the big model alone would produce. The draft's
quality affects only *speed*, never correctness.

**Why it wins:** one 20 GB read yields several tokens instead of one. It
attacks the *numerator* (reads per token), which no other method here does.

**What it costs:** 1.3 GB of VRAM for the draft model, permanently.

**Result: 9.150 s/token, 3.15x vs AirLLM, exact at greedy decoding (T=0).**
By far the largest legitimate full-suite win here. Effect size varied by
prompt family (1.88x–5.36x observed across the suite), which is why a
four-prompt average, not one lucky prompt, is the number to trust. An
earlier one-prompt screen had reported 12.5x, and that number was real for
that prompt but does not generalise (see
[RESULTS_LOG.md](RESULTS_LOG.md)'s bounded-screen entry for the full history).

A learned rejection-hazard stopping policy (H2) was tested against this fixed
chain length and came in slower (9.773 s/token, 2.95x). See
[HYPOTHESIS_LINEAGE.md](HYPOTHESIS_LINEAGE.md) for why adaptive stopping is
structurally hard to win here: the marginal cost of one more draft token
(tens of milliseconds) is tiny next to a streamed target sweep (seconds), so
"draft the maximum" is close to the economically correct answer regardless of
confidence.

It is also the least novel part of this project: standard draft/verify from
the literature, and speculation over an *offloaded* target specifically is
already established by [SpecExec](https://arxiv.org/abs/2406.02532). The
contribution here is that it composes with compression, residency and the
chunked head.

---

## Method 5: Chunked head + speculation together (approximate)

The chunked head frees ~0.8 GB, which is roughly what the draft model needs.
Speculation then fits in **2.056 GB** instead of the ~3.8 GB floor it has on
its own.

**Result (one prompt only): 7.949 s/token at 2.056 GB, 3.70x vs the matching
AirLLM cell.** This is a promising low-memory screen, not a suite-confirmed
result; it has not yet been measured across all four prompt families.

---

## Does the lossy path actually change the answer?

**No, not once, across the whole 4-prompt suite.** Every method, including
both approximate rows, produced token-identical output to the corrected
AirLLM baseline on every shared prompt.

That is a real practical result and it is **not** a correctness proof: four
prompts, four tokens each. It shows the deviation is small relative to typical
gaps between competing tokens, not that it can never flip one. The config
still declares itself lossy, which is correct.

---

## How much of this is noise?

A historical matched-VRAM pair (the earlier one-prompt protocol) was measured
three separate times:

| run | AirLLM | Method 1 | ratio |
|---|---:|---:|---:|
| 1 | 29.10 | 30.71 | 0.95x |
| 2 | 27.71 | 27.05 | 1.02x |
| 3 | 27.37 | 28.01 | 0.98x |

Mean 0.98x, spread ±4%. The bounded protocol treats **anything under ~10% as
not resolvable on a single run**, which is why the four-prompt screen is
directional rather than confirmatory: it is one repeat, not the five-repeat
protocol [RESEARCH_METHODS.md](RESEARCH_METHODS.md) requires for a performance
claim. The 1.66x, 3.15x and
0.89x results are far outside this band; a result inside it (like Method 1's
0.97x-parity chunked-head row) means parity, not a directional win or loss.

---

## Tried and failed: recorded, not hidden

| Method | Idea | Result |
|---|---|---|
| **Self-speculation** | Draft using the big model's *own* first 4–8 layers, freeing the draft model's 1.3 GB | **0% acceptance** at both depths. An untrained checkpoint's early layers don't predict well enough. LayerSkip's reported 2.16x needs a model *trained* for early exit. |
| **Bandit-tuned k** | Learn how many tokens to guess, live | Never beat a tuned constant. ~7 sweeps per answer is far too few to learn from. |
| **CPU/GPU split decode** | Use the idle 16-core CPU to decode | Passed its isolated gate at 1.33 GB/s, then made the engine **0.52x** once integrated: CPU decode ran inside the same thread as the I/O it was meant to overlap. Removed. |
| **Coalesced storage reads (H14)** | Merge adjacent blob reads into fewer, larger requests | Cut read calls 89% with zero byte amplification, then made the engine **0.72x** (27.7% slower): a large contiguous read serialises against decode instead of overlapping with it. |
| **Neural speculative stopping (H11)** | Learn when to stop drafting from a tiny survival network | **Zero stop decisions** in every calibration run: the break-even survival probability is structurally under ~2% given real costs, so the network is fixed-k in disguise unless genuinely low-confidence positions are sampled. |
| **QUBO residency search (H13/H15)** | Anneal a pairwise-interaction Hamiltonian over resident tensor/extent sets | After making repair eviction-only, a fresh run still returned exactly its control: 730 H13 and 369 H15 evaluations, 0% gain and 100% overlap. |

---

## The honest bottom line

1. **At equal memory and equal losslessness, we currently lose to AirLLM
   (0.89x)** at the four-prompt bounded screen. Compression alone trades disk
   time for decode time, and on this richer suite it does not break even.
2. **Our real lossless advantage is spending memory AirLLM cannot:** 1.66x
   from residency, and **3.15x from speculative decoding.**
3. **Our real memory advantage needs the lossy head:** 43% below AirLLM's
   footprint at parity speed.
4. **The best low-memory operating point (unconfirmed)** is method 5, 3.70x
   at 2.06 GB on one prompt, available only if bit-exactness can be relaxed
   and only after it is measured across the full suite.

What this engine actually offers over AirLLM is not "faster at the same
memory." It is **a dial**: spend more memory for large speedups, or far less
memory for the same speed. AirLLM has neither end of that dial.

---

*Every number is a real run on an RTX 3080 Laptop (8 GB), WSL2, NVMe, cold
caches, measured, not projected. Raw JSON in `results/`, full history
including failures and retractions in [RESULTS_LOG.md](RESULTS_LOG.md).
AirLLM is run with `do_sample=False` and EOS stopping disabled via
`eos_token_id=[]` rather than `min_new_tokens`, which suppresses EOS logits
and would make the baseline answer a different question than the greedy
Afterimage path. An earlier version of this comparison had that bug and the
transcripts were not comparable; the corrected run is the denominator used
throughout.*
