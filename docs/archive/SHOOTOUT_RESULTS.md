# Compression Shootout — Afterimage vs. the Competition

Same 6 real layers of Qwen2.5-1.5B, same activations, same metric
(mean relative output error `||Ŵx − Wx|| / ||Wx||`), so compression ratio and
error are directly comparable across every scheme. Measured 2026-08-17.

---

## 1. Where AirLLM sits (the reference you asked for)

**AirLLM does not compress at all.** It is a different axis entirely:

| Method | Bytes stored | Peak VRAM | Output error |
|---|---|---|---|
| **AirLLM (layer streaming)** | **100%** — full weights on disk | ~1 layer | **0.00%** |
| Afterimage (rank 128) | 9.8% | 9.8% | **59.22%** |
| Q4_K_M-class quantization | 26% | 26% | 9.97% |

AirLLM trades **speed** for VRAM, not accuracy — it streams the full,
unmodified weights layer by layer, so its output is bit-identical to the
original model. It is slow (every token re-reads the whole model from disk)
but it is *lossless*.

Afterimage trades **accuracy** for VRAM. That is a fundamentally different and
much worse bargain, because you can always buy VRAM back with speed
(AirLLM/offloading) but you cannot buy accuracy back once the weights are gone.

---

## 2. The full table

| Scheme | Compression | Output error |
|---|---|---|
| Quant grouped-64 **8-bit** (~Q8_0) | 1.9× | **0.55%** |
| LowRank r=128 + 4-bit residual *(my Fix 2)* | 2.9× | 14.93% |
| **Quant grouped-64 4-bit (~Q4_K_M)** | **3.8×** | **9.97%** |
| LowRank r=128 + 3-bit residual *(my Fix 2)* | 3.5× | 32.00% |
| LowRank r=128 + 2-bit residual *(my Fix 2)* | 4.5× | 62.99% |
| Quant grouped-64 **3-bit** | 4.9× | **23.12%** |
| Quant grouped-64 3-bit + 0.1% fp16 outliers | 4.9× | 22.07% |
| PCA projection r=256 *(original Afterimage)* | 5.1× | 49.54% |
| LowRank r=64 + 2-bit residual *(my Fix 2)* | 5.7× | 67.87% |
| Quant grouped-64 **2-bit** | 7.1× | 60.82% |
| Quant grouped-64 2-bit + 0.1% fp16 outliers | 7.0× | 58.78% |
| **PCA projection r=128** *(original Afterimage)* | **10.2×** | **59.22%** |
| Act-weighted SVD r=256 *(my Fix 1)* | 5.1× | 60.18% |
| Act-weighted SVD r=128 *(my Fix 1)* | 10.2× | 68.23% |

**Read it at matched compression:**

| At ~5× | error | | At ~7-10× | error |
|---|---|---|---|---|
| Quant 3-bit | **23.12%** | | Afterimage PCA r=128 (10.2×) | **59.22%** |
| Afterimage PCA r=256 | 49.54% | | Quant 2-bit (7.1×) | 60.82% |
| LowRank+2bit (4.5×) | 62.99% | | | |

**Quantization wins decisively in the usable range (≤5×).** Above ~7×,
low-rank marginally edges out 2-bit quantization — but at 59% vs 61% error
both are unusable, so that crossover is worthless in practice.

---

## 3. Two things I got wrong, corrected by the data

### Fix 1 (activation-weighted SVD) made it WORSE — 68.23% vs 59.22%

I proposed weighting the subspace fit by activation magnitude, on the theory
that PCA optimizes the wrong objective. It passed a synthetic test and
**failed on the real model**, consistently, at both ranks.

**Why, and it's the interesting part:** weighting by activation RMS spends the
rank budget on the highest-magnitude channels. In real transformers those are
the *massive-activation / rogue dimensions* — the ones
[PHASE0_RESULTS.md](PHASE0_RESULTS.md) already showed carry high variance but
**low functional importance**. So activation weighting actively amplifies the
exact pathology it was meant to fix. My synthetic test was misleading because
there the large channels genuinely mattered; in a real transformer they don't.

This is a concrete, measured instance of "variance ≠ importance," and it
retires the whole family of magnitude-weighted subspace fixes.

### My original quantization baseline was unfairly weak

The first run used per-row scales (one scale across 8960 weights), giving
4-bit = 18.87%. Real Q4_K_M uses **group-wise** scales. With groups of 64:
4-bit drops to **9.97%** — nearly 2× better. The competitor was stronger than
I first reported, which makes Afterimage's position *worse*, not better.

---

## 4. Fix 2 (low-rank + quantized residual) works, but doesn't win

It behaves exactly as the theory predicted — error falls monotonically as
residual bits increase (62.99% → 32.00% → 14.93%), confirming that *storing*
the residual instead of discarding it is the right idea. Discarding it is
indeed what makes plain projection catastrophic.

But it never beats pure quantization at matched compression: 14.93% @ 2.9×
loses to 9.97% @ 3.8×. The low-rank component is simply not earning its
memory — those same bytes spent on finer quantization buy more.

---

## 5. Why low-rank cannot win here — the fundamental reason

Low-rank approximation is only efficient when the singular value spectrum
decays fast. Phase 0 measured the actual effective rank of these activations:
**228–990 out of 8960**, depending on workload. A rank-128 basis is *below the
intrinsic dimensionality of the data*, so it is mathematically forced to throw
away real signal — no amount of choosing the basis more cleverly fixes that,
which is precisely what Fix 1's failure demonstrated.

Quantization has no such constraint: it spends its budget uniformly across
*all* dimensions and never discards a direction entirely. For high-rank,
heavy-tailed data — which is what transformer weights are — that is simply the
better allocation.

---

## 6. What would actually reduce the error

Ranked by expected gain, all pointing the same direction — **the frontier is
better quantization, not subspace caching**:

1. **Finer groups + outlier handling** (measured above; groups of 32 and
   proper outlier extraction go further). Standard practice, already gets
   4-bit to ~10%.
2. **GPTQ-style error compensation** — quantize column by column, using the
   inverse Hessian to push each rounding error into the *not-yet-quantized*
   weights so it cancels. Typically halves error at the same bit budget. This
   is the single highest-value unimplemented item here.
3. **Codebook / vector quantization** (AQLM, QuIP#) — the current state of the
   art, reaching genuinely usable quality at 2–3 bits by quantizing *groups of
   weights jointly* rather than each weight independently. This is where the
   ~8× compression Afterimage was chasing actually lives.
4. **Sensitivity-weighted bit allocation across layers** — the shootout shows
   per-layer error varies widely; uniform bit-width across layers is leaving
   quality on the table.

**Recommendation:** stop investing in the activation-subspace cache. If the
goal is high compression at low error, implement GPTQ error compensation next
(cheap, well-understood, large measured gains), then evaluate AQLM/QuIP#-class
codebook quantization. If the goal is running big models on 8 GB *without*
quality loss, use offloading (AirLLM/speculation), which is lossless by
construction.

---

## 7. Caveats

- Metric is relative L2 error on layer output, not task accuracy. It is the
  right metric for comparing approximations at fixed memory, but a given L2
  error does not translate linearly into benchmark score.
- 6 `down_proj` layers of a 1.5B model, not a full model end to end.
- My grouped quantizer is a faithful but simplified stand-in for GGUF
  k-quants (which add super-block scales and non-uniform bit allocation);
  real Q4_K_M is likely slightly *better* than the 9.97% shown.
- Raw data: [shootout.json](shootout.json) · reproduce with
  `bash scripts/shootout_run.sh`.
