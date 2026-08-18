# Engine Evaluation — Math, Novelty, Applicability, vs AirLLM

All numbers measured on the RTX 3080 Laptop / Qwen2.5-1.5B unless marked
PROJECTED. Updated 2026-08-17.

---

## 1. Can we get 10-20x smaller? No. Here is the proof.

A bf16 weight is 16 bits: `[sign:1][exponent:8][mantissa:7]`.
Measured entropy of each field, over 197 real layers:

| Field | Bits allocated | Measured entropy | Compressible? |
|---|---|---|---|
| sign | 1 | ~1.0 | no |
| mantissa | 7 | **6.97** | no — effectively uniform |
| exponent | 8 | **2.595** | **yes, 5.4 bits recoverable** |
| **total** | **16** | **~10.57** | **1.51x** |

Shannon's source coding theorem: no lossless code can average fewer bits than
the entropy. So:

```
realistic lossless ceiling  = 16 / 10.57 = 1.51x
ABSOLUTE ceiling (exponent compressed to literally zero bits)
                            = 16 /  8.00 = 2.00x
```

**10-20x lossless compression of bf16 weights is mathematically impossible.**
Sign+mantissa alone are 8 of every 16 bits and carry ~7.97 bits of real
entropy — there is nothing to remove. This is not an engineering gap that
more work closes; it is the Shannon bound.

### Corollary: better entropy coding buys nothing

Measured Huffman average code length: **2.628 bits** vs the 2.595-bit floor —
**1% above optimal**. Switching to arithmetic or ANS coding (which achieve
fractional bit lengths) would recover 0.033 bits/weight, i.e. 0.3% of total
size. Not worth building. This disproved my own initial hypothesis that
Huffman's integer-length constraint was the main loss.

---

## 2. What the engine actually achieves

| Stage | Ratio | Status |
|---|---|---|
| Theoretical floor | 1.51x | Shannon |
| Uniform-stride chunk padding (first build) | 1.29x | superseded |
| **Real prefix-sum chunk offsets** | **1.45x** | **measured, current** |
| + shared LUT across layers | 1.50x | one change away |

The jump from 1.29x to 1.45x came from one diagnosed fix. The first build
padded every chunk to the layer's longest chunk so the GPU could locate a
chunk with a multiply instead of a lookup. Measured cost of that convenience
on real layers: **35-88% of the entire exponent stream** — a 2.36M-weight
layer stored 5.375 bits/weight against a 2.822-bit floor, because one unlucky
chunk full of rare symbols set the stride for all 2304 of them. Replacing it
with real prefix-sum offsets cost one indexed load per program and recovered
nearly all of it (exponent stream: 22.1% -> 16.5% of original).

**Accuracy: bit-exact.** Verified three independent ways on the real model —
every weight in all 197 layers identical, forward-pass logits identical
(max diff exactly 0.0), and generated text token-for-token identical.
32/32 GPU tests pass.

---

## 3. Head-to-head vs AirLLM

AirLLM streams the **full uncompressed** model from disk **once per token**.
That is its cost model, and it is why it is slow.

Measured machine constants: NVMe 2.0 GB/s, decode 16.87 GB/s, VRAM 6.5 GB.

| Model (bf16) | Size | AirLLM | **Ours today** | Ours + speculation |
|---|---|---|---|---|
| 1.5B | 3 GB | 0.67 tok/s | **1.00 tok/s** | 15.0 tok/s |
| 8B | 16 GB | 0.13 tok/s | **0.19 tok/s** | 2.8 tok/s |
| 27B | 54 GB | 0.04 tok/s | **0.06 tok/s** | 0.8 tok/s |
| 70B | 140 GB | 0.014 tok/s | **0.021 tok/s** | 0.31 tok/s |
| | | 1.00x | **1.50x** | **22.4x** PROJECTED |

**Where the 10-20x you want actually lives: the right-hand column.** Not in
size — in *bus crossings per token*. Compression contributes 1.5x. Speculative
decoding contributes ~15x by amortizing one weight sweep across ~15 accepted
tokens. They multiply.

**Honest status: the 1.50x column is built and measured. The 22.4x column is
projected — speculative decoding is not implemented in this repo.** The
`k~15` figure is from published, reproduced results (SpecExec ~20
tokens/iteration; SubSpec 9.1x at 8 GB), not from anything measured here.

---

## 4. Novelty — honest assessment: LOW-MEDIUM

| Component | Novel? | Prior art |
|---|---|---|
| Entropy-coding LLM weight exponents | **No** | DFloat11 (2025), ZipServ (ASPLOS'26), Huff-LLM |
| Speculative decoding over offloaded weights | **No** | SpecExec (2024), SubSpec (NeurIPS'25), SpecOffload |
| Layer-streaming inference | **No** | AirLLM, FlexGen |
| **Combining compression + speculation + tiered residency** | **Yes** (search found no instance) | — |
| Warp-vectorized Triton Huffman decoder at 16.87 GB/s | Incremental | nvCOMP, DFloat11's CUDA kernel |

**This is a composition, not a new algorithm.** Every mechanism is published.
The defensible claim is narrow:

> Entropy coding is *structurally better suited* to the streaming/offload path
> than to resident inference. Variable-length codes make random access hard —
> which is why DFloat11 needs elaborate CUDA machinery to use them in VRAM —
> but a streaming path consumes weights strictly sequentially in a known
> order, where variable-length codes cost nothing. The property that makes
> entropy coding awkward in VRAM makes it natural on the bus.

Secondary: compressibility varies per layer (measured 65.9%-67.6%), so it
should drive residency planning — keep the *least* compressible layers
resident. No existing system does this because none compresses on the
streaming path.

That is a real observation. It is not a new algorithm, and it should not be
presented as one.

---

## 5. Applicability — where this helps and where it does not

**Helps:**
- bf16/fp16 models too large for VRAM, where quality loss is unacceptable
- Large models on consumer GPUs (the 27B/70B rows above)
- Any tier where bandwidth, not compute, is the bottleneck

**Does not help:**
- **Already-quantized models.** Q4 codes are near their own entropy; almost
  no headroom. This technique is for full-precision weights only.
- Models that already fit in VRAM — adds decode overhead for nothing.
- Compute-bound workloads (large batch, prefill-heavy).

**The competitive caveat that matters most:**

| Approach | Compression | Measured error |
|---|---|---|
| This engine (lossless) | 1.45-1.50x | **0.00%** |
| Q8 grouped quantization | **2.0x** | **0.55%** |
| Q4_K_M grouped | 3.8x | 9.97% |

**Q8 gives more compression than lossless does, at 0.55% relative logit
error.** If that is acceptable — and for most deployments it is, Q8 being
widely treated as lossless-in-practice — then Q8 is simpler, faster, and
smaller than this entire engine. The lossless engine is only the right choice
when "bit-exact" is a hard requirement rather than a preference.

That is the honest boundary of this work's usefulness, and it should be stated
before anyone invests further.

---

## 6. Recommended next step

**Build speculative decoding.** It is the entire remaining gap between 1.5x
and 22x, it is the difference between beating AirLLM by half and beating it by
an order of magnitude, and it is lossless by construction (the
accept/reject/resample step provably samples the target's exact distribution).
`runtime/verify.py` already implements the correctness-critical chain case and
is tested; extending it to a tree is the work.

Everything else on the compression side is within 4% of its Shannon bound and
not worth further optimization.
