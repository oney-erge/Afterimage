# The Lossless Engine — Fundamental Redesign

**Constraint change:** never trade accuracy. Output must be bit-exact with
the reference model. Trade speed instead.

This kills every idea in the previous design (subspace caching, quantization)
and replaces the problem with a cleaner one that has a real answer.

---

## 1. The problem, reduced to one number

In the offloaded regime the GPU is idle waiting for weights. Measured on this
rig: ~2 GB/s from NVMe, ~20 GB/s from host RAM over PCIe, against ~20 TFLOPs
of compute. That is roughly **1000× more spare compute than spare bandwidth.**

So wall-clock time is governed by exactly one quantity:

```
                    bytes that must cross the bus
time per token  =  ───────────────────────────────
                    bandwidth  ×  tokens per crossing
```

Everything else — FLOPs, kernel efficiency, cache blocking — is noise at this
ratio. **Any transform that spends compute to move fewer bytes is a good
trade, and we have three orders of magnitude of compute to spend.**

### The I/O lower bound, and the one legal way around it

This is the Aggarwal–Vitter external-memory model: with fast memory `M` and a
working set `S > M`, you pay I/O for everything that doesn't fit. Autoregressive
decoding makes it worse than a normal streaming problem, because token `t+1`
depends on token `t`'s output. That dependency forbids batching, and without
batching **you must re-read all `S` bytes for every single token.** That is
AirLLM's cost model, and no amount of engineering escapes it.

**Speculative decoding is the only lossless way out.** A draft model proposes
`k` tokens; the target verifies all `k` in one sweep. The dependency is broken
by *guessing*, and correctness is restored by the accept/reject/resample step,
which provably samples from the target's exact distribution regardless of how
bad the draft is. It converts a sequential dependency into a parallel
verification — the exact structure the I/O bound punishes, removed.

So there are precisely **two** lossless levers, and they multiply:

| Lever | Floor / ceiling | Set by |
|---|---|---|
| **Fewer bytes per sweep** | Shannon entropy of the weights | information theory |
| **More tokens per sweep** | ~15–20 accepted tokens | entropy of natural text |

---

## 2. Lever 1, measured on a real model: 1.51×

Audited all 197 linear layers of Qwen2.5-1.5B
([entropy audit](../scripts/run_entropy_audit.py)):

| | bf16 | fp16 |
|---|---|---|
| Raw | 3.087 GB | 3.087 GB |
| **Entropy floor** | **2.040 GB** | 2.042 GB |
| Size fraction | **66.1%** | 66.2% |
| **Compression** | **1.51×** | 1.51× |

**Mean exponent entropy: 2.644 bits, in a field 8 bits wide.** That is where
all the gain is. Mantissa entropy measured 6.97 bits out of 7 — essentially
uniform, i.e. incompressible, exactly as theory predicts for trained weights.

This independently reproduces DFloat11's reported ~2.6-bit exponent entropy
and ~70% size on a different model, so the mechanism is real and general, not
a quirk of one checkpoint.

**It is bit-exact.** Nothing is discarded; the exponent field is simply spelled
with a shorter code. Decompress and you get the identical bits back.

### A finding that corrected my own reasoning

I predicted bf16 would compress much better than fp16 (bf16 wastes ~5.4 bits
on its exponent, fp16 only ~2.4). **The measurement said otherwise: 66.1% vs
66.2%, identical.**

Modern checkpoints are natively bf16. Serving one as fp16 widens the mantissa
7→10 bits and zero-pads the low 3, adding no information — entropy coding
recovers exactly that padding, and fp16's extra mantissa waste cancels its
smaller exponent waste. The original prediction only holds from a true fp32
source (measured there: bf16 65.7% vs fp16 84.4%).

Practical consequence: **for a native-bf16 checkpoint, serving dtype does not
change compression. Choose it for numerics, not for size.**

---

## 3. Lever 2: ~15–20 tokens per sweep

From the published, reproduced literature (SpecExec ~20 tokens/iteration;
SubSpec 9.1× at 8 GB VRAM). The ceiling is not engineering — it is the entropy
of natural language. Covering the true continuation `k` steps ahead needs a
draft tree growing roughly exponentially in `k`, so accepted length saturates.

---

## 4. The combined lossless ceiling

```
bytes per token  =  0.66 × S / 17   ≈  0.039 × S      vs AirLLM's  1.00 × S
```

**≈ 25× faster than AirLLM, with bit-identical output.** No accuracy traded,
anywhere.

---

## 5. Why this specific combination is novel

Both halves exist. **Nobody has put them together**, and there is a structural
reason the combination is better than either alone:

| System | Lossless? | Compression | Speculation | Offload target |
|---|---|---|---|---|
| AirLLM | ✅ | ✗ | ✗ | disk |
| DFloat11 (2025) | ✅ | ✅ 1.4× | ✗ | fits-in-VRAM |
| SpecExec / SubSpec | ✅ | ✗ | ✅ | RAM |
| llama.cpp Q4 | ❌ **9.97% error** | ✅ 3.8× | partial | RAM |
| **This design** | ✅ | ✅ | ✅ | RAM + NVMe |

### The structural insight: entropy coding *wants* to be on the streaming path

Entropy coding produces **variable-length** codes. That is normally a liability —
you cannot jump to weight *i* without decoding everything before it. DFloat11
spends significant complexity on custom CUDA kernels with lookup tables
precisely to make random access work inside VRAM.

**An offloading path has no such problem.** Weights are consumed strictly
sequentially, layer by layer, in a fixed order known in advance. That is the
single best case for entropy coding: decode as bytes arrive, no random access
ever required, and decompression of layer `L` overlaps with the transfer of
layer `L+1`.

So the thing that makes entropy coding awkward for resident inference makes it
*natural* for streamed inference — and the ~1000× spare compute means the
decode cost is invisible. This is the argument that the combination is not
merely additive.

### Second insight: compressibility should drive residency

Layers do not compress equally (measured: 65.9% to 67.6%). The residency
planner should therefore keep the **least compressible** layers in VRAM — they
cost the most bytes to stream — and stream the most compressible ones. That is
a knapsack on *compressed* size, not raw size, and no existing system does it
because no existing system compresses on the streaming path.

---

## 6. What it buys on this specific machine

Measured constants: 6.5 GB usable VRAM (desktop holds ~1.5 GB of 8), 39 GB
host RAM, ~20 GB/s PCIe, ~2 GB/s NVMe, and a measured capacity cliff at
6.2 GB resident (fits) vs 6.9 GB (spills, 5.4× slower) from
[CAPACITY_RESULTS.md](CAPACITY_RESULTS.md).

| Model (bf16, lossless) | Raw | Compressed | Where it lives | Projected |
|---|---|---|---|---|
| 3B | 6.0 GB | 4.0 GB | **fully in VRAM** | full GPU speed |
| 8B | 16 GB | 10.6 GB | 6 GB VRAM + 4.6 GB RAM | ~60 tok/s |
| 27B | 54 GB | 35.6 GB | 6 GB VRAM + 30 GB RAM | ~10 tok/s |
| 70B | 140 GB | 92 GB | VRAM + RAM + NVMe | ~1–2 tok/s |

**The step-function win:** compression moves the "fits entirely in VRAM" line
from ~3B to ~5B in bf16. Crossing that line is worth 5.4× on this hardware
(measured, not projected) — a far bigger jump than any incremental streaming
improvement, because it eliminates streaming altogether.

⚠️ **Every number in the last column is a projection from measured bandwidth
and published acceptance rates, not a measurement.** They are what the design
predicts, and they are exactly what Phase A below is meant to falsify.

---

## 7. Build plan

Ordered so the riskiest assumption is tested first and cheaply.

| Phase | What | Validates |
|---|---|---|
| **A** | Entropy-code one layer; decompress on GPU; verify bit-exact round-trip and measure decode throughput | **Is decompression fast enough to stay off the critical path?** — **DONE, see §7.1** |
| **B** | Streaming pipeline: compressed layer store + async prefetch + overlap decode with transfer | End-to-end lossless streaming |
| **C** | Compressibility-aware residency planner (knapsack on compressed size) | Lever 1 fully realized |
| **D** | Tree speculation on top (existing `runtime/verify.py` extends from chain to tree) | Lever 2 |
| **E** | Token-identity gate: 100.000% vs reference, else fail the build | **The whole premise.** Non-negotiable. |

### 7.1 Phase A — measured, 2026-08-17

Built a canonical Huffman codec for the bf16 exponent field (CPU encode,
package-merge length-limiting for a bounded GPU lookup table) and a chunked
parallel decoder in Triton, on the actual RTX 3080 Laptop.

**Correctness: bit-exact, verified two ways before trusting the GPU.**
1. Two independently-written CPU decoders (bit-by-bit tree walk vs. the
   LUT algorithm the GPU implements) cross-checked against each other across
   15 random distributions plus real bf16 exponent data — before any GPU
   code was written.
2. The GPU kernel then checked against that CPU reference: **32/32 tests
   passing** on real hardware (17 for the first kernel design, 15 for the
   optimized one below), across random alphabets, real bf16 weight
   distributions, degenerate single/two-symbol cases, and misaligned chunk
   boundaries.

This process caught two real bugs before either reached the GPU: a stored
LUT bit-width that didn't match the table's actual size (`table.max_bits`
vs. the length-limiting ceiling passed in — an out-of-bounds LUT index,
caught by a CPU test), and a units error in the first throughput
measurement itself (counting int32 output bytes instead of the real 2
bytes/weight bf16 reconstruction — caught by hand-checking the number
against what it was supposed to mean, since nothing had asserted the
metric's definition).

**Throughput, against the ~20 GB/s target (measured host-RAM bandwidth):**

| Kernel | Design | Best measured | vs. 20 GB/s target |
|---|---|---|---|
| v1 | One Triton program per chunk, scalar state | 2.04 GB/s | 10× short |
| v2 | BLOCK_CHUNKS decoded together, vectorized | **16.87 GB/s** | **1.19× short — close** |

v1's shortfall diagnosed correctly on the first attempt: it used one program
per chunk with shape-`[1]` ("scalar") state throughout, running the GPU's
32-wide SIMD execution at roughly 1/32 of its width. Restructuring so a
program decodes `BLOCK_CHUNKS` independent chunks together, advancing in
lockstep, gave an **8.27× speedup with zero change to the algorithm's
correctness** (32/32 tests still pass). The best `BLOCK_CHUNKS` value
measured, 32, is exactly the GPU's native warp size — independent
confirmation that SIMD underutilization was the real cause, not a guess that
happened to work.

**Verdict: clears the NVMe threshold decisively (8.4×), approaches but does
not yet clear the RAM threshold (84% of it).** This is not the clean pass
the design hoped for, and not a failure either — it is close enough that the
next optimization pass (see below) is a reasonable investment rather than a
sign the approach is wrong. **Phase B should proceed for the NVMe tier now**
(already viable) while Phase A gets one more iteration for the RAM tier.

**What's likely left on the table**, in order of expected payoff:
- `num_warps` was left at Triton's default; not yet tuned.
- The 2× unconditional refill-check per symbol (bounded-but-fixed at 2 steps
  for safety, per the module's design note) likely wastes cycles on symbols
  that need only 0 or 1 refills — worth measuring the real distribution and
  specializing.
- No overlap with H2D transfer yet; Phase B's async pipeline should hide
  a meaningful fraction of whatever decode time remains, which changes what
  "clearing 20 GB/s in isolation" actually requires.

Reproduce: `scripts/throughput_run.sh` (v1), `scripts/throughput_v2_run.sh`
(v2), `pytest tests/test_gpu_decode.py tests/test_gpu_decode_v2.py -v` (needs
CUDA + triton; skips cleanly otherwise).

---

## 8. Honest position

- **Not invented here:** entropy coding of weights (DFloat11, ZipServ, Huff-LLM,
  Unweight), speculative decoding (SpecExec, SubSpec), offloading (AirLLM,
  FlexGen). All published.
- **The contribution is the composition** plus two arguments that make it more
  than addition: entropy coding is *structurally suited* to the streaming path
  in a way it is not to resident inference, and compressibility becomes a
  residency-planning variable.
- **The claim is bounded:** ~25× over AirLLM, bit-exact, on consumer hardware.
  Not "beats quantization on speed" — quantized Q4 will remain faster. The
  claim is **quality-per-VRAM at zero accuracy loss**, which is a different and
  currently unserved axis.
- **1.51× is measured. ~17 tokens/sweep is borrowed from literature. The
  product of the two is a projection until Phase E runs.**
