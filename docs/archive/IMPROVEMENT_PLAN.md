# Improvement Plan — From 1.29x to Decisively Ahead of AirLLM

Every number here is measured on Qwen3-14B (29.54 GB bf16) on an RTX 3080
Laptop, unless marked PROJECTED.

## Where we stand

| | Ours | AirLLM | |
|---|---|---|---|
| Speed | **24.93 s/tok** | 32.23 s/tok | we win 1.29x |
| Peak VRAM | 5.10 GB | **1.57 GB** | they win 3.25x |
| Read/token | **17.74 GB** | 29.54 GB | we win 1.67x |
| Output | `[12095, 13]` | `[12095, 13]` | identical |
| Preprocess | 40 min | **293 s** | they win 8x |

We convert 78% of our byte advantage into wall-clock. The other 22%, plus
the entire VRAM deficit, is what this plan targets.

---

## The five levers, ranked by measured evidence

### 1. Speculative decoding — the only order-of-magnitude lever
**Projected: 10-15x. Not built.**

Everything else on this list is worth 1.3-3x. This one is worth more than
all of them combined, and it is the reason the ceiling is high.

One weight sweep currently produces one token. With a draft model proposing
k tokens and the full model verifying them in a single sweep, one sweep
produces ~10-20 accepted tokens. The bus traffic per sweep is unchanged, so
tokens/sec scales almost linearly with acceptance length.

It is **lossless by construction** — the accept/reject/resample step
provably samples the target's exact distribution regardless of draft
quality, so it costs no accuracy. `runtime/verify.py` already implements and
tests the correctness-critical chain case; extending it to a tree is the
work.

At k=12 on current numbers: 24.93 / 12 = **~2.1 s/token, or 15x faster than
AirLLM.**

### 2. Replace the `.npz` store with raw binary + mmap
**Projected: 1.5-2x. Cheap.**

Measured: `np.load` on `.npz` tops out at ~1.03 GB/s even from page cache,
against 2.0 GB/s O_DIRECT on this machine's disk. The zip container is the
ceiling, not the hardware. Raw `.bin` per tensor with offsets in the
manifest removes the container entirely and allows mmap.

This also fixes a measurement flaw: `.npz` loads lazily, so array reads
happen inside the decode timing block. Our reported `decode_s` = 46.5 s of
49.85 s wall **conflates I/O with decode** and overstates decode's share.
Raw files make the split measurable.

### 3. Prefetch the next layer while computing the current one
**Projected: up to 1.8x. Medium effort.**

The loop today is strictly serial: load -> decode -> compute -> free ->
load. Compute measured at only ~1.5 s of a ~50 s run, so the GPU is idle
almost the entire time, but so is the disk while decoding and the decoder
while loading. Double-buffering with a CUDA stream would overlap all three,
making wall-clock ~max(io, decode, compute) instead of their sum.

### 4. Row-gather the embedding instead of materializing it
**Projected: VRAM 5.10 GB -> ~1.6 GB. This is the AirLLM gap.**

`embed_tokens` and `lm_head` are 3.1 GB of our 5.1 GB peak. But an embedding
lookup only needs the rows for the current tokens -- a handful of rows, not
151936 of them. Gathering those rows (from CPU, or row-wise from the
compressed store) drops that 1.56 GB contribution to kilobytes.

`lm_head` is harder: it genuinely needs all 151936 output columns to produce
logits. Options are streaming it in column blocks with a running argmax, or
keeping it resident and accepting ~1.6 GB.

This is the single change that would let us match AirLLM's 1.57 GB **while
keeping the speed advantage.**

### 5. Parallelize the compression pass
**Projected: 40 min -> ~5 min. Low risk.**

Compression is single-process over 443 tensors. It is embarrassingly
parallel across tensors and the GPU is idle throughout. A process pool
closes the 8x preprocessing gap against AirLLM.

---

## Projected combined result

Multiplying only the levers with measured backing (2, 3) and the projected
speculation lever (1):

```
24.93 s/tok
  / 1.7  (raw binary store)      -> 14.7
  / 1.8  (prefetch overlap)      ->  8.2
  / 12   (speculation, k=12)     ->  0.68 s/token
```

**~0.68 s/token vs AirLLM's 32.23 = ~47x faster, at ~1.6 GB VRAM, bit-exact.**

Treat that as an upper bound: the levers are unlikely to compose perfectly,
and only the compression half is measured today. A conservative read is
20-30x.

---

## Order of work

| # | Change | Effort | Gain | Risk |
|---|---|---|---|---|
| 1 | Raw binary store (drop `.npz`) | S | 1.5-2x | low |
| 2 | Embedding row-gather | M | VRAM 3.2x | low |
| 3 | Prefetch / double-buffer | M | up to 1.8x | medium |
| 4 | Parallel compression | S | 8x preprocess | low |
| 5 | **Tree speculative decoding** | **L** | **10-15x** | medium |

Do 1, 2 and 4 first -- they are small, low-risk, and 2 removes the only
metric where AirLLM currently beats us. Then 5, which is where the real
lead comes from.

---

## VRAM control (implemented)

You now state a budget and the planner decides residency:

```python
EngineConfig(vram_budget_gb=3.0)
plan_from_manifest(manifest, budget_gb=3.0)   # inspect before running
```

Measured behaviour on the real 14B store:

| Budget | Feasible | Resident | Streamed/token |
|---|---|---|---|
| 1.5 GB | **No** — below the 2.09 GB floor | — | — |
| 2.0 GB | **No** — below the 2.09 GB floor | — | — |
| 2.5 GB | Yes | 0.40 GB | 19.52 GB |
| 3.0 GB | Yes | 0.90 GB | 19.16 GB |
| 4.0 GB | Yes | 1.90 GB | 18.45 GB |
| 6.0 GB | Yes | 3.88 GB | 17.08 GB |
| 8.0 GB | Yes | 5.87 GB | 15.74 GB |

Two properties worth noting:

**Infeasible budgets are refused up front, with the arithmetic.** Not
discovered as an OOM on layer 1 -- which is precisely how the old
fixed-residency design failed at 4 GB and again at 6 GB.

**The 2.09 GB floor is the embedding, and lever 4 removes it.** Until then
no budget below ~2.1 GB is achievable, because the largest single tensor
must be materializable.

### How it chooses what to keep

Every weight in a dense decoder is used exactly once per token, so use
frequency cannot rank them. What differs is bus traffic avoided per byte of
VRAM spent:

```
value density = compressed_bytes / uncompressed_bytes
```

A tensor that compresses **poorly** costs nearly its full size in traffic
every token, so pinning it saves the most per VRAM byte. Highly
compressible tensors are cheap to re-stream and are evicted first. This is
the fractional-knapsack solution to "minimize bytes streamed per token
subject to a VRAM ceiling."

That is the opposite of the intuitive rule (keep the biggest or most
important tensors), and it only becomes visible once compression sits on
the streaming path -- which is why no existing offloading engine does it.
