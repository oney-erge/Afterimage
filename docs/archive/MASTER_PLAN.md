# Afterimage Master Plan

**Goal: run a 27B-class model on an 8 GB consumer GPU, bit-exact, at 4–6 s/token —
and be measurably the fastest lossless option on hardware that cannot hold the model.**

Every "measured" number here comes from this repo on an RTX 3080 Laptop (8 GB VRAM,
19 GB RAM, WSL2, NVMe). Every "PROJECTED" number is labelled and its derivation shown.
Literature claims are cited to their source and never mixed with our own measurements.

---

## 0. Where we actually stand

Measured 2026-08-18, Qwen3-14B (29.54 GB bf16), cold page cache, greedy decoding.
Two rows: before this plan's Phase 1-3 work (foundation fixes, three-tier residency,
deeper I/O prefetch, KV cache), and after.

| Metric | Before | After | AirLLM | Verdict |
|---|---|---|---|---|
| Speed | 17.38 s/tok | **14.93 s/tok** | 28.5–29.8 s/tok* | **~2x faster** |
| Bytes read / token | 17.74 GB | 17.74 GB | 29.54 GB | 1.67x less |
| Peak VRAM (live) | 3.55 GB | **2.66 GB** | ~1.57 GB** | gap narrowed |
| Output tokens | bit-exact vs reference | bit-exact vs reference | bit-exact vs reference* | see note |
| Compression (one-time) | 289 s (parallel) | 370 s*** | ~293 s | parity |
| Compression ratio | 1.453x lossless | 1.453x lossless | none | — |

\* **AirLLM's own greedy output was not stable run-to-run** in our testing: two
back-to-back runs on the identical prompt produced `" Paris."` then `" Paris,"` --
a different second token, with a new "attention mask cannot be inferred" warning
appearing in the log on the divergent run. Afterimage's output was bit-identical to
the reference model on every single run across this project's history (dozens of
runs, several models). This is reported plainly, not used to inflate the speedup
claim -- the speed comparison above uses AirLLM's own measured wall-clock either way.

\*\* AirLLM VRAM is from an earlier separate run; the head-to-head harness does not
capture it in the same invocation as AirLLM's timing.

\*\*\* Ran on a machine with a concurrently-active GPU test suite; the 289 s figure
from an isolated run is more representative of steady-state compression time.

Speculative mode (separate, sampled not greedy): 5.4 s/token measured, 2.5
tokens/sweep, 18.8% draft acceptance with Qwen3-0.6B at temperature 1.0 -- see
§5 Lever D for why this is below the literature's typical 40%+ and what would
close the gap.

**The honest read:** we beat the direct competitor, we are strictly lossless, and the
codec is at 97% of its theoretical ceiling. The remaining wins are *not* in the codec.

---

## 1. Competitive landscape

The literature splits cleanly into two camps, and **nobody occupies the intersection
Afterimage sits in.**

| System | Lossless | Compressed | Streams from disk | Single consumer GPU | Niche |
|---|---|---|---|---|---|
| [DFloat11](https://arxiv.org/pdf/2504.11651) (NeurIPS'25) | yes | 1.43x | no | multi-GPU | model fits in GPU(s) |
| [ZipServ](https://www.cse.ust.hk/~weiwa/papers/zipserv-asplos26.pdf) (ASPLOS'26) | yes | ~1.43x | no | no | serving, GPU-resident |
| [NeuZip](https://arxiv.org/html/2504.11651v3) | yes | 1.5x | no | yes | runtime memory |
| ZipNN | yes | 1.2–1.5x | n/a | n/a | checkpoint storage |
| [FlexGen](https://arxiv.org/pdf/2303.06865) | **no** (4-bit) | yes | yes | yes | batch throughput |
| DeepSpeed ZeRO-Inference | yes | no | yes | yes | layer-wise offload |
| llama.cpp / GGUF | **no** | yes | mmap | yes | quantized local |
| AirLLM | yes | **no** | yes | yes | "it runs at all" |
| **Afterimage** | **yes** | **1.45x** | **yes** | **yes** | **all four** |

**DFloat11 is our closest peer and independently validates our approach**: same insight
(bf16 exponents are low-entropy, sign/mantissa are not), same technique (Huffman on
exponents, GPU LUT decode, block-level decompression), same ratio (~30% reduction).
They target models that fit across GPUs; we target models that fit in neither VRAM
nor RAM. We must benchmark against them directly, not just AirLLM.

**ZipServ contributes the idea worth stealing** (Lever C): a fused decompression-GEMM
kernel that decodes weights *directly into Tensor Core register files*, never
materializing decompressed weights in global memory. They apply it to GPU-resident
serving. **Nobody has applied it to a disk-streamed model** — that is a genuinely
novel combination and it attacks our VRAM deficit at the root.

**SpecOffload contributes the second idea** (Lever D): during offloaded inference the
GPU is almost entirely idle waiting on I/O — they measured 240 ms to load an FFN layer
against 0.1 ms to compute it, a 2400x disparity. **We measure the same phenomenon:
~375 ms read vs ~19 ms compute per layer, ~20x.** They fill that idle window by running
the draft model there. That idle GPU is free real estate and we are wasting it.

---

## 2. The fundamental math — why the next 10x is not in the codec

### 2.1 The codec is finished as a source of large wins

bf16 = sign(1) + exponent(8) + mantissa(7). Measured field entropies on real weights:
exponent ≈ 2.60 bits of 8 allocated; mantissa ≈ 6.97 of 7; sign ≈ 1.0 of 1.
Floor ≈ 10.6 bits/weight ⇒ **hard ceiling ≈ 1.51x**. We achieve **1.453x = 96% of it.**

The 2026 "[Approaching Shannon Bound](https://arxiv.org/pdf/2606.15789)" work squeezes
the remainder with context modeling + rANS instead of Huffman. That is worth doing
(Lever F) but it buys **~1.45 → ~1.55x, not 10x.** Anyone promising more from a
lossless codec on bf16 is wrong, and we should say so publicly — it is a defensible
differentiator that our claims are bounded by an information-theoretic argument.

### 2.2 The real cost model

```
t_token  ≈  bytes_moved_per_token / effective_bandwidth        (I/O bound; compute is ~5%)
bytes_moved_per_token = compressed_model_bytes                  (unless cached or amortized)
```

There are exactly **three** ways to reduce this, and the codec only touches the first:

| # | Lever | Mechanism | Ceiling |
|---|---|---|---|
| 1 | Move fewer bytes | compression (**done, 1.45x**) + **RAM-tier caching (not built)** | RAM size |
| 2 | Move bytes faster | I/O parallelism, O_DIRECT, fused decode | disk seq. read |
| 3 | Get more tokens per sweep | speculation, batching | acceptance rate |

**Levers 1 (cache) and 3 (speculation) are where the remaining order of magnitude is,
and neither is meaningfully built today.** 19 GB of RAM sits nearly idle on this machine
while we re-read the same bytes off disk every single token.

### 2.3 The 27B-on-8GB target, derived

27B bf16 ≈ 54 GB → compressed at 1.45x ≈ **37.2 GB**. Machine: 6.5 GB usable VRAM,
~14 GB RAM available for cache, NVMe ~2.0 GB/s (measured 2.5 GB/s achievable with
better I/O).

```
RAM-cached share : 14.0 GB  @ ~10 GB/s (PCIe)  =  1.4 s
Disk share       : 23.2 GB  @ ~2.5 GB/s        =  9.3 s
                                          total ≈ 10.7 s/token   (Levers A+B)
with tree speculation @ 3 accepted tok/sweep    ≈  3.6 s/token
                                     conservative:  4–6 s/token   [PROJECTED]
```

AirLLM on the same model has a hard floor of 54 GB / 2.0 GB/s = **27 s/token**, and
realistically 35–45 s/token. **That is a 6–10x lead, on a workload it can barely do
at all.** On a 32 GB RAM machine the RAM tier covers most of the model and the gap widens
further.

---

## 3. Bug & gap register

### P0 — correctness and credibility

| ID | Issue | Why it matters |
|---|---|---|
| P0-1 | **README describes the abandoned Phase-0 subspace-cache idea and says "NO-GO"** | The repo publicly misrepresents itself. The working engine that beats AirLLM 1.64x is undocumented. |
| P0-2 | **No integrity check on `weights.bin`** | A product whose entire claim is bit-exactness has no checksum. Silent disk corruption ⇒ silently wrong weights, no error. |
| P0-3 | **No manifest schema version** | Old `.npz` stores and new binstores are mutually unreadable with no clear error. |
| P0-4 | **No KV cache** (`use_cache=False`) | Every token re-runs the whole prefix. At 500 tokens this is a large and entirely avoidable compute cost. |
| P0-5 | **`vram_planner` is not wired into the engine** | The user asked to dictate VRAM. What exists is a hard cap plus a *hardcoded* residency policy — the planner is advisory only. |
| P0-6 | **`EngineConfig` is unused by `StreamingLosslessModel`** | Docs reference `EngineConfig(vram_budget_gb=3.0)`; that field does not exist and the engine ignores the class entirely. |
| P0-7 | No `close()` discipline / no context manager | File handles leak; only tests call `close()`. |
| P0-8 | `vram_planner` cost model stale w.r.t. row-gather | It still charges 1.56 GB of traffic to evict an embedding that row-gather now makes ~free. |

### P1 — performance left on the table

| ID | Issue | Est. cost |
|---|---|---|
| P1-1 | **No RAM tier** — every token re-reads everything from disk | **~2x** |
| P1-2 | Prefetch depth = 1, single thread; NVMe wants QD 8–32 | 1.4–1.8x |
| P1-3 | Full decompressed tensor materialized in VRAM | VRAM floor + a wasted round trip |
| P1-4 | Speculation is chain-only, 18.8% acceptance, weak draft | 2–3x |
| P1-5 | Draft model re-runs without its own KV cache → O(k²) | draft cost |
| P1-6 | Draft model competes for the same VRAM budget | VRAM |
| P1-7 | `compute_seconds` wraps the whole forward incl. hooks, so it ≈ wall and double-counts I/O and decode | metrics are misleading |

### P2 — portability and product

| ID | Issue |
|---|---|
| P2-1 | CUDA-only assumptions (`.cuda()`, `torch.cuda.synchronize`, `set_per_process_memory_fraction`) — no ROCm path |
| P2-2 | No CPU fallback for machines without a usable GPU |
| P2-3 | ~1,185 LOC of dead code from the abandoned approach (`basis`, `gate`, `sketch`, `engine`, `draft`, `resident`, `streamer`, `tiers`, `layout`) |
| P2-4 | No CLI, no server, no Docker, no installer, no UI |
| P2-5 | `configs/hardware.yaml` describes a rig that was never used and calls this a CPU-only box — stale |
| P2-6 | No end-to-end distributional test for `generate_speculative` |

---

## 4. Engine roadmap

Ordered by expected gain per unit of risk. Each lever states its mechanism, its
evidence, and how it will be proven.

### Lever A — Three-tier residency: VRAM / RAM / disk
**Expected ~2x. This is the single biggest missing win.**

Today every byte comes from disk on every token. RAM is 5–10x faster than NVMe and
19 GB of it is idle. Extend the existing `value_density` knapsack from a two-way
(resident / streamed) decision to a three-way tier assignment:

```
tier(t) = VRAM  if it fits the VRAM budget and has the highest traffic-per-byte
          RAM   if it fits the RAM budget            (pinned, page-locked)
          DISK  otherwise
```

The planner already ranks by `comp_bytes / orig_bytes` — the correct metric for
"traffic avoided per byte of residency spent". It generalizes to tiers by charging each
tier its own bandwidth. Pinned (page-locked) RAM buffers also enable true async DMA to
GPU, which regular pageable memory cannot do.

**Proof:** bytes-from-disk per token must fall by the cached fraction, with output
still bit-identical. New `tier` field in `StreamStats`.

### Lever B — Deep I/O parallelism
**Expected 1.4–1.8x.**

We currently achieve ~1.1 GB/s effective against ~2.0 GB/s O_DIRECT measured on this
disk, using **one** prefetch thread at queue depth 1. NVMe requires QD 8–32 to reach
peak. Literature is unambiguous that
[io_uring achieves the lowest latency and competitive IOPS for NVMe](https://arxiv.org/pdf/2512.04859),
and that GPUDirect Storage removes the CPU bounce buffer entirely.

Work: a reader pool with configurable queue depth using positional reads (`os.pread`,
thread-safe, no shared seek cursor); optional `io_uring` backend on Linux; prefetch
horizon of N layers rather than 1; O_DIRECT to bypass double-buffering. GPUDirect
Storage is a stretch goal (needs a supported driver stack, unlikely under WSL2).

**Proof:** a microbenchmark reporting GB/s vs queue depth, plus end-to-end `io_seconds`.

### Lever C — Fused decode → GEMM (adapted from ZipServ)
**Expected: VRAM floor to ~1–2 GB, plus removal of a full VRAM write+read round trip.**

Today: decode the whole tensor into VRAM, then matmul against it. ZipServ's ZipGEMM
instead decodes on the fly directly into the registers feeding the matmul, eliminating
the intermediate buffer. **They do this for GPU-resident serving; applying it on a
disk-streaming path is new.** Combined with Lever A it means a streamed layer never
needs a full materialized copy in VRAM at all — which is precisely the metric where
AirLLM currently beats us.

This is the highest-effort, highest-risk lever (a fused Triton kernel), and it should
be gated behind a config flag with the current path as fallback.

**Proof:** bit-exact output vs the unfused path, plus peak-VRAM measurement.

### Lever D — Speculation overhaul
**Expected 2–3x on top of everything else.**

Three compounding changes:
1. **Tree verification** instead of a single chain — verify many branches per sweep
   ([SpecInfer](https://arxiv.org/pdf/2305.09781), EAGLE-2/3). Our sweep cost is fixed
   by I/O, so extra branches are nearly free.
2. **SpecOffload-style interleaving** — run the draft model *during* the target's I/O
   stalls, in the ~20x idle GPU window we measured, instead of serially before it.
3. **A better draft.** 18.8% acceptance with Qwen3-0.6B is poor;
   [EAGLE-3 reports ~2.4 accepted tokens/step](https://arxiv.org/pdf/2508.08192) with a
   trained feature-level drafter. Options: an EAGLE-style head, or self-speculation from
   the target's own early layers (no extra VRAM, no second model).

Correctness is already guaranteed by `verify.py`'s accept/reject/resample step, which
samples the target's exact distribution regardless of draft quality — **draft quality
is purely a speed knob, never an accuracy one.** That property must be preserved and
re-tested for the tree case.

### Lever E — KV cache
**Expected: large for long outputs, ~0 for 2-token benchmarks.**

`use_cache=False` today. With layer streaming, the cache must be written as each layer
is resident and kept across the layer's eviction — mechanically straightforward via the
existing hooks, but it interacts with `to_empty()`/`to("meta")` and needs care.

### Lever F — Codec push toward the Shannon bound
**Expected 1.45x → ~1.55x. Modest, but it multiplies every other lever.**

Replace/augment Huffman with **rANS** (fractional bit costs; Huffman loses 2–4% to
integer code lengths) plus **context modeling** — condition each exponent on its
predecessor, since neighbouring weights in a row have correlated magnitude, so
`H(e_i | e_{i-1}) < H(e_i)`. This is exactly the recipe the 2026 Shannon-bound work uses.

Must be benchmarked honestly: a better ratio that decodes slower can be a net loss on an
I/O-bound path. Ship it as a selectable codec, not a forced upgrade.

### Lever G — Method registry (pluggable, config-selectable)
**Not a speedup — the mechanism that makes every lever above a user choice.**

```python
EngineConfig(
    codec        = "huffman" | "rans" | "rans-ctx" | "none",
    residency    = "static" | "knapsack" | "three-tier",
    io           = "sync" | "threadpool" | "io_uring",
    decode       = "materialize" | "fused",
    decoding     = "greedy" | "spec-chain" | "spec-tree",
    vram_budget_gb, ram_budget_gb, io_queue_depth, ...
)
```

Named, documented, benchmarked methods the user picks — with `auto` choosing by detected
hardware. This is also what makes the benchmark table honest: every row is a config.

---

## 5. Product roadmap

### 5.1 One-command install-or-run
`install.sh` / `install.ps1` that: detects OS, GPU vendor (NVIDIA/AMD/none), VRAM, RAM,
and disk speed → installs the right torch/Triton stack (CUDA or ROCm or CPU) → creates a
venv → writes a hardware-matched config → and **if already installed, just launches the
app instead.** Idempotent, re-runnable, no manual dependency work.

### 5.2 Docker
Three images: `cuda`, `rocm`, `cpu`. `docker-compose.yml` with a model-cache volume and
a compressed-store volume so the expensive one-time compression survives container
rebuilds. GPU passthrough documented for both vendors.

### 5.3 FastAPI server
- **OpenAI-compatible**: `POST /v1/chat/completions`, `/v1/completions`, `/v1/models`,
  with SSE streaming — so existing clients work unchanged.
- **Native control**: `/api/compress` (with progress), `/api/models`, `/api/config`,
  `/api/hardware`, `/api/stats`.
- **Job control the user asked for**: WebSocket `/ws/progress` streaming layer/token
  progress, plus `/api/jobs/{id}/pause` and `/resume` and `/cancel`. Pause is natural
  here — the engine is a layer loop, so it can stop cleanly at a layer boundary.

### 5.4 Web UI
A single-page config + monitor: hardware detection readout, model picker, VRAM/RAM
budget sliders with the planner's feasibility answer shown live, method dropdowns
(Lever G), compression progress bar, live token stream, pause/resume. Deliberately
small — this is a control panel, not a chat product.

### 5.5 CLI
```
afterimage doctor          # hardware + install diagnosis
afterimage compress MODEL  # build a store, with progress
afterimage run MODEL       # one-off generation
afterimage serve           # FastAPI + UI
afterimage bench           # the comparison table
```

### 5.6 Cross-vendor and small-VRAM support
- **AMD/ROCm**: Triton runs on ROCm, but
  [the out-of-box experience lags and version matching is fragile](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-triton.html).
  Plan: abstract a device layer (`torch.accelerator`), replace CUDA-specific calls,
  autotune `block_chunks` per vendor (warp 32 vs wavefront 64 — **our kernel's tuned
  constant is literally NVIDIA's warp width and is likely wrong on AMD**), and gate on
  the existing bit-exactness tests running green on ROCm.
- **4 GB tier**: requires Lever C (fused decode) to get the working set under ~1.5 GB.
  Until then, 4 GB is honestly out of reach for 14B+ and we should say so rather than
  claim it.
- **CPU fallback**: slow but functional, so `doctor` never dead-ends a user.

### 5.7 Documentation
Rewrite `README.md` around what the engine actually is and does (P0-1), plus:
`docs/QUICKSTART.md`, `docs/ARCHITECTURE.md`, `docs/METHODS.md` (every Lever-G option,
what it does, when to use it), `docs/BENCHMARKS.md` (the full comparison), and
`docs/LIMITS.md` (the Shannon ceiling argument and what we will *not* claim).

---

## 6. Benchmark protocol

Same protocol as today, extended. Cold page cache (`drop_caches`) before every run,
identical prompt, identical greedy decoding, and **bit-exactness asserted, not assumed**.

**Baselines**: AirLLM, DeepSpeed ZeRO-Inference, HF `accelerate` disk-offload,
DFloat11 (where the model fits), llama.cpp (flagged **lossy** — comparison is
speed-at-what-accuracy, not like-for-like).

**Models**: Qwen2.5-1.5B (fast iteration), Qwen3-14B (the current reference),
a 27B-class model (the goal).

**VRAM tiers**: 4 / 6 / 8 GB via the existing cap, so the table shows where each system
stops working entirely.

**Reported per run**: s/token, tokens/s, GB read/token, peak VRAM live *and* reserved,
io/decode/compute split (fixed per P1-7), preprocessing time, output token ids, and a
bit-exactness verdict. No curated subsets — the full field dump, as established.

---

## 7. Phasing

| Phase | Content | Gate to proceed |
|---|---|---|
| **1. Foundation** | P0-1..P0-8, dead-code removal, `EngineConfig` wired, metrics fixed, checksums, manifest versioning | full suite green; head-to-head unchanged or better |
| **2. Memory hierarchy** | Lever A (three-tier) + Lever B (deep I/O) | bit-exact; ≥1.5x over today's 17.38 s/tok |
| **3. Token amortization** | Lever E (KV cache) + Lever D (tree + interleave + better draft) | distributional correctness tests pass; ≥2x on top |
| **4. Product** | CLI, FastAPI, UI, Docker, installer, docs, ROCm | one-command install works on a clean machine |
| **5. Frontier** | Lever C (fused decode), Lever F (rANS+context) | bit-exact vs fallback path; VRAM floor ≤2 GB |
| **6. Proof** | Full benchmark matrix incl. 27B, all baselines | published table with reproduction scripts |

Phases 1–3 are where the engine lead comes from and should not be traded for product
polish. Phase 4 is what makes it a thing other people can actually use.

---

## 8. Risks and honest uncertainties

- **Lever C (fused decode) may not pay off.** Variable-length Huffman decode is a poor
  fit for the SIMT model — this is exactly why ZipServ moved to a *fixed-length* bitmap
  format (TCA-TBE) instead. We may have to adopt a fixed-length format and give back
  some ratio to make fusion work. That trade needs measuring, not assuming.
- **RAM tier gains are machine-dependent.** 19 GB RAM against a 37 GB compressed 27B
  caches ~38%. On a 64 GB machine it approaches 100% and the disk disappears; on a 16 GB
  machine it barely helps. Claims must be stated per-configuration.
- **Speculation gains depend on the draft**, and a good draft may need training. The
  18.8% acceptance measured today is the floor, not the expectation — but a trained
  EAGLE head is real work with no guarantee at this scale.
- **ROCm is a genuine unknown** on this hardware (we have no AMD GPU to test on). It
  will be implemented against the abstraction and marked **untested** until someone runs
  it on real AMD silicon. It must not be claimed as supported before then.
- **4 GB may remain out of reach** for 14B+ regardless of effort; the floor is set by
  the largest single tensor plus activations.

---

## 9. What will not be claimed

Discipline that has served this project and should survive contact with a README:

- No compression ratio above ~1.5x for lossless bf16 — it is information-theoretically
  impossible and we will publish the argument rather than hide it.
- No "lossless" label on any run using `quantize="q8"` or a lossy baseline.
- No speed number that has not been measured on real hardware with cold caches.
- No projection presented as a result. `[PROJECTED]` stays on the label until measured.
