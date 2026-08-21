# Proposal — Heterogeneous Decode and Compressed Residency

**Target: a 40B-class model on an 8 GB consumer GPU, bit-exact, at a usable
token rate.**

This document states four hypotheses, why each is plausible from measurement
rather than intuition, exactly how each will be falsified, and what result would
make me abandon it. It builds on [LITERATURE.md](LITERATURE.md) Part II §17.

Nothing here changes the existing engine's behaviour. Every proposed mechanism
is an **opt-in `EngineConfig` field defaulting to today's behaviour**, so a
failed hypothesis costs a flag, not a regression.

---

## 0. Where the time actually goes

Measured, Qwen3-14B, RTX 3080 Laptop, per token, at the 2 GB budget:

```
disk I/O            ~14 s     <-- co-bottleneck
GPU entropy decode  ~13 s     <-- co-bottleneck
GPU compute        ~0.0 s     <-- negligible, as expected for a memory-bound decode
PCIe transfer       ~1.6 s    <-- 18.8 GB compressed at ~12 GB/s: NOT a bottleneck
CPU utilisation     ~idle     <-- 16 cores, doing almost nothing
```

Two facts drive everything below:

1. **Decode is not a rounding error — it is roughly half the wall clock.** Every
   previous plan treated I/O as *the* bottleneck. It is only half of one.
2. **There is a whole idle CPU, and PCIe has ~8x headroom.** The system is using
   one of its three compute resources and one of its three data paths.

### The 40B arithmetic, honestly

| | Dense 40B | MoE 40B-class (e.g. ~13B active) |
|---|---|---|
| bf16 weights | 80 GB | 80 GB |
| compressed @1.45x | 55 GB | 55 GB |
| **touched per token** | **55 GB (all of it)** | **~18 GB (active experts only)** |
| disk-only, @2 GB/s | ~27 s/tok | ~9 s/tok |
| + 14 GB compressed RAM tier | ~20 s/tok | ~4-6 s/tok |
| + speculation @3 tok/sweep | **~7 s/tok** | **~1.5-2 s/tok** |

**The honest headline: dense 40B on this machine lands around 7 s/token; a
40B-class MoE lands near 2 s/token.** Anyone promising dense-40B interactivity
on 8 GB is not doing this arithmetic. The MoE column is where "40B on 8 GB"
becomes genuinely comfortable, and it is also where the least work has been done
by anyone else (§15).

---

## H1 — A compressed RAM tier holds ~1.45x more model per GB

### Hypothesis
The RAM tier currently caches **decoded** bf16 weights
(`_ram_cache[key] = gpu_tensor.to("cpu").pin_memory()`). Caching the
**compressed** bytes instead — and decoding on the way to the GPU, exactly as
the disk tier already does — fits 1.45x more model in the same RAM, cutting
disk bytes/token proportionally for any budget where the RAM tier is not empty.

### Why it should hold
Purely mechanical: the same RAM holds `ratio` times more bytes of a compressed
representation, and the decode path required to use it already exists and is
already exercised on every disk-tier tensor. The cost is GPU decode work for
RAM-tier tensors, which today are free memcpys.

### Why it might not
Trading a memcpy for a decode is only a win if disk time saved exceeds decode
time added. If decode is already saturated (H2 says it nearly is), a compressed
RAM tier could push the decoder further into being *the* bottleneck and lose.
**H1 and H2 are coupled, and H1 should be evaluated with H2 both on and off.**

### Test
`EngineConfig(ram_tier_format="decoded" | "compressed")`, default `"decoded"`.
Fixed `ram_budget_gb=8`, sweep `vram_budget_gb`, measure disk bytes/token and
wall time both ways on the 14B.

### Predicted result
Tensors resident in RAM up ~1.45x; disk bytes/token down 10-25% depending on
budget. Wall time down if and only if disk remains the binding constraint.

### Falsified if
Disk bytes/token does not drop by at least 10%, or wall time regresses at every
budget tested.

---

## H2 — Splitting entropy decode across CPU and GPU raises aggregate decode throughput

> **STATUS: TESTED AND FALSIFIED (2026-08-19). Implementation removed.**
> The isolated throughput gate below passed (1.33 GB/s at 16 threads via a
> numba-compiled decoder). Wired into the engine it made wall time flat at a
> 25% CPU share and monotonically worse beyond — 0.88x / 0.72x / 0.52x. The
> intended overlap never happened: CPU decode ran inside the same thread as
> the disk reads it was meant to overlap with. Full analysis and disposition:
> [RESULTS_LOG.md](RESULTS_LOG.md). Successor work: [PROPOSAL_ADAPTIVE.md](PROPOSAL_ADAPTIVE.md).

**This was the central claim of the proposal, and the one I believed was novel.**

### Hypothesis
Decode (~13 s/token) is a co-bottleneck with disk (~14 s/token) while 16 CPU
cores sit idle and PCIe runs at ~13% utilisation. Dispatching a fraction of
tensors to a **multithreaded CPU decoder**, concurrent with the GPU decoding
others, raises total decode throughput and lowers wall time.

### Why it should hold
- The chunked format was built so chunks decode independently *on the GPU*.
  That property is device-agnostic: it makes multicore CPU decode equally
  possible, with **no format change** (§16).
- Published single-thread Huffman decode is ~1.3-1.4 GB/s; even at a
  conservative 300 MB/s/core across 12 usable cores that is ~3.6 GB/s of
  exponent symbols, i.e. ~7 GB/s of bf16 weight output — the same order as the
  GPU's *effective* (not microbenchmark) decode rate in situ.
- The extra cost is PCIe: a CPU-decoded tensor crosses uncompressed (2
  bytes/weight) instead of compressed (~1.38). At a 50% split that is ~23 GB/token
  over PCIe ≈ 1.9 s — still far below the ~14 s wall. **We are trading an
  abundant resource for a scarce one.**

### Why it might not
- Python/GIL overhead could make the CPU decoder far slower than the C-level
  numbers suggest. Mitigation: the inner loop must be numpy-vectorised per chunk
  (as `pack_bits` already had to be) or released into a C extension; a pure
  Python per-symbol loop will not work — that lesson is already recorded in
  `huffman.py`.
- CPU threads contend with the prefetch reader threads for memory bandwidth and
  cores. The APEX result (§14) is explicit that badly-scheduled CPU work
  *destroys* the benefit.
- If disk is more dominant than measured, freeing decode capacity changes nothing.

### Test
1. **Microbenchmark first, before any engine work.** Decode a real 14B tensor's
   chunk range on CPU with N threads, N ∈ {1,2,4,8,16}; report GB/s of output.
   *Gate: if aggregate CPU throughput < 1 GB/s of weight output, stop — the
   hypothesis is dead and no engine change is justified.*
2. If it passes, add `EngineConfig(cpu_decode_fraction=0.0)` (default off) and a
   least-loaded dispatcher that routes each tensor to whichever decoder will
   finish it soonest.
3. A/B wall time on the 14B at fractions {0, 0.25, 0.5, 0.75}, with
   bit-exactness asserted at every setting.

### Predicted result
Wall time minimum at an intermediate fraction (~0.3-0.5), worth **1.2-1.5x**.
Fraction 1.0 should be *worse* than 0.5 — if it isn't, the GPU decoder is
underperforming and that is a separate bug worth finding.

### Falsified if
The microbenchmark gate fails, or no fraction beats 0.0 by more than run-to-run
noise (~5%).

---

## H3 — Compressed MoE expert caching makes 40B-class models practical

### Hypothesis
For an MoE model, per-token traffic is proportional to *activated* experts, and
activation is skewed. A **value-density-ranked, losslessly-compressed expert
cache** in VRAM+RAM cuts disk traffic far below the dense-model floor — and
holds ~1.45x more experts per GB than every uncompressed expert cache in the
literature (§15).

### Why it should hold
The MoE offloading literature is unanimous that expert fetching dominates
(98.9% of time for Mixtral-8x7B) and that caching works. This engine already
has the residency planner and the codec; MoE support is mostly *routing
awareness* — knowing which tensors are experts and that only some are needed.

### Why it might not
- Requires real MoE support in the engine (expert-granular tiers, gate-aware
  loading). That is genuine new machinery, not a config flag.
- Skew may be weaker than published for the specific model tested.
- Compressed experts add decode latency on the *critical path* of a cache miss,
  where an uncompressed cache would just memcpy. For a miss discovered late in a
  layer this could hurt more than the extra cache capacity helps.

### Test
Compress an MoE model (Mixtral-8x7B or a Qwen3-MoE variant), instrument
per-expert activation counts over a real prompt set, then measure disk
bytes/token vs expert-cache size for compressed and uncompressed caches at equal
GB.

### Predicted result
Disk bytes/token for an MoE 40B-class model well below the dense floor, and the
compressed cache dominating the uncompressed one at equal GB.

### Falsified if
Expert activation is close to uniform (no skew to exploit), or the compressed
cache's miss-path decode cost exceeds the benefit of its extra capacity.

---

## H4 — The composition reaches the 40B target

### Hypothesis
H1 + H2 + existing speculative decoding put a **dense** 40B-class model at
≤10 s/token and an **MoE** 40B-class model at ≤3 s/token on 8 GB VRAM / 19 GB
RAM, bit-exact.

### Test
Compress a real 40B-class model of each kind and run the standard cold-cache
protocol. No projections in the result — the number is whatever the clock says.

### Falsified if
Either target is missed by more than 2x, in which case the arithmetic in §0 is
wrong somewhere and the error must be found and published before any further
claim is made.

---

## Order of work, and why

| # | Work | Effort | Risk | Gate before proceeding |
|---|---|---|---|---|
| 1 | **CPU decode microbenchmark** (H2 step 1) | S | none | ≥1 GB/s aggregate or H2 dies here |
| 2 | Compressed RAM tier (H1) | S | low | bit-exact; disk bytes/token down ≥10% |
| 3 | Heterogeneous decode dispatcher (H2) | M | medium | bit-exact at every fraction; ≥1.2x |
| 4 | 40B dense run (H4, part 1) | S | low | it completes and is bit-exact |
| 5 | MoE expert residency (H3) | L | medium | disk bytes/token below dense floor |
| 6 | 40B MoE run (H4, part 2) | S | low | published with full field dump |

**Step 1 is deliberately a throwaway benchmark, not a feature.** It costs an
afternoon and can kill the most expensive idea in this document before any
engine code is written. Every previous lever in this project that went wrong
(the `imap_unordered` locality regression, the prefetch race) went wrong because
a plausible mechanism was built before being measured in isolation.

---

## What would make me abandon this direction

- CPU decode microbenchmark below 1 GB/s aggregate → H2 and much of the novelty
  claim collapse; fall back to fused decode (ZipServ-style) as the remaining
  lever, accepting the ratio loss their paper implies.
- Compressed RAM tier failing to reduce disk traffic → the residency model is
  wrong somewhere and needs re-deriving before more is built on it.
- Dense 40B landing above 20 s/token → the honest conclusion is that dense 40B
  on 8 GB is a *capability* result, not a *usability* one, and the project
  should say so plainly and focus entirely on MoE.

---

## What will not be claimed

Unchanged from the rest of this repository:

- No lossless bf16 ratio above ~1.51x. Proven, not aspirational.
- No number that was not measured on real hardware with cold caches.
- No projection reported as a result — everything in §0's 40B table is marked
  arithmetic until a clock says otherwise.
- No "faster than X" without stating the VRAM both systems used.
