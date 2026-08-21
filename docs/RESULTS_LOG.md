# Results Log

Every entry is a real run on the RTX 3080 Laptop (8 GB VRAM, 19 GB RAM,
WSL2, NVMe), Qwen3-14B unless noted, cold page cache before every timed run.
Entries are append-only — a regression stays in the log, it doesn't get
edited away. This is what lets us tell "we improved it" from "we broke it
and didn't notice," which has happened twice before in this project
(the `imap_unordered` locality regression, the prefetch race).

Baseline for comparison is always the most recent row *before* the change
being evaluated, not row 1.

---

## Baseline (docs/PROPOSAL.md written against this)

| Date | Config | Peak VRAM | s/token | GB/token | Note |
|---|---|---|---|---|---|
| 2026-08-19 | `vram_budget_gb=1.8, decode_slice_elems=1<<22` | 1.69 GB | 19.78 | 18.73 | matched-VRAM floor |
| 2026-08-19 | `vram_budget_gb=2.0, decode_slice_elems=1<<22` | 1.89 GB | 19.94 | 18.58 | |
| 2026-08-19 | `vram_budget_gb=3.0, decode_slice_elems=1<<22` | 2.88 GB | 17.52 | 17.86 | |
| 2026-08-19 | `vram_budget_gb=4.0` (default slices) | 3.90 GB | 14.15 | 17.41 | |
| 2026-08-19 | `vram_budget_gb=6.0` (default slices) | 5.88 GB | 14.25 | 16.05 | |
| 2026-08-19 | AirLLM | 1.57 GB | 28.12 | 29.54 | reference |

Time split at the 2 GB config: **io ≈ 14 s/tok, decode ≈ 13 s/tok, compute ≈
0.0 s/tok** — decode and disk are co-bottlenecks, which is the premise
PROPOSAL.md's H2 is built on.

---

## H2 gate — CPU decode throughput microbenchmark

Real tensor from the store (`model.layers.10.mlp.down_proj.weight`,
89.1M weights, 87,040 chunks), gate threshold ≥1 GB/s decoded output.

**Attempt 1 — numpy fancy-indexing, vectorized across chunks.** Bit-exact
(verified against the reference decoder, including a partial-chunk-range
test proving the cross-chunk-boundary read is provably safe). Throughput:
**0.037 GB/s peak at 4 threads, then regresses** (0.007 GB/s at 16 threads
— more threads made it *worse*, real contention, not just "needs tuning").
**27x below gate. Failed.**

**Attempt 2 — numba `@njit(parallel=True)`, per-chunk-bounded (mirrors
`ChunkedBitReader` exactly, no cross-chunk-boundary reads needed).**
Bit-exact. Throughput:

| Threads | GB/s |
|---|---|
| 1 | 0.109 |
| 2 | 0.214 |
| 4 | 0.424 |
| 8 | 0.777 |
| 16 | **1.33** |

**Clears the gate at 8+ threads.** For context: this engine's own GPU
decode, measured in situ, runs at ~1.1 GB/s aggregate (20.33 GB compressed
/ ~13 s at the 2 GB config) — the CPU path at 16 threads is now in the
*same order of magnitude as the GPU*, which is what makes splitting work
between them plausible rather than pointless.

**Decision: H2 proceeds, using the numba path exclusively.** The numpy
path is kept in the codebase (documented as a failed approach, not
deleted — same policy as every other negative result in this project) but
is not used by the engine.

---

## H2 — real engine measurement (Qwen3-14B, `vram_budget_gb=2.0`, N=4 tokens)

Isolated gate passed clearly (1.33 GB/s at 16 threads, same order as GPU).
**Wired into the engine, the result reverses:**

| `cpu_decode_fraction` | s/token | io (s) | gpu_decode (s) | cpu_decode (s) | vs baseline |
|---|---|---|---|---|---|
| 0.00 | 19.05 | 93.1 | 67.4 | 0.0 | 1.00x |
| 0.25 | 19.04 | 100.1 | 57.7 | 18.2 | 1.00x (flat, not the win the gate predicted) |
| 0.50 | 21.54 | 113.6 | 48.8 | 34.4 | **0.88x — regression** |
| 0.75 | 26.47 | 123.5 | 24.7 | 60.5 | **0.72x — regression** |
| 1.00 | 36.69 | 175.9 | 4.7 | 84.9 | **0.52x — nearly 2x worse** |

`gpu_decode_s` drops exactly as predicted as more tensors move to CPU — that
part of the mechanism works. But `io_s` grows in lockstep (93→176s), and
total wall time is flat-at-best, monotonically worse beyond 25%.

**Root cause (most likely): the overlap this lever depends on doesn't
actually happen.** CPU decode runs *inside* `_read_layer_tensor_arrays` --
the same background-thread function that does the disk reads it was meant
to overlap with. `_load_layer` `join()`s that thread as one atomic unit, so
CPU decode time is added to that thread's own critical path (delaying when
the REST of that layer's tensors become available too) rather than running
concurrently with GPU work on other tensors. It also puts a
16-thread-wide `numba.prange` call, repeated per tensor (up to 1063 times
at fraction=1.0), directly in the path of the thread doing the actual
`read()` syscalls — plausibly starving it of scheduling time, which would
explain `io_s` inflating even though the measured disk-read boundary
didn't move.

**This is exactly the failure mode the literature warned about**
(LITERATURE.md §14, APEX/Q-Infer: "long CPU tasks leave the GPU idle and
destroy the benefit of offloading") -- except here it's not the GPU
sitting idle, it's the I/O thread being starved by CPU-side contention. A
throughput gate measured in isolation did not predict this; the
integration/scheduling design mattered as much as raw throughput.

**Verdict: H2 does not deliver a wall-clock win as currently implemented.**

> *Superseded by "Disposition of H2" at the end of this file: the flag was
> subsequently **removed** rather than left default-off. This paragraph is
> kept as written because the log is append-only.*
Kept in the codebase, bit-exact and fully tested, because the mechanism
(and the gate result) may still be salvageable with a different
integration -- decoding on a thread genuinely independent of the I/O
reader, rather than nested inside it -- but that is new design work, not
a config flag, and is out of scope for this pass. Recorded as a real
negative result, not deleted.

---

## H1 — real engine measurement (Qwen3-14B, `vram_budget_gb=2.0`, N=4 tokens)

**`ram_tier_format="decoded"` (the pre-existing default) failed with a CUDA
OOM on this machine, reproduced 3 times across ram_budget_gb ∈ {4.0, 1.5,
1.5-with-a-fix}**, always at the identical line: `gpu_tensor.to("cpu")
.pin_memory()`. Root cause, isolated rather than guessed at:

```
$ ulimit -l
65536        # 64 MB -- WSL2's default max-locked-memory limit
```

`pin_memory()` page-locks (`mlock`s) host memory so the GPU can DMA into it
directly. 64 MB is smaller than a single realistic decoder-layer tensor,
so the very FIRST RAM-tier tensor to pin fails, regardless of
`ram_budget_gb` -- which is exactly why the failure didn't move when the
budget did, and why adding `torch.cuda.empty_cache()` after each transient
tensor (a real fix for a real, separate peak-VRAM concern, kept because
it's correct regardless) didn't change the outcome. This is an environment
limit, not a bug in this codebase, and not something raising
`ram_budget_gb` or `vram_budget_gb` can work around.

**`ram_tier_format="compressed"` (H1) has no pin_memory() requirement at
all -- it holds plain numpy arrays -- and worked normally at both budgets
tested:**

| `ram_budget_gb` | s/token | GB/token | RAM tensors (orig GB) | disk tensors |
|---|---|---|---|---|
| 4.0 | 18.14 | 15.80 | 100 (3.98 GB) | 172 |
| 1.5 | 17.93 | 17.51 | 15 (1.50 GB) | 257 |

Bit-exact in both runs (verified separately by the synthetic-model test
suite; these real runs additionally match the known-correct reference
output).

**Verdict: on this development machine, H1's "compressed" option is not
merely more RAM-capacity-efficient than the original hypothesis
predicted -- it is currently the ONLY functional RAM-tier option for
tensors of real-model size.** A clean decoded-vs-compressed apples-to-apples
speed comparison could not be completed here for that reason; the
capacity claim (more tensors resident per GB) is directly confirmed by the
tensor counts above scaling with budget. Raising the host's `ulimit -l`
(or the equivalent WSL2 `.wslconfig` setting) would very likely unblock
"decoded" for a future comparison, but is a system-level change outside
this codebase and was not made unprompted.

---

## Disposition of H2 (2026-08-19)

**The engine-side CPU-decode dispatch was removed, not left as a default-off
flag.** A knob that is measured to only ever make things worse is a footgun,
and leaving it reachable invites someone (including a future me) to "just try
turning it up."

Removed: `EngineConfig.cpu_decode_fraction`, the `_should_cpu_decode` /
`_recombine_cpu_decoded` methods, the CPU branch in
`_read_layer_tensor_arrays` / `_load_layer`, the extra `StreamStats` counters,
and the `--cpu-decode-fraction` CLI flag. The prefetch path is back to the
GPU-only 2-tuple form it had before. 223 tests pass after removal.

Kept, repurposed rather than deleted:
- **`afterimage/runtime/cpu_decode.py`** — now documented as (a) the basis for
  a **CPU-only fallback** path, since the Triton kernels need CUDA and this is
  the only decoder that runs without a GPU, and (b) a verified negative result
  with its failure mode recorded in the module docstring.
- **`tests/test_cpu_decode.py`** (7 tests) — the decoder is bit-exact and
  stays proven so.
- **`scripts/cpu_decode_gate.py`**, **`scripts/h2_cpu_decode_sweep.py`** — the
  evidence, re-runnable.

**The transferable lesson, recorded because it cost a full implementation
cycle to learn:** the isolated throughput gate *passed* (1.33 GB/s, matching
GPU decode in situ) and the integrated result was still 0.52x. Component
benchmarks validate a component; only end-to-end wall-clock validates a
change. Every hypothesis in [PROPOSAL_ADAPTIVE.md](archive/PROPOSAL_ADAPTIVE.md) is
gated on an end-to-end A/B for exactly this reason.

---

## Adaptive speculation (PROPOSAL_ADAPTIVE.md / ADAPTIVE_TEST_PLAN.md), real
## measurements (Qwen3-14B, RTX 3080 Laptop, 2026-08-19)

All runs below: `scripts/adaptive_bench.py`, cold page cache, peak VRAM via
`torch.cuda.max_memory_allocated()`, prompt "What is the capital of
France?". **Scale note, disclosed up front:** the test plan specifies 8
prompts x 32 tokens per cell; these runs use 1 prompt x 4-16 tokens (same
reduced-N convention as the H1/H2 sweeps above), to keep real GPU wall-clock
inside this session. T0 in particular is too small-N to be conclusive (see
below) — everywhere else, the effect sizes are large enough that N=4-16 still
gives a clean, decisive read, which is stated explicitly per result, not
implied.

### T1 — matched-VRAM arm comparison (the headline)

Total budget 6.0 GB. Arm B's engine budget = 6.0 − 1.3 (draft model's own
measured resident footprint); Arm C's is the full 6.0 (self-drafting has no
cost outside the planner's own budget — see adaptive_bench.py's docstring).

| Arm | peak VRAM | s/token | tok/sweep | vs A |
|---|---|---|---|---|
| A — greedy | 5.87 GB | 20.58 | 1.00 (n/a) | 1.00x |
| **B — small draft model** | **5.77 GB** | **6.06** | **4.00** | **3.40x** |
| C — self-draft (layers 0-3 pinned) | 5.84 GB | 19.71 | 1.00 | 1.04x (noise) |

**B is a real, matched-VRAM win: 3.4x faster at slightly LESS peak VRAM than
greedy** (draft model resident cost roughly cancels the VRAM the shrunk
engine budget gives up). tok/sweep=4.00 at n_tokens=4 means every drafted
token in the single sweep was accepted — a strong, clean result, though a
longer run (T2 below) shows acceptance is not usually this high.

**C is flat — self-drafting produced no benefit over greedy at this budget.**
tok/sweep=1.00 means the draft was rejected essentially every time; every
sweep degraded to exactly the one-token "resample" case verify.py's
mechanism guarantees regardless of draft quality. Investigated further below.

### Self-draft acceptance vs depth (mechanism A — investigating the flat result)

| exit_layer | depth | tok/sweep | pin_draft_layers=True s/tok | =False s/tok |
|---|---|---|---|---|
| 4 / 40 | 10% | 1.00 | 19.71 | 22.17 |
| 8 / 40 | 20% | 1.00 | 20.25 | 31.80 |

Doubling draft depth (4→8 layers) did not move acceptance at all — still
effectively 0%. **This falsifies mechanism A as implemented: an untrained
Qwen3-14B checkpoint's own early layers do not produce next-token logits
useful enough to accept, at either depth tested.** This was flagged as the
leading risk in PROPOSAL_ADAPTIVE.md §6 before any of this was run ("Early-
exit drafts may be poor without training... exactly what step 2 measures
before anything is built on it") — the risk materialized. LayerSkip's
published 2.16x depends on training the model with an early-exit auxiliary
loss; nothing here was trained. Going deeper still (e.g. 20/40 layers) was
not tested — 20 layers alone is ~13 GB, infeasible on this 8 GB card
regardless of outcome, so the question is moot for this hardware even if
deeper exit layers would eventually help.

**Mechanism C (pinning) is CONFIRMED independently of A's failure.** At both
depths, unpinned self-draft is markedly slower than pinned (1.13x at depth 4,
**1.57x at depth 8** — the gap widens with depth, exactly as predicted: more
draft layers means more re-streamed bytes per sweep when they aren't pinned).
The vram_planner ranking change (TensorInfo.uses, docs in vram_planner.py)
does exactly what it was built to do. It currently has no working consumer
(mechanism A doesn't produce useful drafts with this checkpoint), but the
infrastructure is real, tested, and would immediately benefit a *trained*
early-exit draft if one were ever added.

### T2 — bandit policy vs fixed k (mechanism B, tested against the WORKING
### arm — small draft model, since self-draft had nothing to adapt)

16 tokens, spec_k initial/max=6, total budget 6.0 GB, temperature=1.0
(realistic sampling regime, not the temp=0 correctness check):

| policy | s/token | sweeps | final k |
|---|---|---|---|
| fixed (k=6) | 7.18 | 7 | 6 |
| gamma | 7.66 (**worse**) | 8 | 1 (collapsed) |
| threshold | 7.07 (~tied, within noise) | 7 | 6 |

**Neither adaptive policy beat the tuned constant — T2's own pre-stated kill
criterion.** GammaTune's EWMA saw a run of low-acceptance sweeps under
realistic temperature=1.0 sampling and contracted k all the way to 1, which
then can't recover within the ~7 sweeps this run had — exactly risk #2 from
PROPOSAL_ADAPTIVE.md §6 ("too few sweeps per run for a bandit to converge").
`spec_policy_state` persistence (letting a policy carry state across runs)
is implemented and unit-tested but was not exercised live in this pass —
open item, not a claimed result.

**Verdict: per the test plan's own rule, ship the constant.** `spec_k_policy`
stays available (opt-in, off by default) as tested, working infrastructure,
but `"fixed"` is the one actually recommended pending either longer runs or
cross-run state persistence being verified live.

### T0 — coupling check (best-k vs VRAM budget)

k ∈ {2,4,8}, budget ∈ {2.5, 6.0} GB, n_tokens=4. **Not conclusive at this
N** — flagging this rather than overclaiming: with only 4 tokens per cell,
`tok_per_sweep` was 2.00 for every single cell regardless of k or budget,
and s/token varied by less than 25% across the whole grid (9.6-12.2s) with
no clean monotonic pattern — consistent with run-to-run noise dominating any
real coupling signal at this sample size, not with a confident "yes/no" on
whether best-k shifts with budget. The plan's full 12-run x more-tokens
version would be needed to actually answer this; what ran here does not
retroactively justify skipping §1.4 (the planner change) OR confirm it was
needed — it's an open question, correctly left open rather than guessed at.

### Disposition (2026-08-19)

- **Mechanism A (self-speculation via early layers): FALSIFIED as tested.**
  Zero measured acceptance at two depths on an untrained checkpoint. Not
  removed from the codebase (draft_self_logits is bit-exact and tested,
  same "verified negative result, not deleted" policy as cpu_decode.py) but
  **not recommended for use** without a trained early-exit head, which is
  out of scope (this project compresses and streams weights; it does not
  fine-tune models).
- **Mechanism B (small resident draft model, now with generate_adaptive's
  temperature=0 correctness guarantee): CONFIRMED, real 3.4x speedup at
  matched (in fact slightly lower) peak VRAM vs greedy.** This was already
  roughly known qualitatively from generate_speculative; what's new is the
  clean matched-VRAM number and the T=0 correctness proof (see
  verify.temperature_probs) that makes every adaptive-arm result in this
  section a real per-run correctness check, not a distributional assumption.
- **Mechanism C (draft-layer-aware VRAM planning): CONFIRMED mechanically**
  (1.13-1.57x from pinning alone), **but currently has no working draft
  mechanism to pair with** — A is what C was built to make cheap, and A
  doesn't work yet. Kept as tested, correct, opt-in infrastructure.
- **spec_k_policy (bandit-tuned k): NOT justified at the sweep counts a
  single short generation provides.** Ships as opt-in, defaults to
  `"fixed"`; cross-run persistence remains untested live.

**Net effect actually usable today: mechanism B alone, via
`EngineConfig(draft_mode="model", spec_k=8)` through the new
`generate_adaptive` entry point** (functionally similar to the pre-existing
`generate_speculative`, but with the temperature=0 correctness guarantee and
a shared entry point with the other mechanisms). A and C remain real,
tested, opt-in code with a documented negative/inconclusive result each —
not vaporware, not deleted, not oversold.

### T4 — final head-to-head vs AirLLM, VRAM-matched sweep, temperature=0

`scripts/adaptive_bench.py t4`, prompt "What is the capital of France?",
8 tokens, cold cache, real 14B, real AirLLM. temperature=0 chosen
specifically so every system's answer is directly comparable (see
verify.temperature_probs) rather than diverging under sampling.

| system | budget requested | peak VRAM (measured) | s/token | answer |
|---|---|---|---|---|
| AirLLM | — | 1.57 GB | 29.38 | *(see caveat below)* |
| Afterimage greedy | 2.1 GB | 1.99 GB | 22.57 | " The capital of France is Paris. It" |
| Afterimage greedy | 4.0 GB | 3.89 GB | 19.14 | " The capital of France is Paris. It" |
| Afterimage greedy | 6.0 GB | 5.87 GB | 16.67 | " The capital of France is Paris. It" |
| Afterimage + small draft model | 2.1 GB | **INFEASIBLE** | — | — |
| Afterimage + small draft model | 4.0 GB | 3.78 GB | **3.24** | " The capital of France is Paris. It" |
| Afterimage + small draft model | 6.0 GB | 5.78 GB | **3.39** | " The capital of France is Paris. It" |

**Data-quality caveat, reported rather than hidden:** AirLLM's own generated
text this run was " What is the largest city in the United" — off-topic,
not "Paris." AirLLM printed `The attention mask is not set... you may
observe unexpected behavior` for this exact call (`return_attention_mask=
False`, inherited unchanged from the existing vram_matched_bench.py
baseline call, not something changed in this pass). Its peak-VRAM and
wall-time numbers are unaffected (pure instrumentation, independent of which
tokens came out) and are reported as measured; its answer text should not be
read as "AirLLM got the question wrong" so much as "this baseline call has a
known rough edge" — worth fixing (pass a real attention_mask) before ever
publishing a worked-example transcript that shows AirLLM's text.

**The VRAM-matching mistake this whole workstream exists to avoid, caught in
its own output:** `adaptive_bench.py`'s auto-generated summary line labelled
the small-draft-model numbers "speedup at comparable peak VRAM" — 8.67-9.06x.
That label is not accurate at face value: 3.78-5.78 GB is 2.4-3.7x AirLLM's
1.57 GB, not comparable. The honest statements are:

- **At genuinely matched, AirLLM-level VRAM (~1.6-2.1 GB), the small-draft-
  model arm does not run at all** — infeasible, confirmed by measurement
  (`vram_budget 0.80 GB is below the 1.73 GB needed`). The draft model's own
  ~1.3 GB plus the target's own ~1.7 GB minimum (lm_head + scratch) floor is
  structurally above AirLLM's footprint. This is a real, previously-unstated
  limit of mechanism B, not a rounding error.
- **Greedy-vs-AirLLM remains the correct matched-VRAM claim**: 1.30x-1.76x
  at 1.99-5.87 GB vs AirLLM's 1.57 GB, consistent with the existing
  vram_matched_bench.py numbers this project already publishes.
- **The small-draft-model numbers (3.24-3.39 s/token, 6.6-6.9x over
  Afterimage's OWN greedy at the SAME budget) are real and worth stating**,
  just correctly labelled: "6.9x faster than this engine's own greedy mode
  at 4 GB," not "9x faster than AirLLM at matched VRAM." Both framings are
  true; only one is apples-to-apples on memory, and RESULTS_LOG.md exists
  specifically so the second one never gets published as if it were the
  first.
- tok/sweep=8.00 (every proposed token accepted) for this specific easy,
  high-agreement prompt at temperature=0 — a clean, real, verified result,
  but not a general acceptance-rate claim; T2's 16-token/temperature=1 run
  above shows real acceptance is well under 100% on a less trivial sample.

---

## Chunked lm_head projection — lowers the VRAM floor, but is NOT lossless
## (2026-08-19)

**Why it was attempted.** Peak VRAM for a layer-streaming engine is bounded
below by the largest tensor it must hold at once. On Qwen3-14B that is
`lm_head` at 1.556 GB — and AirLLM sits at 1.57 GB for the same reason.
**Both systems were pinned to the same floor by the same tensor**, so no
"we use less memory than AirLLM" claim was reachable at all.

Logits are a concatenation over output rows with no interaction between
blocks (`logits[..., a:b] = x @ W[a:b].T`), so the projection can be
computed a block at a time, holding ~84 MB instead of 1.556 GB.
Implemented as `compressed_store.decompress_rows_gpu` +
`EngineConfig.lm_head_slice_rows`.

**The weights side works and is exact.** `decompress_rows_gpu` reassembles
byte-for-byte what whole-tensor decoding produces, at every block size
including ones that do not divide the row count (tested). Row boundaries
land on chunk boundaries for free on this model (5120 cols / 1024 chunk =
5 chunks per row), and the general covering-chunk-range path handles the
unaligned case.

**The matmul side is NOT bit-exact, measured at production shape:**

| accumulation setting | block size | bit-identical | max logit deviation |
|---|---|---|---|
| default (bf16 reduction) | whole (9496) | yes | 0.0 |
| default | 2374 | **no** | **2.0** |
| default | 7 | **no** | 1.0 |
| `allow_bf16_reduced_precision_reduction=False` | 2374 | **no** | 1.0 |
| `allow_bf16_reduced_precision_reduction=False` | 7 | **no** | 1.0 |

cuBLAS selects a different kernel and split-K reduction strategy per output
shape, so a blocked product accumulates in a different order than one full
product and bf16 rounding diverges. **Forcing fp32 accumulation does not fix
it** — that was tested specifically, and the deviation only fell from 2.0
to 1.0.

**Caught only because it was checked at real dimensions.** At the tiny
dimensions the synthetic test model uses (hidden=64) the blocked and
unblocked products ARE bit-identical, so a tiny-model test would have
passed and encoded a false guarantee. `hidden=5120` is where it breaks.

**Disposition: shipped opt-in and declared lossy.** `lm_head_slice_rows`
defaults to 0 (whole head, lossless). Setting it makes
`EngineConfig.is_lossless` return False and `describe()` print
`LOSSY: lm_head_slice_rows=N -- output is NOT bit-exact`, exactly as
`quantize="q8"` already does; the CLI prints the warning to stderr. **No
head-to-head number may be reported from a run with it enabled**, since the
comparison against AirLLM is a lossless-vs-lossless comparison.

`tests/test_chunked_lm_head_gpu.py::test_blocked_matmul_is_not_bit_exact_at_production_shape`
is a characterization test: if a future cuBLAS makes the two agree, it
fails, and that failure is the signal to re-evaluate promoting this to the
lossless path — not to loosen the assertion.

---

## CORRECTION — the matched-VRAM AirLLM claim was wrong (2026-08-19)

`scripts/matched_vram_final.py`, 8 tokens, cold cache, same peak-VRAM
counter for both, lossless both sides. This run was built specifically to
find the configuration that TIES AirLLM's memory, rather than reporting a
speedup at whatever memory happened to be convenient.

| System | peak VRAM | s/token | GB read/token |
|---|---|---|---|
| AirLLM | **1.57 GB** | **29.10** | 29.2 |
| Afterimage, budget 1.69 GB | — | **planner refused** | — |
| Afterimage, budget 1.70 GB | — | **planner refused** | — |
| Afterimage, budget 1.80 GB | 1.68 GB | **30.71** | 14.7 |
| Afterimage, budget 2.10 GB | 1.99 GB | 20.18 | 11.8 |

**Findings, including one that invalidates a previously reported number:**

1. **We cannot reach AirLLM's 1.57 GB at all.** `lm_head` is 1.556 GB and
   must be materialized; plus decode scratch and activation slack the floor
   is ~1.68 GB. Budgets of 1.69 and 1.70 GB were refused up front.

2. **At the closest reachable point (1.68 GB — still 7% MORE memory than
   AirLLM) we are SLOWER: 30.71 vs 29.10 s/token.** Reported plainly
   because it is the number the whole comparison rests on.

3. **The previously logged "1.30x faster at 1.99 GB vs AirLLM 1.57 GB" was
   not a matched-VRAM result** — 1.99 GB is 27% more memory. The speedup was
   real; the label was wrong. Same class of error as the original 2.66 GB
   vs 1.57 GB mistake this project already corrected once. Corrected here
   rather than edited away, per the append-only rule.

4. **Compression works, and it is still not enough at the floor.** Disk
   reads halve (14.7 vs 29.2 GB/token) — the mechanism does exactly what it
   claims. But wall time does not improve, because the saved I/O is spent on
   GPU decode that AirLLM never performs. At a 1.80 GB budget only ~3 MB
   remains after headroom (170 tiny norm tensors), so nothing stays
   resident, and decode slices must be small enough to fit the scratch
   budget, which multiplies kernel launches.

   **The compression advantage converts to wall-clock only once there is
   enough VRAM headroom to decode in large slices and keep tensors
   resident** — visible as the crossover at 1.99 GB (1.44x) and above.

**Standing claim after this correction:** at equal memory this engine is
NOT faster than AirLLM on this hardware. Its real advantage is the ability
to spend additional VRAM when available (1.44x at 2.0 GB, 1.75x at 5.9 GB,
and 9x at 4 GB with speculative decoding), which AirLLM has no mechanism to
do. That is a genuine capability difference and it is a different claim from
"faster at the same memory."

**Path to an actual matched-VRAM win:** the floor is `lm_head`, and the
chunked projection above removes it — but is not bit-exact, so it cannot
carry a lossless number. Making it shape-stable (pad every block to
identical dimensions so cuBLAS selects one kernel throughout, or a Triton
kernel with a fixed reduction order) would restore bit-exactness AND drop
the floor to ~0.2 GB, which would additionally let speculative decoding fit
under 3 GB. That is the concrete next step.

**Data-quality note:** AirLLM's generated text differs from ours because HF
`generate()` honours Qwen3's `generation_config` (which enables sampling)
while our greedy path takes the argmax. Timing and VRAM are unaffected;
AirLLM's answer text is not like-for-like and must not be presented as a
worked-example comparison without forcing `do_sample=False`.

---

## Chunked lm_head on the real 14B — the VRAM floor does drop, a lot
## (2026-08-19)

Same protocol, 4 tokens, cold cache, `decode_slice_elems=1<<20`,
`lm_head_slice_rows` as shown. **All of these are LOSSY runs**
(`is_lossless=False`) for the reason recorded in the previous section.

| `vram_budget_gb` | `lm_head_slice_rows` | peak VRAM | s/token | vs AirLLM VRAM |
|---|---|---|---|---|
| AirLLM (reference) | — | 1.57 GB | 29.10 | — |
| 1.20 | 8192 | **1.557 GB** | 28.55 | −1% |
| 0.80 | 4096 | **1.159 GB** | 29.08 | **−26%** |
| 0.50 | 2048 | **0.855 GB** | 29.25 | **−46%** |

**The floor really is gone.** The engine previously could not go below
1.68 GB; it now runs the same 14B in **0.855 GB — 46% less VRAM than
AirLLM — at the same speed** (29.25 vs 29.10 s/token, within noise).
Generated text was identical across all three (" The capital of France"),
matching the lossless runs, so the bf16 blocking deviation did not flip an
argmax on this prompt — evidence that it is small in practice, though not a
guarantee, which is exactly why the config still declares itself lossy.

**Speed is flat across the whole sweep** (28.55-29.25 s/token). Blocking the
head costs almost nothing in time, because per-token wall clock is dominated
by streaming the 40 decoder layers, not by the head. So this lever buys
memory, essentially for free, and buys no speed.

### What this does and does not license as a claim

- **Legitimate:** "runs a 14B in 0.855 GB of VRAM, 46% less than AirLLM, at
  the same tokens/second, with output that matched on the prompts tested" —
  stated together with "this configuration is not bit-exact."
- **NOT legitimate:** any lossless claim, and any speed claim. At equal
  memory and equal losslessness this engine is still *slower* than AirLLM
  (previous section), and that remains the standing result.
- **Still the right next step:** making the blocked matmul shape-stable
  (pad every block to identical dimensions, or a Triton kernel with a fixed
  reduction order) would move this entire table onto the lossless path and
  make it the headline result rather than a footnote. It would also let
  speculative decoding — which needs 1.3 GB for the draft model and
  currently cannot fit under ~3 GB — run in roughly 2 GB total, where its
  measured 3.24 s/token would be a genuine matched-VRAM win over AirLLM's
  29.10.

---

## Can a chunked head be made bit-exact by choosing the block size?
## No. (2026-08-19)

The previous section proposed "pad every block to identical dimensions so
cuBLAS selects one kernel throughout" as the fix that would move the chunked
head onto the lossless path. **That was tested and it does not work.**

Swept block size N against a single full matmul at the real lm_head shape
(V=151936, K=5120), for three sequence lengths M:

| N (rows/block) | M=1 | M=5 | M=9 |
|---|---|---|---|
| 75968 (V/2) | exact | exact | exact |
| 37984 | diff 2.0 | exact | exact |
| 18992 | diff 2.0 | diff 2.0 | diff 2.0 |
| 9496 | diff 2.0 | diff 2.0 | diff 1.0 |
| 8192 | diff 2.0 | diff 2.0 | diff 2.0 |
| 4096 | diff 1.0 | diff 2.0 | diff 2.0 |
| 2048 | diff 2.0 | diff 2.0 | diff 2.0 |
| 1024 | **exact** | diff 2.0 | diff 1.0 |
| 512 | diff 1.0 | diff 2.0 | diff 2.0 |

**The pattern is not a pattern.** N=1024 is bit-exact at M=1 and wrong at
M=5. N=37984 is wrong at M=1 and exact at M=5. Kernel selection depends on
the full (M, N, K) triple in a way that is not monotone, not
alignment-based, and not predictable from the outside — so there is no
"safe block size" to pin, and any that appeared safe would silently stop
being safe the moment the sequence length changed. A speculative sweep
alone varies M from call to call.

Only N = V/2 is consistently exact, and two blocks saves just half the
head (~0.78 GB), which is not where the interesting configurations are.

**Consequence, stated as a limit rather than a to-do:** splitting the output
projection cannot be made bit-identical to an unsplit reference through
cuBLAS. A custom kernel with a fixed reduction order would make the engine
self-consistent, but still would not match stock HF+cuBLAS output, which is
what "bit-exact" means for this project. **The chunked head is therefore
inherently a lossy option, not a not-yet-finished one.** The earlier
"concrete next step" framing in HOW_IT_WORKS.md and in the section above was
optimistic and is corrected here.

What remains legitimately checkable is whether the deviation ever changes
the emitted token — measured in the next section.

---

## ALL METHODS vs AirLLM — one table, one protocol (2026-08-19)

`scripts/methods_vs_airllm.py`, Qwen3-14B, 8 tokens, cold page cache before
every run, peak VRAM from the same counter for every row including AirLLM's.
**AirLLM now runs with `do_sample=False`** — without it HF `generate()`
honoured Qwen3's sampling `generation_config` and the baseline produced
different text than our greedy path, making transcripts incomparable. With
it, every row below returns the identical string.

| method | peak VRAM | s/token | lossless | vs AirLLM | VRAM vs AirLLM |
|---|---|---|---|---|---|
| **AirLLM (baseline)** | 1.568 GB | 27.37 | yes | 1.00x | — |
| 1. Compression only | 1.677 GB | 28.01 | yes | **0.98x** | +7% |
| 2. + residency @2.1 GB | 1.990 GB | 17.20 | yes | 1.59x | +27% |
| 2. + residency @4.0 GB | 3.888 GB | 15.22 | yes | 1.80x | +148% |
| 3. + chunked head | **0.855 GB** | 26.38 | **NO** | 1.04x | **−45%** |
| 4. + speculation @4 GB | 3.784 GB | **2.19** | yes | **12.48x** | +141% |
| 5. chunked head + speculation | 2.051 GB | 3.28 | **NO** | 8.34x | +31% |

Prompt "What is the capital of France?" → every row answered
" The capital of France is Paris. It".

### Run-to-run variance, measured rather than assumed

The matched-VRAM pair (AirLLM vs method 1) has now been measured three
times independently:

| run | AirLLM | method 1 | ratio |
|---|---|---|---|
| 1 | 29.10 | 30.71 | 0.95x |
| 2 | 27.71 | 27.05 | 1.02x |
| 3 | 27.37 | 28.01 | 0.98x |

Mean 0.98x, spread ±4%. **At N=8 tokens, single run, differences under
~10% are not resolvable.** Two consequences, both recorded so they are not
re-litigated later: the retracted "1.30x at matched VRAM" was inside the
noise band as well as at the wrong memory; and run 2's favourable 1.02x
must not be quoted as a win either. **The supported statement is parity.**

### Token agreement — does the non-bit-exact path change the answer?

Every method, including both LOSSY chunked-head rows, produced
**token-identical output** to the lossless greedy path on this prompt. The
blocking deviation (1-2 absolute in logits) did not flip a single argmax.

That is a genuine practical result and it is **not** a correctness proof:
one prompt, 8 tokens. It says the deviation is small relative to typical
logit gaps, not that it can never change a token. The config still declares
itself lossy, correctly.

### What each row actually shows

1. **Compression alone does not beat AirLLM.** It halves disk I/O
   (14.7 vs 29.2 GB/token, measured earlier) and spends the saving on GPU
   decode that AirLLM never performs. Net: parity.
2. **Residency is where compression pays off** — but it is bought with
   memory, not cleverness. Note the diminishing return: 2.0→3.9 GB (2x the
   memory) buys 17.20→15.22 s/token (13%).
3. **Chunked head is the only row that beats AirLLM on memory** — 45% less,
   at parity speed. Not bit-exact.
4. **Speculation is by far the largest lossless win: 12.48x.** It is also
   the least novel part of this engine — standard draft/verify — but it
   composes with everything else and costs only the draft model's 1.3 GB.
5. **The two levers compose.** Chunked head frees ~0.8 GB, which is roughly
   what the draft model needs, so speculation now fits in 2.05 GB instead of
   the ~3 GB floor it had before — 8.34x at +31% VRAM rather than 12.48x at
   +141%. This configuration did not exist before this pass.

### Standing summary

- **Lossless, matched VRAM: parity with AirLLM (0.98x).** No speed claim.
- **Lossless, more VRAM: 1.6x-1.8x (residency), 12.5x (speculation).**
  Real, and honestly attributable to spending memory AirLLM cannot spend.
- **Lossy, less VRAM: 45% below AirLLM's footprint at parity speed.**
- **Lossy, balanced: 8.3x at +31% VRAM** — arguably the most useful
  operating point on an 8 GB card, and the one to reach for if the
  bit-exactness requirement is ever relaxed.

---

## Bounded multi-prompt research screen (2026-08-21)

The prior one-prompt table above is preserved as history. A new immutable,
cold-cache screen used four held-out semantic prompt types × four tokens, with
two disjoint calibration prompts and a 58-minute cap. AirLLM was rerun with EOS
stopping disabled (rather than `min_new_tokens`, which suppresses EOS logits)
so the four-token sequences are comparable.

| method | peak VRAM | s/token | vs corrected AirLLM | contract |
|---|---:|---:|---:|---|
| AirLLM 3.1.0 | 1.583 GB | 28.861 | 1.00x | BF16 greedy baseline |
| minimum-memory exact | 1.723 GB | 32.514 | 0.89x | exact |
| residency @4 GB | 3.934 GB | 17.360 | 1.66x | exact |
| chunked head | 0.901 GB | 29.606 | 0.97x | approximate |
| fixed-k speculation | 3.813 GB | 9.150 | 3.15x | greedy-token exact at T=0 |
| frozen hazard speculation | 3.814 GB | 9.773 | 2.95x | greedy-token exact at T=0 |

Expected answers and output token IDs matched on every shared prompt after the
AirLLM protocol correction. The screen is one repeat, so it is not a
confirmatory confidence interval.

H0 oracle headroom was 2.56% (<12% gate); H1 critical-path placement gained
1.61% (<8% gate); H2 hazard stopping was 6.4% lower-throughput than fixed-k;
H4 PI prefetch was 35.7% lower-throughput than fixed depth; H5 certified MIPS
was 30.5% lower-throughput and pruned only 0.084% of rows. H3/H8 were gated;
H6 lacked alternative artifacts; H7 was inapplicable to this dense checkpoint.

Full analysis, literature boundary, caveats, and raw-file links:
[BOUNDED_RESEARCH_REPORT_2026-08-21.md](BOUNDED_RESEARCH_REPORT_2026-08-21.md).
