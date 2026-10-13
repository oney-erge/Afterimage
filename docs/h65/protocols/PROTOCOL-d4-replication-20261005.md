# Frozen protocol: D4 replication with storage-stall prevention (RTX 5090, Llama-3.3-70B)

Frozen 2026-10-05, before any measurement. Requested by the author after earlier
comparisons with the same arms were affected by storage stalls. This study replicates D4
(Figure 6, schedule-mechanism comparison) and P1's T/K/H arms, using fresh prompts and
symmetric stall-prevention measures. It replaces nothing and is reported whatever the
outcome. It is not pooled with D4, P1 or any other study.

## Arms (identical to D4 and P1; frozen files)

| Arm | Plan | Source |
|---|---|---|
| T | traffic control, `926b0adb…` | P1 copy of the pilot traffic plan |
| K | greedy (profiled knapsack), profile `b88db4b5…`. Runtime tier fingerprint must equal `344a8f46…` (mechanism control v2 / P1). | P1 copy |
| H | H6.5 endpoint plan, `a31a0d4b…` (the plan used in D1, D2, D4 and P1) | P1 copy |

## Engine contract (as D4 and P1, identical for every arm)

- **Inference:** batch one, greedy decoding, exactly one output token, full output head, no
  draft model, exact weights.
- **Memory limits:** 8 GB logical budget, 10 GB allocator cap, 12 GB whole-process admission
  ceiling, 16 GB host budget, 0.5 GB safety margin.
- **Engine settings:** 4,194,304-element decode slices, prefetch depth 2, per-blob reads,
  decode-table reuse, decoded RAM tier, static execution.
- **Per cell:** fresh process, one warm-up token, cold page cache before the timed request,
  and the GPU cooldown (10 s, at or below 75 °C). The archived thermal and power gates apply.
- **Checks:** runtime plan and read counters must match the frozen T and H plans, and the K
  tier fingerprint must match.

## Storage-stall prevention (new; symmetric, fixed before timing)

1. **Keep awake.** A Windows keep-awake (`SetThreadExecutionState`) holds the system awake for
   the run, because the previous run died at a system sleep. Power settings are not changed.
2. **Storage idle.** Before every cell, in every arm, the controller waits 120 s with no work
   so the SSD can cool after the previous cell's ~150 GB of reads. Earlier stalls clustered
   in the second or later cell of a block and late in runs.
3. **No competing work.** No other campaign, agent or disk-heavy job runs. The lock file is
   `paper-run.lock`.

## Design

- **Blocks:** 12, each with one fresh prompt (`d4r-prompts.json`, 72-110 input tokens, no
  overlap with any earlier set).
- **Order:** a Williams design for three arms. All 6 orders are used in blocks 1-6 and again
  in blocks 7-12: TKH, KHT, HTK, THK, KTH, HKT.
- **Limits:** 10-minute cell cap and 270-minute total cap.
- **Stops:** a frozen input change, a T/H runtime plan mismatch or a K fingerprint mismatch
  stops the run. A thermal or power window failure, a failed cell, or token mismatch makes
  that block ineligible. No reruns, replacement blocks, outlier removal or sample extension.

## Predeclared analysis

- **Primary:** latency ratios T/H and K/H over all eligible blocks. Geometric mean with a
  two-sided 95% Student-t interval on block log ratios (df = eligible − 1). Above 1 means H6.5
  is faster.
- **Also primary:** peak incremental GPU memory ratios T/H and K/H.
- **Secondary, predeclared:** the same statistics excluding any block in which any cell's
  prefetch wait exceeds twice its own arm's median prefetch wait in this run.
- **Also reported:**
  - blocks 1-6 and 7-12 separately (each a complete Williams set);
  - per-arm means for wall time, peaks, host RSS, read bytes and calls, and prefetch wait;
  - the stall rate compared with earlier runs.
- **Interpretation:** an interval crossing 1 leaves the direction unresolved. Equivalence is
  not claimed.

## Scope

- **Measured case:** cold one-token latency and peak GPU memory on this RTX 5090, NVIDIA
  driver 616.92.
- **Not comparable without caveat:** D4 ran on driver 610.88, with different prompts and no
  storage idle.
