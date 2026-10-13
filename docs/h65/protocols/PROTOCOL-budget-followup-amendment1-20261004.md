# Frozen protocol amendment 1: measured RAM costs and live budgets (RTX 5090, Llama-3.3-70B)

Frozen 2026-10-04, before any RAM calibration or live measurement under this amendment.
It amends `docs/h65/FINAL_5090_TODO_2026-10-04.md` (the plan), which fixed one live budget
(16 GB GPU / 16 GB host) and forbade choosing another budget. The drivers print this file's
SHA-256 and store it with every result.

## Why the plan is amended (disclosed in the paper)

1. **Pre-declared 16/16 budget: H6.5 returned the traffic plan.** Planning at 16/16 with the
   archived inputs gave a byte-identical plan to traffic (`RESULT-16v16r-PLAN-IDENTITY.md`;
   independently reproduced in `llama-16v16r-independent-verification/`). No live run.
2. **The archived RAM profile has a calibration gap at 16/16.** The RAM-preparation profile was
   measured under the 8/16 traffic placement, where the output head was on storage. At 16/16,
   traffic holds the head decoded in host RAM, but the head's RAM cost was never measured. The
   planner priced it at the bandwidth floor of 0.123 s. Offline, a head cost of 0.3 s or more
   changes the 16/16 plan (`diagnostics/head-breakeven-16v16r.json`). The method's own procedure
   (the pilot) measures RAM costs under traffic placement at the planning budget. Reusing the
   8/16 profile at another budget was therefore a deviation from the method, not a choice of
   the method.
3. **A full planning sweep was examined before this amendment.** Budgets were 4, 6, 8, 10, 12,
   14, 16, 20 and 24 GB GPU, each with 16 GB host
   (`diagnostics/ram-cost-gap-and-budget-sweep.json`; every budget is reported). With the
   archived costs, H6.5 differs from traffic in 46-198 tensors at 6-14 GB, where it keeps the
   head on the GPU. It differs in 0-4 tensors at 4, 16, 20 and 24 GB.
4. **The added budgets were chosen after that sweep and are disclosed as such.**
   - 12 GB: the midpoint between the measured 8 GB and the pre-declared 16 GB. It has the
     smallest predicted held-out gain (+0.27%) of the 6-14 GB budgets where the plans differ,
     so it is not the most favorable point.
   - 24 GB: added at the author's request as the largest budget that fits this GPU with the
     same headroom rule.

No other budget will be measured, and nothing here is selected from live results.

## Step A: RAM calibration (one cell per budget, no comparison)

For each budget B in (12, 16, 24) GB GPU, with 16 GB host, the pilot's RAM-calibration cell is
repeated with only the budget changed:

- Engine placement: `traffic_density`, uniform representation, decoded RAM tier, trace events on.
- Prompts: `json-status` and `code-squares` from the calibration split, the pilot's prompts.
  They are disjoint from the evaluation prompts, which is checked.
- One warm-up token, one output token, cold page cache, fresh process.
- The archived engine contract, and the archived cell gates with the limits below.

The profile is built and validated exactly as in the pilot: `summarize_ram_prepare`, schema 2,
an execution contract over the runtime source and H2D artifact, and
`validate_ram_profile` at budget B.

**Memory limits for every cell at budget B (calibration and live):** logical budget B, allocator
cap B + 2 GB, whole-process admission ceiling B + 4 GB. These are the same offsets as the
8 GB studies (8 / 10 / 12), and they are identical for both arms. The calibration cell records
whether its engine traffic placement has the same tier assignment as the planner's traffic plan
at B. If it does not, the run stops.

## Step B: replanning with measured costs

At each B, the archived planner (checkout `637bbd55`, `h65_planner.py` `a7bb25ab…`) runs with
the archived disk traces and settings. The settings are seed 202609093, 256 iterations,
variability weight 1, fragmentation penalty 0, compressed RAM disabled, 0.5 GB safety margin,
pageable blocking H2D 17.11 GB/s, two training traces and one held-out trace,
`PYTHONHASHSEED=0`. The only differences are budget B and **the RAM profile measured at B**.
The traffic control is the planner's guarded deployment. H6.5 is its placement candidate. Both
are frozen with their hashes.

## Live decision rule (fixed now)

A budget B receives a live comparison only if both hold:

1. Its H6.5 plan differs from its traffic plan in at least 1% of the model's tensors (8 of 723).
   Smaller differences are a near A/A repeat.
2. The plan passes the method's own pre-live replay gate, as in the pilot: no training or
   held-out trace is predicted slower than traffic.

Qualifying budgets run in the order 12, 16, 24, until each finishes or hits its limit. A budget
that does not qualify is reported with its plans and predictions, and is not timed.

## Live comparison (each qualifying budget, separately)

- **Design:** as the plan's step 2. Llama-3.3-70B-Instruct, batch one, greedy, exactly one
  output token, full output head, no draft, exact weights.
- **Blocks:** 8, one fresh prompt each (`budget-prompts.json`; the same 8 prompts at every
  budget), two arms (traffic, H6.5).
- **Order:** T H, H T, T H, H T, T H, H T, T H, H T.
- **Cells:** fresh process per cell, one warm-up token, cold page cache, archived cooldown
  (10 s, at or below 75 °C) and thermal and power gates. Each runtime plan is verified against
  its frozen file and read counters.
- **Limits:** 10-minute cell cap and 2-hour cap per budget, including setup, cooling and
  failures.
- **Stops:** a changed input, a runtime plan mismatch, or a failed cell stops that budget, and
  partial data are kept. A thermal or power window failure, or token mismatch, makes that block
  ineligible. No reruns, replacement blocks, outlier removal or sample extension.

## Analysis (per budget; budgets are never pooled with each other or with D1/D2/P1)

Per eligible paired block, compute the traffic/H6.5 ratio for request latency and for peak
incremental GPU memory. Report the geometric mean with a two-sided 95% Student-t interval on
the block log ratios (df = eligible blocks - 1). Also report absolute GPU peaks, host RSS,
initialization, read bytes and calls, prefetch wait, and thermal and power summaries.

An interval crossing 1 means the direction is unresolved. Memory and latency are reported
separately. Every budget's outcome is reported, including non-qualifying, inconclusive or
reversed results.

## Scope

Regulated secondary evidence on one RTX 5090 desktop. The NVIDIA driver is now 616.92, against
610.88 for D1/D2, so cross-study differences are not attributed to budget alone. Planning
results at non-qualifying budgets are predictions, not measurements.
