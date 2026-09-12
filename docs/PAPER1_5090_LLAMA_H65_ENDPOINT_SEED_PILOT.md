# RTX 5090 Llama H6.5 endpoint-seed viability pilot

Status: frozen before fresh calibration, candidate selection, or live timing.
This is a regulated development pilot, not primary confirmation.

## Motivation fixed before measurement

The corrected V3 pilot completed all four exact paired blocks but produced a
0.9437x geometric speedup and two wins. Its selected H6.5 plan was close to
traffic placement and saved only about 0.1% of disk bytes. One H6.5 cell also
had a large prefetch stall; that result remains in V3 and is not discarded.

An offline search audit then identified a deterministic search-coverage bug.
Traffic, all-disk, and density-greedy seeds could not reach the feasible region
where the schedule-critical output head is GPU-resident because promoting that
large tensor requires several existing residents to be evicted in the same
proposal. Supplying an old head-resident plan merely as a seed allowed the
corrected objective to find a candidate with positive predicted improvement on
all three traces. The code fix does not import that historical plan. It now
generates a generic candidate by promoting `lm_head.weight` from the traffic
incumbent and evicting the lowest-density VRAM residents until feasible, before
running the ordinary search and all existing safety gates.

## Fixed memory and runtime configuration

- Model: `meta-llama/Llama-3.3-70B-Instruct`, exact existing store.
- Logical residency-planning budget: 8.0 decimal GB.
- PyTorch caching-allocator cap: 10.0 decimal GB.
- Whole-process NVIDIA-SMI admission ceiling: 12.0 decimal GB.
- Host RAM placement budget: 16.0 decimal GB.
- Additional planner safety allowance: 0.5 decimal GB.
- Decoder-table reuse enabled; 4,194,304-element decode slices.
- Fixed prefetch depth 2; per-blob storage reads; exact full output head.
- Batch size one, no speculation, one untimed warm-up token, one timed token.
- Cold page/allocator cache per timed request and clean thermal windows.
- Frozen pageable/blocking RTX 5090 H2D artifact.
- Search seed `202609093`, 256 iterations.
- Six-minute cell timeout and 55-minute total timeout.

Both methods receive identical limits. Every table must state the 8 GB logical
residency budget and report measured whole-process VRAM separately.

## Fresh calibration and selection

Fresh disk traces are collected in separate processes on:

1. `summary-photosynthesis`
2. `logic-glippets`
3. `copy-nonce`

Fresh decoded-RAM preparation profiling uses `json-status` and `code-squares`.
The first two disk traces select the candidate; the third is held out and may
only reject it. Placement-only H6.5 must have nonnegative predicted improvement
on every trace. Full H6.5 is searched offline from the placement incumbent and
is an ablation, not another live arm.

Stop before live timing if the new candidate is identical to traffic, does not
keep the output head in decoded VRAM, violates logical budgets, fails replay,
or if full search returns a worse offline objective than placement-only. Plans,
profiles, traces, source, configuration, and protocol hashes are frozen before
evaluation.

## Untouched live prompts and fixed order

V3 used the four short evaluation prompts. This pilot instead freezes the
previously untouched `paper_generation` cases so the endpoint-seed change is
not evaluated only on prompts already inspected:

| Block | Prompt | Adjacent order |
|---|---|---|
| 0 | `explain-bicycle-balance` | traffic, H6.5 |
| 1 | `summarize-coral-bleaching` | H6.5, traffic |
| 2 | `code-binary-search` | traffic, H6.5 |
| 3 | `analyze-hashmap-vs-btree` | H6.5, traffic |

All eight fresh-process cells require zero exits, exact paired output token
IDs, successful cache drops, a completed warm-up, clean observable thermal and
power windows, exact frozen plan/tier fingerprints, expected read counters,
allocator enforcement, and measured whole-process VRAM no greater than 12 GB.
No automatic retry, optional stopping, prompt replacement, outlier deletion,
or limit change is allowed.

Primary estimand: geometric mean of the four paired ratios
`traffic_seconds / h65_seconds`, with every ratio and the win count. Advance to
a separately frozen independent confirmation only if all gates pass, geometric
speedup is at least 1.03x, and H6.5 wins at least three blocks. Report
separately whether every block improves by the frozen 1% deployment threshold.

No external baseline, other model, budget sweep, multi-token run, or
confirmation is automatically included in this pilot.
