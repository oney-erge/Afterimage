# Frozen protocol: H6.5 calibrated for 32-token generation (RTX 3080 Laptop, Qwen3-14B)

Frozen 2026-10-03, before any measurement of the arms below. The SHA-256 of this
file is printed by `scripts/analyze_h65_32token_followup.py` and recorded beside
the run's results.

## Question

The paper's 32-token Qwen test (D8) found the disk control faster than the H6.5
plan, and Section 5.5 says H6.5 "can be re-optimized by calibrating on
representative multi-token requests". Does an H6.5 plan calibrated on 32-token
requests beat the disk control at 32 tokens?

## Why the answer is not obvious

D8's per-request rows show where the time went. The disk control read 8.05 GB per
token from the SSD, because the operating system's page cache served 57% of its
18.77 GB of requested bytes. The one-token H6.5 plan requested fewer bytes (12.22 GB)
but read 11.85 GB from the SSD: its 8 GB of decoded host-RAM weights leave the page
cache too little room. Past the first token both runs are SSD-bound, so time per
token follows SSD bytes. Calibrating on 32-token traces gives the planner the
repeated schedule, but its replay keeps the read durations it measured on the
all-disk plan and cannot see page-cache displacement. Plan A tests whether
calibration alone is enough. Plan B removes the displacement.

## Arms (all exact; output token ids must match across arms for every request)

| Arm | Plan | Host-RAM budget |
|---|---|---|
| `exact-min` | the harness's minimum-memory control (required by the runner; reference only) | n/a |
| `disk-frozen` | uniform per-tensor disk plan, the same control as D8 | 8 GB declared, 0 used |
| `h65-cal32-ram8` | H6.5 full candidate searched on the 32-token calibration traces | 8 GB |
| `h65-cal32-ram0` | H6.5 full candidate searched on the same traces | 0 GB |

Both H6.5 plans use a 4 GB VRAM budget, the same as `disk-frozen`.

## Calibration and planning

- Three calibration requests from the `calibration_long` prompt split
  (`calibration-long-explain`, `-code`, `-compare`), 32 forced greedy tokens each, on the
  all-disk plan, cold cache, recorded with scheduler traces. Each trace must contain
  exactly 32 forward passes and each request must generate exactly 32 tokens, or the
  run stops.
- None of these prompts appears in the evaluation set (`paper_generation`), by test.
- Both plans are searched from the same three traces with the planner at the commit
  recorded in the plan artifact: 128 search iterations, seed 20260909, 0.5 GB VRAM
  safety margin, decode slice 4,194,304, pageable H2D 4.58 GB/s, no RAM-preparation
  profile. The planner is the current one, not the 6c37700 planner that built the
  D6 to D10 plans; this is a deliberate difference and is reported with the result.
- The H6.5 search is unchanged. Only the calibration length and the RAM budget differ
  from D8.

## Fixed inputs

Qwen3-14B store at `/root/afterimage/store_14b` (the D8 store); the RTX 3080 Laptop
GPU and WSL2 environment of D6 to D10; engine settings as in the D8 driver (decode
slice 4,194,304, prefetch depth 2, per-blob reads, draft model none, one warm-up token,
45 s cooldown to 50 C). Evaluation prompts: the four `paper_generation` prompts of D8.

## Design

32 output tokens, four prompts per block, three randomized blocks (method order
shuffled per block, cold page cache before every request, one fresh process per
block and method). Blocks run in sequence, so every block that finishes is complete.

## Gates

- Every request in a block that is analysed completed with 32 output tokens and a
  successful cache drop.
- Output token ids of each H6.5 arm equal `disk-frozen`'s for the same block and prompt.
  An arm with any mismatch is reported as invalid, whatever its speed.
- Throttle and power-limit flags are recorded and reported per arm. No request is
  excluded for them, as in D6 to D10.
- No retries, no replaced requests, no change to the arms after the first measurement.
  A run stopped early is resumed with identical settings or reported incomplete.

## Predeclared analysis

Primary: for each H6.5 arm, the speed ratio `disk-frozen wall seconds / arm wall
seconds` per request, its geometric mean over the four prompts per block, then the
geometric mean over blocks with a two-sided 90% t interval on the log of the block
ratios (`scripts/analyze_h65_32token_followup.py`). Only complete blocks count, and at
least 3 complete blocks are required for any verdict.

- The lower bound of the interval is above 1.0: the arm is faster than the disk control.
- The upper bound is below 1.0: the arm is slower.
- Otherwise: inconclusive.

Reported for each arm, with no verdict attached: per-request wins out of 12, SSD
bytes per token, share of requested bytes served from the page cache, peak
whole-process VRAM, peak host RSS, and throttled requests. Two arms are tested
against one control with no multiplicity correction; the verdicts are separate.

## Scope

Exploratory (regulated, L2) evidence on one laptop for Qwen3-14B at this memory
contract and one output length. Three blocks give a wide interval: an effect of a few
percent will not clear it. A faster arm shows that a placement exists that beats the
disk control here, not that the planner finds it without help from the RAM budget.
