# Frozen protocol v2: H6.5 calibrated for 32-token generation (RTX 3080 Laptop, Qwen3-14B)

Frozen 2026-10-03, before any arm below was timed. The run's driver prints this
file's SHA-256 at launch and stores it beside the results, and
`scripts/analyze_h65_32token_followup.py` prints it again with the verdict.

**Revision history.** v1 (commit `6fb3da5`, same date) was launched at 08:33 and
stopped at 08:40 during its first calibration request, before any arm was timed,
for this redesign. v2 drops the `exact-min` arm (it answers nothing here and cost
about a third of the run), adds a zero-RAM traffic control so a win can be
attributed, makes plan B the primary arm, and states the failure, thermal and
drift rules below. No measurement informed these changes.

## Question

The paper's 32-token Qwen test (D8) found the disk control faster than the H6.5
plan, and Section 5.5 says H6.5 "can be re-optimized by calibrating on
representative multi-token requests". Can an H6.5 plan calibrated on 32-token
requests beat the disk control at 32 tokens, and if so, under which host-RAM budget?

## What D8 shows

Past the first token the runs are SSD-bound, so time per token follows the bytes
actually read from the SSD. The disk control keeps nothing resident: its process
uses about 3.8 GB, the operating system's page cache holds much of the 20.3 GB store
in the remaining memory of the 20.7 GB WSL2 VM, and 57% of its 18.77 GB of requested
bytes per token came from that cache (8.05 GB from the SSD, 17.93 s/token). The
one-token H6.5 plan pins 8 GB of decoded weights in host RAM. It requested fewer
bytes (12.22 GB) but its smaller cache served 3% of them, so it read 11.85 GB from
the SSD (22.95 s/token), and its cells were power-limited for 97% of their measured
time against 33% for the disk control.

The H6.5 replay prices a disk-resident tensor at the read time measured in the
all-disk calibration trace. A plan whose own memory use is close to the all-disk
plan's (no host-RAM residency) runs in the cache regime its calibration measured,
so the replay describes it. A plan that takes 8 GB of host RAM does not: the replay
cannot see the cache it displaces. Hence two H6.5 plans from the same traces.

## Arms (all exact; output token ids must match across arms for every request)

| Arm | Plan | VRAM / host-RAM budget | Role |
|---|---|---|---|
| `disk-frozen` | uniform per-tensor disk plan, the D8 control | 4 / 8 GB (uses none) | control |
| `h65-cal32-ram0` | H6.5 full candidate, 32-token calibration | 4 / 0 GB | **primary** |
| `simple-v4-r0` | traffic-density placement, the D8 traffic control with no RAM tier | 4 / 0 GB | attribution |
| `h65-cal32-ram8` | H6.5 full candidate, same traces | 4 / 8 GB | the paper's Section 5.5 claim at the paper's budget |

`exact-min` is not run; the runner pairs every arm against `disk-frozen`.

## Calibration and planning

- Three calibration requests from the `calibration_long` prompt split
  (`calibration-long-explain`, `-code`, `-compare`), 32 forced greedy tokens each, on
  the all-disk plan, cold cache, recorded with scheduler traces. A request that
  generates fewer than 32 tokens, or a trace without exactly 32 forward passes, stops
  the run before anything is timed.
- None of these prompts appears in the evaluation set (`paper_generation`), by test.
- Both H6.5 plans are searched from those same three traces by the planner at the
  run's commit: 128 search iterations, seed 20260909, 0.5 GB VRAM safety margin,
  decode slice 4,194,304, pageable H2D 4.58 GB/s, no RAM-preparation profile. This is
  the current planner, not the 6c37700 planner that built the D6 to D10 plans.
- The H6.5 search and objective are unchanged. Only the calibration length and the
  host-RAM budget differ from D8.
- Gate before timing: both plans built, within budget, each diverged from its own
  traffic control, 32 forward passes per trace. Any failure stops the run.

## Fixed inputs

Qwen3-14B store at `/root/afterimage/store_14b` (the D8 store); the RTX 3080 Laptop
GPU and WSL2 environment of D6 to D10; engine settings as in D8 (decode slice
4,194,304, prefetch depth 2, per-blob reads, no draft model, one warm-up token, 45 s
cooldown to 50 C before each method and request). Evaluation prompts: the four
`paper_generation` prompts of D8. Launch preconditions, checked by the launcher: AC
power, the WSL2 VM at 19 to 22 GB of memory (the page-cache regime depends on it),
Docker's WSL distro stopped (it shares that memory), no other GPU compute process on
the host, a clean git tree.

## Design

32 output tokens, four prompts per block, three blocks. Method order is shuffled
per block with seed 0, a cold page cache precedes every request, and each block and
method runs in one fresh process. Blocks run in sequence, so every finished block
is complete.

## Failure, thermal and drift rules

- A cell that ends with an error has produced no measurement. The driver reruns
  failed cells (and only those) up to two more times with identical settings, after
  the pass in which they failed. A cell that completed is never rerun or replaced.
  Cells still failing after that leave the run incomplete, and it is reported as
  incomplete with the failure recorded.
- Thermal and power-limit flags are recorded for every cell and reported as the
  share of measured time each arm spent power limited. They are not an exclusion
  rule: in D8 every cell, the control included, reported both, because the GPU
  mostly waits on reads. Randomized order inside each block keeps slow drift from
  favouring one arm.
- The control's per-block speed is reported. A spread above 5% between its fastest
  and slowest block marks the run as drifted, and every verdict is reported with that
  caveat. Its mean is also compared with D8's 17.93 s/token.
- Output token ids of each arm must equal `disk-frozen`'s for the same block and
  prompt; an arm with any mismatch is invalid, whatever its speed.

## Predeclared analysis

For each arm against `disk-frozen`: the speed ratio `control wall seconds / arm wall
seconds` per request, its geometric mean over the four prompts per block, then the
geometric mean over blocks with a two-sided 90% t interval on the log of the block
ratios (`scripts/analyze_h65_32token_followup.py`). Only complete blocks count, and at
least 3 complete blocks are required for any verdict.

- The lower bound of the interval is above 1.0: the arm is faster than the control.
- The upper bound is below 1.0: the arm is slower.
- Otherwise: inconclusive.

Attribution: the same estimator with `simple-v4-r0` as the control and
`h65-cal32-ram0` as the arm. Faster: the H6.5 search adds measurable benefit at a
zero RAM budget. Otherwise the benefit, if any, is not separable from VRAM residency.

Reported for each arm, with no verdict attached: per-request wins out of 12, SSD
bytes per token, the share of requested bytes served from the page cache, peak
whole-process VRAM, peak host RSS, and the share of measured time power limited.
Three arms are tested against one control with no multiplicity correction; each
verdict stands alone, and the primary arm is named in advance.

## Scope

Exploratory (regulated, L2) evidence on one laptop for Qwen3-14B at this memory
contract, one output length and a VM whose memory is about the store's size. Three
blocks give a wide interval: an effect of a few percent will not clear it. A faster
primary arm shows that H6.5, given a host-RAM budget that leaves the page cache
alone, beats the disk control here. It does not show the planner would choose that
budget unaided: its replay cannot see the page cache.
