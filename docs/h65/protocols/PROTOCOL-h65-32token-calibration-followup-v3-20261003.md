# Frozen protocol v3: H6.5 calibration length and host residency at 32 tokens (RTX 3080 Laptop, Qwen3-14B)

Frozen 2026-10-03, before any arm below was timed. The run's driver prints this
file's SHA-256 at launch and stores it beside the results, and
`scripts/analyze_h65_32token_followup.py` prints it again with the verdicts.

**Revision history.** v1 (commit `6fb3da5`) was launched at 08:33 and stopped at
08:40 during its first calibration request, before any arm was timed. v2 (commit
`40d3350`) was never launched. v3 adds one-token calibration arms from the same
planner version, so calibration length is isolated; gives the traffic control the
planner's 0.5 GB VRAM reserve, so resident capacity matches; uses 95% intervals, as
the paper does; and reports each plan's predicted against measured time. No
measurement informed these changes.

## Questions

1. Does an H6.5 plan beat the disk control at 32 tokens? (primary: `h65-cal32-ram0`)
2. Does 32-token calibration help, at 8 GB and at 0 GB of host residency?
3. Does removing host residency help, at each calibration length?
4. Does the best H6.5 plan beat traffic placement with the same resident capacity?

This run is an exploratory screen. A positive primary result is to be confirmed by a
separate frozen run against `disk-frozen` and `simple-v4-r0` only, with its block
count fixed in advance from this run's variability.

## Why

In D8 the disk control (17.93 s/token) beat the one-token H6.5 plan (22.95 s/token)
and the traffic control (26.02 s/token). Linux-accounted storage reads inside the
WSL2 VM were 8.05 GB per token for the disk control, against 11.85 GB for H6.5,
which requested fewer bytes (12.22 GB against 18.77 GB); the page cache served 57%
of the disk control's requests and 3% of H6.5's. That is consistent with 8 GB of
explicit host residency (7.99 GB of decoded weights replacing 5.36 GB of encoded
reads, 7.16 GB of it pageable) displacing useful file cache, though storage volume
alone does not explain every difference: the traffic control read less than H6.5 and
was slower. H6.5 itself beat the traffic control by about 13% in D8. Its replay keeps
the read durations recorded for tensors left on disk, so it cannot see a cache
change caused by its own residency; longer calibration exposes repeated costs but
does not remove that limitation.

## Arms (all exact; output token ids must match `disk-frozen` for every request)

| Arm | Calibration | Host-RAM budget | Role |
|---|---|---|---|
| `disk-frozen` | none | 8 GB declared, 0 used | control (the D8 control) |
| `h65-cal1-ram8` | 1 token | 8 GB | D8's design on the current planner |
| `h65-cal32-ram8` | 32 tokens | 8 GB | the paper's Section 5.5 remedy |
| `h65-cal1-ram0` | 1 token | 0 GB | |
| `h65-cal32-ram0` | 32 tokens | 0 GB | **primary** |
| `simple-v4-r0` | none | 0 GB | traffic placement, 0.5 GB VRAM reserve as the planner |

All arms use a 4 GB VRAM budget. `exact-min` is not run; the runner pairs every arm
against `disk-frozen`.

## Calibration and planning

- Both calibration sets use the same three prompts from the `calibration_long` split
  (`calibration-long-explain`, `-code`, `-compare`), recorded on the all-disk plan with
  a cold cache: once at 32 tokens and once at 1 token. A request that generates fewer
  tokens than asked, or a trace whose forward passes do not equal its token count,
  stops the run before anything is timed.
- None of these prompts appears in the evaluation set (`paper_generation`), by test.
- Each set yields two plans (8 GB and 0 GB host RAM) by the planner at the run's
  commit, with identical settings: 128 search iterations, seed 20260909, 0.5 GB VRAM
  reserve, decode slice 4,194,304, pageable H2D 4.58 GB/s, no RAM-preparation
  profile, two training traces and one held-out trace. H6.5's search and objective
  are unchanged.
- Gate before timing: all four plans built and within budget, trace lengths as above,
  calibration and evaluation prompts disjoint; any failure stops the run. Whether a
  plan diverged from its own traffic control is recorded, not a stop condition.

## Fixed inputs

Qwen3-14B store at `/root/afterimage/store_14b` (the D8 store); the RTX 3080 Laptop
GPU and WSL2 environment of D6 to D10; engine settings as in D8 (decode slice
4,194,304, prefetch depth 2, per-blob reads, no draft model, one warm-up token, 45 s
cooldown to 50 C before each method and request). Evaluation prompts: the four
`paper_generation` prompts of D8. Launch preconditions, checked by the launcher: AC
power, the WSL2 VM at 19 to 22 GB of memory, Docker's WSL distro stopped, no local LLM
server, no other GPU-heavy process on the host, a clean git tree.

## Design

32 output tokens, four prompts per block, three blocks, six arms (72 requests). Method
order is shuffled per block with seed 0, a cold page cache precedes every request
(normal caching within it), and each block and method runs in one fresh process.
Blocks run in sequence, so every finished block is complete.

## Failure, thermal and drift rules

- A cell that ends with an error has produced no measurement. The driver reruns
  failed cells (and only those) up to two more times with identical settings, after
  the pass in which they failed. A cell that completed is never rerun or replaced.
  Cells still failing after that leave the run incomplete, reported as such.
- NVIDIA's idle, power-cap and thermal-slowdown signals are distinct. Each cell's
  power-cap and thermal-slowdown exposure is recorded and reported as the share of
  measured time per arm (D8: 97% power-capped for H6.5, 33% for the disk control).
  They are reported and examined, not used to exclude data.
- The control's per-block speed is reported. A spread above 5% between its fastest
  and slowest block marks the run as drifted, and every verdict carries that caveat.
  Its mean is also compared with D8's 17.93 s/token.
- An arm whose output token ids differ from `disk-frozen`'s is invalid, whatever its
  speed.

## Predeclared analysis

For each comparison: the speed ratio `control wall seconds / arm wall seconds` per
request, its geometric mean over the four prompts per block, then the geometric mean
over blocks with a two-sided 95% t interval on the log of the block ratios
(`scripts/analyze_h65_32token_followup.py`). Only complete blocks count, and at least
3 complete blocks are required for any verdict. The lower bound of the interval
above 1.0: the arm is faster than that control; the upper bound below 1.0: slower;
otherwise inconclusive.

- Primary: `h65-cal32-ram0` against `disk-frozen`.
- Every other arm against `disk-frozen`.
- Calibration length: `h65-cal32-ram8` against `h65-cal1-ram8`, and `h65-cal32-ram0`
  against `h65-cal1-ram0`.
- Host residency: `h65-cal32-ram0` against `h65-cal32-ram8`, and `h65-cal1-ram0`
  against `h65-cal1-ram8`.
- Attribution: `h65-cal32-ram0` and `h65-cal1-ram0` against `simple-v4-r0`.

Also reported, without verdicts: per-request wins, Linux-accounted storage reads per
token, the share of requested bytes served from the page cache, peak whole-process
VRAM, peak host RSS, power-cap share, and each plan's replay-predicted time per token
next to its measured time per token (a plan that wins with a poor prediction still
shows the model's limitation). With three blocks the intervals are wide; no number is
predicted in advance, and blocks are not added after looking at results.

## Scope

Exploratory (regulated, L2) evidence on one laptop for Qwen3-14B at this memory
contract, one output length and a VM whose memory is about the store's size. Storage
reads are Linux accounting inside WSL2, not measured physical SSD reads.
