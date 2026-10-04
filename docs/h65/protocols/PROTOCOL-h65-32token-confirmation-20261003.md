# Frozen protocol: confirming the 32-token H6.5 result on new questions (RTX 3080 Laptop, Qwen3-14B)

Frozen 2026-10-03, during the v3 screen and before any confirmation measurement,
without access to the screen's third block. The driver prints this file's SHA-256
at launch and stores it beside the results.

## Question

Does the primary H6.5 plan of the v3 screen
(`PROTOCOL-h65-32token-calibration-followup-v3-20261003.md`), unchanged, beat the disk
control again on four new questions, and does it beat traffic placement with the same
resident capacity?

## Launch condition

The confirmation starts only if the v3 run ends COMPLETE with its primary verdict,
`h65-cal32-ram0` against `disk-frozen`, "faster than the control". Otherwise it is
not launched, and the reason is recorded.

## Arms (all exact; output token ids must match `disk-frozen` for every request)

| Arm | What runs |
|---|---|
| `disk-frozen` | the v3 run's uniform disk plan file, unchanged (control) |
| `h65-cal32-ram0` | the v3 run's primary plan file, unchanged: no recalibration, no new search |
| `simple-v4-r0` | traffic placement at 4 GB VRAM, 0 GB RAM, 0.5 GB VRAM reserve |

Both plan files are copied from the v3 run, and their SHA-256 hashes are printed and
stored before the first measurement.

## Prompts

The four `paper_generation_confirm` prompts (`confirm32-explain-refrigerator`,
`confirm32-summarize-waggle-dance`, `confirm32-code-word-frequencies`,
`confirm32-analyze-ssd-vs-hdd`): explanation, summarization, code and analytical,
the same buckets as the screen's prompts, with no id, text or topic shared with any
other prompt split, by test. Every arm receives the same prompts. 32 output tokens
each.

## Design

Same machine, store, engine settings, memory contract and cache regime as the v3 run
(cold page cache before every request, normal caching within it, one warm-up token,
45 s cooldown to 50 C, one fresh process per block and method). Method order comes
from seed 6, whose first six blocks are the six permutations of the three arms (each
arm twice in each position), with blocks 7 and 8 keeping every arm-position count at
most three.

**Block count, fixed once before the first confirmation measurement.** From the
completed v3 run, take the per-block ratio `simple-v4-r0 wall seconds /
h65-cal32-ram0 wall seconds` (geometric mean over its four prompts) for its three
blocks, and the standard deviation `s` of their logs. The count is the smallest `n`
of 6, 7 or 8 with `t(0.975, n-1) * s / sqrt(n) <= 0.03`; if none qualifies, 8. The
driver computes it, prints the inputs and the result, and stores them before timing
anything. A three-block `s` is itself uncertain; this is a planning rule, not a power
guarantee. Blocks are never added after looking at confirmation results.

## Failure, thermal and drift rules

As in v3: failed cells (no measurement) are rerun up to two more times; a completed
cell is never rerun; power and thermal signals are recorded and reported, not used to
exclude data; a control spread above 5% between blocks marks the run as drifted; an
arm with any token mismatch is invalid. The run's per-process storage counter is
unavailable on this WSL kernel and is reported as unavailable; guest disk reads come
from a system-wide `/proc/diskstats` sampler, labelled as such.

## Predeclared analysis (fixed order)

Per comparison: the speed ratio `control wall seconds / arm wall seconds` per request,
its geometric mean over the four prompts per block, then the geometric mean over
blocks with a two-sided 95% t interval on the log of the block ratios
(`scripts/analyze_h65_32token_followup.py`).

1. Primary: `h65-cal32-ram0` against `disk-frozen`. The lower bound above 1.0
   confirms that H6.5 beats the disk control on new questions.
2. Only if 1 is confirmed: `h65-cal32-ram0` against `simple-v4-r0`. The lower bound
   above 1.0 confirms that it also beats traffic placement.

Both estimates are reported whatever the outcome. "Beats both" is claimed only if
both are confirmed. An interval that includes 1.0 means the advantage is not
established. The v3 screen's data are not pooled into these intervals.

## Scope

Regulated exploratory evidence (L2) on one laptop for Qwen3-14B at this memory
contract and output length, in a WSL2 VM whose memory is about the store's size.
Storage figures are guest disk reads, not physical SSD reads.
