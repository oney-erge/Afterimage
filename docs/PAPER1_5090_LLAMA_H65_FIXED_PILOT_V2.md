# RTX 5090 Llama H6.5 corrected-placement viability pilot v2

Status: frozen before v2 calibration or live timing. This supersedes the
infeasible 9/16 protocol for future measurements but does not modify or erase
that failed attempt. This is a regulated pilot, not primary confirmation.

## Correction motivated by the failed preflight

The first protocol incorrectly equated the planner's logical GPU budget with a
whole-process physical device-memory ceiling. At 9 GB logical/physical, the
traffic RAM-profile cell reached 8.3883 GB in the PyTorch allocator but 9.4854
decimal GB as a whole-process NVIDIA-SMI delta. It therefore stopped before
candidate selection, exactly as required.

This v2 protocol separates the two quantities before collecting any v2 timing:

- **8.0 decimal GB logical planner budget**, including the planner's transient
  reserve and 0.5 GB additional safety allowance. This recreates the placement
  pressure of the historical positive setting.
- **10.0 decimal GB physical process ceiling**, enforced by runtime cap and
  sampled whole-process device-memory delta. This accommodates CUDA/runtime
  memory that is not represented by persistent tensor choices.

Both traffic and H6.5 use identical limits. The comparison is not a claim that
the complete process occupies only 8 GB. Both logical and physical measurements
are reported. If either arm exceeds 10 GB, v2 stops; neither limit changes.

## Fixed configuration

- Model: `meta-llama/Llama-3.3-70B-Instruct`, existing exact store and manifest.
- Logical GPU/RAM placement budgets: 8.0/16.0 decimal GB.
- Physical whole-process GPU ceiling: 10.0 decimal GB.
- Additional planner safety allowance: 0.5 decimal GB.
- Decoder-table reuse enabled; 4,194,304-element decode slices.
- Fixed prefetch depth 2 and per-blob storage reads.
- Exact full output head, batch size one, no speculation.
- One untimed warm-up token and one timed output token in every fresh process.
- Cold page/allocator cache per timed request.
- Frozen pageable/blocking RTX 5090 H2D artifact.
- Six-minute cell timeout and 50-minute total timeout.

## Calibration and selection

Fresh causal disk traces, in separate processes:

1. `summary-photosynthesis`
2. `logic-glippets`
3. `copy-nonce`

Fresh decoded-RAM profile: `json-status` and `code-squares`. These are disjoint
from live evaluation. The first two disk traces select the candidate; the third
is held out and may only accept or reject it.

Corrected placement-only search uses seed `202609103`, 256 iterations, and
requires nonnegative predicted improvement on every training and held-out
calibration trace. Full search is checked offline with placement's selected
candidate as an incumbent; it is not another live arm. Plans, traces, profile,
source and configuration hashes are frozen before evaluation.

Stop before live timing if placement is physically identical to traffic, fails
replay, violates either logical budget, or full returns a worse offline objective
than its placement incumbent.

## Live paired pilot

| Block | Prompt | Adjacent order |
|---|---|---|
| 0 | `fact-gold` | traffic, H6.5 |
| 1 | `arithmetic-17x6` | H6.5, traffic |
| 2 | `code-square` | traffic, H6.5 |
| 3 | `retrieval-7319` | H6.5, traffic |

All eight cells require captured zero exits, exact paired token IDs, successful
cache drops, a completed warm-up, clean observable thermal windows, exact frozen
plan/tier fingerprints, expected read counters, and valid whole-process memory
no greater than 10 GB. No automatic retry, optional stopping, prompt replacement,
outlier removal or limit change is permitted. Failures remain visible.

Primary pilot estimand: geometric mean of the four paired block ratios
`traffic_seconds / h65_seconds`, accompanied by all ratios and win count.

Advance to a separately frozen independent confirmation only if all gates pass,
geometric speedup is at least 1.03x, and H6.5 wins at least three of four blocks.
Report the stricter deployment gate separately: every eligible live block must
improve by at least the planner's frozen 1% threshold.

No external framework, additional model, broader budget sweep, multi-token run
or confirmation is automatically launched by this protocol.
