# Frozen protocol: H6.5 endpoint confirmation 2 (fresh prompts)

Frozen 2026-09-11, before any confirmation-2 measurement. The SHA-256 of this file is
recorded in the run's `status.json` at prepare time.

## Why

The first frozen confirmation (eight pairs, 2026-09-10) measured traffic / H6.5-endpoint
latency 1.018x with 95% interval [0.884, 1.173]: parity. The later secondary mechanism
control measured 1.237x, interval [1.048, 1.460], on reused confirmation prompts. This
test resolves that disagreement on prompts neither earlier run has seen, with 50% more
pairs.

## Arms, inputs and settings

Identical to the first confirmation:

- Frozen `per_tensor` plans: `traffic` sha256 `926b0adb…`; `h65-endpoint` sha256 `a31a0d4b…`, tier fingerprint `10079cfa…`.
- Memory contract: 8 GB logical VRAM budget, 10 GB allocator cap, 12 GB whole-process ceiling, 16 GB RAM, 0.5 GB safety margin.
- Engine: decode-table reuse on, decode slice 4,194,304, prefetch depth 2, per-blob reads, full LM head, static execution, no draft model, decoded RAM tier.
- One warm-up token, 10 s cooldown, one-token cold-cache request.
- Exec commit `637bbd55`. The only change since the first confirmation's commit `c89af234` is one documentation file, so the engine code is identical.

## Prompts

Twelve new prompts in `h65-confirmation2-prompts-20260911.json`, written for this test
and matched to the first confirmation's prompt families and lengths:
`confirm2-explain-seasons`, `confirm2-arithmetic-tickets`, `confirm2-code-chunks`,
`confirm2-compare-queues`, `confirm2-summarize-soil`, `confirm2-reasoning-couriers`,
`confirm2-retrieval-rivers`, `confirm2-procedure-migration`, `confirm2-explain-rainbows`,
`confirm2-code-palindrome`, `confirm2-compare-caching`, `confirm2-arithmetic-garden`.
Prepare verifies that none shares an ID or text with any existing prompt split and that
all render with the model's chat template. They reach the frozen worker through
`h65_worker_confirmation2_20260911.py`, which loads that worker unchanged.

## Design

Twelve blocks, one prompt each, in the order listed. `traffic` runs first in even blocks
and `h65-endpoint` first in odd blocks; 24 cells.

## Gates

Any failure stops the run. No retries, exclusions or extensions.

- All 24 cells complete, token-identical output in every pair.
- Every cell passes its measurement gates: cold page cache, thermal and power windows, resolved engine settings, one-token warm-up.
- Whole-process peak VRAM at or below 12 GB in every row.
- Frozen inputs unchanged; runtime tiers match the frozen plans.

## Predeclared analysis and decision rule

- Confirmation 2 alone: geometric-mean speedup (`traffic_s / endpoint_s`) with a 95% paired-t interval (df 11), and the paired peak-VRAM ratio.
- Pooled: all 20 pairs from both confirmations, one geometric mean with a 95% paired-t interval (df 19).
- A "faster than traffic" latency claim is made only if the pooled interval lies above 1.0 **and** confirmation 2 alone has a geometric mean above 1.0. Otherwise the paper reports latency parity.
- Both confirmations and the pooled estimate are reported whatever the outcome. The mechanism control's latency result is not pooled: it reused prompts and had a different design.

## Scope

Cold one-token latency and peak VRAM for Llama-3.3-70B on this RTX 5090 at this memory
contract.
