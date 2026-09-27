# RTX 5090 Llama H6.5 endpoint-plan confirmation 2 (fresh prompts)

Frozen on 2026-09-11, before any confirmation-2 measurement. This is the second of
the two pooled confirmations behind the paper's headline latency claim; the first is
[PAPER1_5090_LLAMA_H65_ENDPOINT_CONFIRMATION.md](PAPER1_5090_LLAMA_H65_ENDPOINT_CONFIRMATION.md).

## Why a second confirmation

The first frozen confirmation (eight pairs, 2026-09-10) measured traffic/H6.5-endpoint
latency 1.018x with 95% interval [0.884, 1.173]: parity. A later secondary mechanism
control measured 1.237x, interval [1.048, 1.460], on reused confirmation prompts. This
test resolves that disagreement on prompts neither earlier run has seen, with 50% more
pairs.

## Fixed question and methods

Same causal placement claim as the first confirmation: does the automatically
discovered, schedule-aware H6.5 plan reduce cold one-token latency relative to
traffic-density placement on Llama-3.3-70B and this RTX 5090 machine?

The two immutable plans are the same ones frozen for the first confirmation:

- traffic plan SHA-256:
  `926b0adb51275de35473a76ac3fc7dc7d82c58d3669027118f5bd433bb585563`;
- H6.5 endpoint plan SHA-256:
  `a31a0d4bbe2309998514c70182a04bce2024c2a385dc99bfd1c0585f0aa3ebb4`;
  frozen tier fingerprint
  `10079cfa340912b45b27f7be200402db2d47e1bb96d3ddf3948b91ee25b58cac`.

No calibration, optimization, guard replacement, or plan editing occurs in this
confirmation. Execution commit `637bbd55`; the only change since the first
confirmation's commit `c89af234` is one documentation file, so the engine code is
identical between the two confirmations.

## Fixed workload and limits

Identical to the first confirmation: 8.0 decimal GB VRAM and 16.0 decimal GB RAM
logical budget plus the frozen 0.5 GB planner safety allowance, a 10.0 decimal GB
PyTorch caching-allocator cap, a 12.0 decimal GB whole-process NVIDIA-SMI admission
ceiling, one untimed warm-up token followed by one timed token, a fresh process and
successful cold page-cache drop for every method cell, decoder-table reuse,
4,194,304-element slices, prefetch depth two, per-blob reads, and exact full output
head.

## Independent blocks

Twelve prompts written for this test, matched to the first confirmation's prompt
families and lengths, and disjoint from every earlier prompt split:
`confirm2-explain-seasons`, `confirm2-arithmetic-tickets`, `confirm2-code-chunks`,
`confirm2-compare-queues`, `confirm2-summarize-soil`, `confirm2-reasoning-couriers`,
`confirm2-retrieval-rivers`, `confirm2-procedure-migration`, `confirm2-explain-rainbows`,
`confirm2-code-palindrome`, `confirm2-compare-caching`, `confirm2-arithmetic-garden`.
Twelve blocks, one prompt each, in the order listed; `traffic` runs first in even
blocks and `h65-endpoint` first in odd blocks (24 cells total).

## Outcomes and validity

The primary estimand is the mean of the twelve paired log latency ratios, reported as
geometric `traffic_seconds / h65_seconds` speedup with a two-sided 95% paired
Student-t interval (df 11). The pooled estimand combines all 20 pairs from both
confirmations with a paired-t interval at df 19.

A "faster than traffic" latency claim is made only if the pooled interval lies above
1.0 **and** confirmation 2 alone has a geometric mean above 1.0; otherwise the paper
reports latency parity. Both confirmations and the pooled estimate are reported
whatever the outcome. The separate secondary mechanism control's latency result
(reused confirmation prompts) is not pooled into this dataset.

All 24 cells must finish in the fixed order with exact paired output token IDs,
matching token counts, successful cache drops and warm-ups, observable non-throttled
thermal/power windows, exact frozen plan hashes and runtime tier fingerprints, logical
budget compliance, and no whole-process peak above 12.0 GB. There is no automatic
retry, outlier deletion, prompt substitution, optional stopping, or limit change. A
failed gate remains visible and makes the dataset non-confirmatory.

This experiment confirms only bounded cold one-token latency. Multi-token throughput
and external-framework performance require separate validation and must not be
inferred from this result. See
[../evidence/h65-paper1/README.md](../evidence/h65-paper1/README.md) for the frozen
artifact this protocol produced and a script that recomputes the paper's numbers from
it.
