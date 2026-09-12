# RTX 5090 Llama H6.5 endpoint-plan confirmation protocol

Status: frozen on 2026-09-10 before endpoint-pilot blocks 1--3 completed.
The eight reserved confirmation prompts were committed before any endpoint
pilot live result was available. The confirmation is launched only if the
pilot's already-frozen advancement rule passes, but its result is reported
regardless of direction and cannot extend its own sample size.

## Fixed question and methods

This experiment tests the causal placement claim only: does the automatically
discovered, schedule-aware H6.5 plan reduce cold one-token latency relative to
traffic-density placement on Llama-3.3-70B and this RTX 5090 machine?

The two immutable plans were produced from fresh calibration before any pilot
evaluation cell:

- traffic plan SHA-256:
  `926b0adb51275de35473a76ac3fc7dc7d82c58d3669027118f5bd433bb585563`;
- H6.5 endpoint plan SHA-256:
  `a31a0d4bbe2309998514c70182a04bce2024c2a385dc99bfd1c0585f0aa3ebb4`.

The H6.5 plan must retain `lm_head.weight` as `decoded_vram`; its frozen tier
fingerprint is
`10079cfa340912b45b27f7be200402db2d47e1bb96d3ddf3948b91ee25b58cac`.
No calibration, optimization, guard replacement, or plan editing occurs in
this confirmation.

## Fixed workload and limits

- Model: `meta-llama/Llama-3.3-70B-Instruct`, exact existing store.
- Store manifest SHA-256:
  `abb500ea82e7b979f20a32ffd466de639e31773678091ad823586c5a6798ac92`.
- H2D artifact SHA-256:
  `ef310198e36140dd1f315a09ce6bee287ee2c9ed029b08fe774814b6000e4a0b`.
- Logical planning budget: 8.0 decimal GB VRAM and 16.0 decimal GB RAM,
  plus the frozen 0.5 GB planner safety allowance.
- PyTorch caching-allocator cap: 10.0 decimal GB.
- Whole-process NVIDIA-SMI admission ceiling: 12.0 decimal GB.
- One untimed warm-up token followed by one timed token.
- Fresh process and successful cold page-cache drop for every method cell.
- Decoder-table reuse, 4,194,304-element slices, prefetch depth two,
  per-blob reads, and exact full output head.
- Six-minute hard cap per cell and 60-minute total controller cap.

## Independent blocks

The `h65_confirmation` prompt split contains eight cases that were added and
frozen before the endpoint pilot produced a live result. Each case forms one
adjacent traffic/H6.5 pair. Pair order alternates and is balanced four/four:

| Block | Case | Order |
|---:|---|---|
| 0 | `confirm-explain-tides` | traffic, H6.5 |
| 1 | `confirm-arithmetic-boxes` | H6.5, traffic |
| 2 | `confirm-code-deduplicate` | traffic, H6.5 |
| 3 | `confirm-compare-indexes` | H6.5, traffic |
| 4 | `confirm-summarize-wetlands` | traffic, H6.5 |
| 5 | `confirm-reasoning-switches` | H6.5, traffic |
| 6 | `confirm-retrieval-quartz` | traffic, H6.5 |
| 7 | `confirm-procedure-firewall` | H6.5, traffic |

## Outcomes and validity

The primary estimand is the mean of the eight paired log latency ratios,
reported as geometric `traffic_seconds / h65_seconds` speedup with a two-sided
95% paired Student-t interval. Report every pair, the win count, median paired
latency reduction, and both methods' whole-process peak VRAM. A confidence
interval wholly above 1.0 supports the positive latency claim; any other
result remains the fixed confirmation result.

All 16 cells must finish in the fixed order with exact paired output token
IDs, matching token counts, successful cache drops and warm-ups, observable
non-throttled thermal/power windows, exact frozen plan hashes and runtime tier
fingerprints, logical budget compliance, and no whole-process peak above
12.0 GB. There is no automatic retry, outlier deletion, prompt substitution,
optional stopping, or limit change. A failed gate remains visible and makes
the dataset non-confirmatory.

This experiment confirms only bounded cold one-token latency. Multi-token
throughput and external-framework performance require separate validation and
must not be inferred from this result.
