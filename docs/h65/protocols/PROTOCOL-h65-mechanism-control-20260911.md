# Frozen protocol: H6.5 mechanism control (greedy profiled knapsack)

Frozen 2026-09-11, before any mechanism-control measurement. The SHA-256 of this
file is recorded in the run's `status.json` at prepare time.

## Question

Does the engine's simple cost-ranked placement (`placement_policy="profiled_knapsack"`),
given the same measured disk-calibration costs the H6.5 planner used, reproduce the
H6.5 endpoint plan's latency and peak-VRAM behaviour on Llama-3.3-70B at the
confirmation's memory contract?

## Why it is needed

The endpoint plan's measured benefit comes from making the output head resident and
demoting early-layer tensors to disk. A referee's first question is whether a greedy
rule produces the same result. The September 9 attempt
(`paperB-greedy-control-8v16r-20260909-v1`) failed on its first knapsack cell because
no measured profile was supplied, so no such comparison exists.

## Arms (all exact; output tokens must match across arms in every block)

| Arm | Placement policy | Representation |
|---|---|---|
| `traffic` | `traffic_density` | frozen `per_tensor` plan, sha256 `926b0adb…` |
| `h65-endpoint` | `traffic_density` | frozen `per_tensor` plan, sha256 `a31a0d4b…`, tier fingerprint `10079cfa…` |
| `knapsack` | `profiled_knapsack` | uniform; critical-path profile built at prepare from the pilot's three disk calibration traces |

## Fixed inputs

- Exec checkout commit `637bbd55…`, clean; model-store manifest `abb500ea…`; H2D artifact `ef310198…`.
- Profile source: pilot `disk-trace-0/1/2.json`. Calibration prompts are disjoint from the evaluation prompts.
- Engine settings identical to the eight-pair confirmation: decode-table reuse on, decode slice 4,194,304, prefetch depth 2, per-blob reads, full LM head, static execution, no draft model, decoded RAM tier, one warm-up token, 10 s cooldown.
- Memory contract: 8 GB logical VRAM budget, 10 GB allocator cap, 12 GB whole-process ceiling, 16 GB RAM, 0.5 GB safety margin.

## Design

Six blocks, one reserved confirmation prompt each (`confirm-explain-tides`,
`confirm-arithmetic-boxes`, `confirm-code-deduplicate`, `confirm-compare-indexes`,
`confirm-summarize-wetlands`, `confirm-reasoning-switches`); 18 one-token cells.
Arm orders T-E-K, E-K-T, K-T-E, T-K-E, K-E-T, E-T-K put each arm twice in every
position and each ordered pair of arms adjacent equally often. These prompts were
used by the confirmation; this control is secondary and supports no primary claim.

## Gates

Any failure stops the run. No retries, no exclusions, no extension.

- All 18 cells complete; token-identical outputs across the three arms in every block.
- Every cell passes its measurement gates: cold page cache, thermal and power windows, resolved engine settings, one-token warm-up.
- Whole-process peak VRAM at or below 12 GB in every row.
- Frozen inputs unchanged; frozen plans verified at runtime for `traffic` and `h65-endpoint`.

## Predeclared analysis

Per-block ratios, geometric mean and 95% paired-t interval (df 5) for endpoint versus
knapsack latency (`knapsack_s / endpoint_s`), endpoint versus knapsack peak VRAM, and
each arm versus traffic.

- Both endpoint-versus-knapsack intervals include 1.0: attribute the benefit to cost-ranked placement, not to the H6.5 search.
- Either interval excludes 1.0 in the endpoint's favour, and neither in the knapsack's favour: the search adds measurable benefit beyond a greedy rule at this setting.
- Either interval favours the knapsack, and neither favours the endpoint: the search is not justified by this control.
- Otherwise: report both axes without a single attribution.

Six blocks cannot establish equivalence; "indistinguishable" is not "equal".

## Scope

Secondary mechanism evidence: cold one-token latency and peak VRAM for Llama-3.3-70B
on this RTX 5090 at this memory contract only.
