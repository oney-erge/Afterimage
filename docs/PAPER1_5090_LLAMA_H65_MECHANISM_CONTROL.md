# RTX 5090 Llama H6.5 mechanism control (greedy profiled knapsack)

Frozen on 2026-09-11, before any mechanism-control measurement. This is the
secondary control behind the paper's schedule-aware-versus-isolated-cost comparison
(paper Section 5.3 / Table 8).

## Question

Does the engine's simpler cost-ranked placement (`placement_policy="profiled_knapsack"`),
given the same measured disk-calibration costs the H6.5 planner used, reproduce the
H6.5 endpoint plan's latency and peak-VRAM behaviour on Llama-3.3-70B at the
confirmation's memory contract? This isolates whether the benefit needs H6.5's
whole-schedule replay search, or whether ranking tensors by isolated preparation cost
per byte gets the same result.

## Arms

All three arms are exact; output tokens must match across arms in every block.

| Arm | Placement policy | Representation |
|---|---|---|
| `traffic` | `traffic_density` | frozen plan, SHA-256 `926b0adb51275de35473a76ac3fc7dc7d82c58d3669027118f5bd433bb585563` |
| `h65-endpoint` | `traffic_density` | frozen plan, SHA-256 `a31a0d4bbe2309998514c70182a04bce2024c2a385dc99bfd1c0585f0aa3ebb4`, tier fingerprint `10079cfa340912b45b27f7be200402db2d47e1bb96d3ddf3948b91ee25b58cac` |
| `knapsack` | `profiled_knapsack` | uniform; critical-path profile built at prepare time from the pilot's three disk calibration traces |

## Fixed inputs

Execution commit `637bbd55`; model-store manifest SHA-256
`abb500ea82e7b979f20a32ffd466de639e31773678091ad823586c5a6798ac92`; H2D artifact
SHA-256 `ef310198e36140dd1f315a09ce6bee287ee2c9ed029b08fe774814b6000e4a0b`. The knapsack
profile comes from the pilot's three disjoint all-disk calibration traces; calibration
and evaluation prompts are disjoint. Engine settings match the endpoint confirmations:
decode-table reuse on, 4,194,304-element decode slices, prefetch depth two, per-blob
reads, full LM head, static execution, no draft model, decoded RAM tier, one warm-up
token, 10 s cooldown. Memory contract: 8 GB logical VRAM budget, 10 GB allocator cap,
12 GB whole-process ceiling, 16 GB RAM, 0.5 GB safety margin.

## Design

Six blocks, each on one of the reserved confirmation prompts (`confirm-explain-tides`,
`confirm-arithmetic-boxes`, `confirm-code-deduplicate`, `confirm-compare-indexes`,
`confirm-summarize-wetlands`, `confirm-reasoning-switches`) — 18 one-token cells. Arm
orders T-E-K, E-K-T, K-T-E, T-K-E, K-E-T, E-T-K put each arm twice in every position
and each ordered pair of arms adjacent equally often. These prompts were reused from
the first endpoint confirmation, so this control is secondary and supports no primary
claim on its own.

## A harness bug found and fixed before v2

The first attempt (v1) completed its `traffic` and `h65-endpoint` cells, then stopped
on its first `knapsack` cell with
`placement_policy='profiled_knapsack' requires critical_path_profile`, even though the
cell's own config held the correct profile path. The frozen matrix worker was calling
`run_afterimage(..., critical_profile=None)`, and the engine's `engine_for` helper
replaced the configured profile path with that `None` argument whenever the placement
policy needed a profile — so every `profiled_knapsack` cell run through that worker
failed the same way. v2 runs every cell through a small wrapper that loads the frozen
worker unchanged and forwards the cell's own `critical_path_profile` only when the
worker itself passes none; cells that need no profile (`traffic`, `h65-endpoint`) take
exactly the original frozen code path. v1 is preserved and marked invalid; none of its
cells are reused. The question, arms, frozen inputs, engine settings, memory contract,
six blocks, arm orders, gates, and predeclared analysis below are unchanged between v1
and v2.

## Gates

Any failure stops the run. No retries, exclusions, or extensions.

- All 18 cells complete; token-identical outputs across the three arms in every block.
- Every cell passes its measurement gates: cold page cache, thermal and power windows,
  resolved engine settings, one-token warm-up.
- Whole-process peak VRAM at or below 12 GB in every row.
- Frozen inputs unchanged; frozen plans verified at runtime for `traffic` and
  `h65-endpoint`.

## Predeclared analysis

Per-block ratios, geometric mean, and 95% paired-t interval (df 5) for endpoint versus
knapsack latency (`knapsack_s / endpoint_s`), endpoint versus knapsack peak VRAM, and
each arm versus traffic.

- Both endpoint-versus-knapsack intervals include 1.0: attribute the benefit to
  cost-ranked placement, not to the H6.5 search.
- Either interval excludes 1.0 in the endpoint's favour, and neither in the knapsack's
  favour: the search adds measurable benefit beyond a greedy rule at this setting.
- Either interval favours the knapsack, and neither favours the endpoint: the search
  is not justified by this control.
- Otherwise: report both axes without a single attribution.

Six blocks cannot establish equivalence; "indistinguishable" is not "equal".

## Scope

Secondary mechanism evidence only: cold one-token latency and peak VRAM for
Llama-3.3-70B on this RTX 5090 at this memory contract. See
[../evidence/h65-paper1/README.md](../evidence/h65-paper1/README.md) for the frozen
artifact this protocol produced and a script that recomputes the paper's Table 8
numbers from it.
