# Frozen protocol: external comparison v4 (measurement controls added)

Frozen 2026-09-11, before any v4 measurement. The SHA-256 of this file is recorded in
the v4 launch record. Science unchanged from
`PROTOCOL-external-comparison-v3-20260911.md`; this adds measurement controls only.

## Why v4

v3 was stopped after four blocks on an outcome-independent criterion: the H6.5 arm read
exactly 78.2 GB in every request, yet its I/O time moved from 78-86 s in block 1 to
186-281 s in block 2, with prefetch wait rising from 9-15 s to 60-109 s, while
`airllm` (47-53 s) and `exact-8gb` (I/O 79-104 s) stayed flat. Arms were therefore not
measured under comparable storage conditions.

A standalone probe immediately afterwards read the store at 3.75 GB/s, then 6.11-7.34
GB/s over the following ten minutes. The slow cells were achieving 0.28-0.6 GB/s, so the
device was not the limit and the cause is unresolved. v3 is preserved and marked
invalid; none of its cells are reused.

## Changes from v3

1. **Idle rest** of 120 s before every cell.
2. **Per-cell storage probe**: with the page cache dropped, 4 GB of the store blob is
   read across three concurrent streams and the throughput recorded to
   `llama70b-stage5-external-endpoint-rtx5090-20260911-v4-storage-probe.jsonl`, keyed by
   block and arm, immediately before the cell runs.
3. Time budget raised to 420 minutes to cover the added rest; seed 20260913.

Arms, prompts, block count, engine settings, memory contract, cold-cache policy,
thermal gate and `--require-complete` are exactly as in v3.

## Predeclared validity rule

The comparison is reported as **environment-confounded, and not used for a comparative
claim**, if either holds:

- per-cell probe throughput across the run spans more than 2x, or
- any Afterimage arm's recorded I/O time for its fixed byte count spans more than 2x
  across blocks.

In that case the run is preserved and reported with the probe series, and the paper
makes no cross-framework latency claim. A confounded run is never repaired by deleting
cells.

## Predeclared analysis

Unchanged from v3: per block, the geometric mean over the four prompts of
`arm_wall / h65-endpoint_wall`; across the eight blocks, geometric mean and 95% t
interval (df 7) per arm; peak VRAM as measured; output token IDs checked against
`exact-8gb`. H6.5-endpoint is faster than an arm if that arm's interval lies above 1.0,
slower if below, otherwise indistinguishable. AirLLM's lower peak VRAM is reported as a
trade-off whatever the latency outcome.
