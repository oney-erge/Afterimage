# Frozen protocol: external comparison v6 (probe instrument corrected)

Frozen 2026-09-11, before any v6 measurement, and before v5's comparative result was
read. Science unchanged from `PROTOCOL-external-comparison-v4-20260911.md`; this
corrects one measurement instrument.

## Why v6

v4/v5 added a per-cell storage probe and a rule that reports the run as
environment-confounded if probe throughput spans more than 2x. The probe was sized at
4 GB, which on this device completes in 0.35-0.86 s. At that duration the reading is
dominated by start-up and scheduling noise: across v5's first sixteen cells it spanned
3.06-11.38 GB/s (3.7x) while the cell timings themselves stayed stable (AirLLM 46-48 s,
`exact-8gb` 88-92 s, `h65-endpoint` 35-43 s per request).

The rule is therefore being applied to an instrument that cannot resolve it. v6 changes
the probe size only.

## Change from v5

The per-cell probe reads **20 GB** (about 3 s) instead of 4 GB. Arms, prompts, block
count, engine settings, memory contract, inter-cell rest, cold-cache policy, thermal
gate, `--require-complete` and the analysis are exactly as in v4/v5.

## Standing of v5

v5 is preserved and reported descriptively. Its probe limb trips the pre-registered
rule, so it supports no cross-framework claim, and it is not pooled with v6. The
threshold was not relaxed to rescue it.

## Predeclared validity rule

Unchanged in substance, now applied to the corrected instrument. The comparison is
reported as environment-confounded, and not used for a comparative claim, if either:

- per-cell probe throughput across the run spans more than 2x, or
- any Afterimage arm's recorded I/O time for its fixed byte count spans more than 2x
  across blocks.

## Predeclared analysis

Per block, the geometric mean over the four prompts of `arm_wall / h65-endpoint_wall`;
across the eight blocks, geometric mean and 95% t interval (df 7) per arm; peak VRAM as
measured; output token IDs checked against `exact-8gb`. H6.5-endpoint is faster than an
arm if that arm's interval lies above 1.0, slower if below, otherwise indistinguishable.
AirLLM's lower peak VRAM is reported as a trade-off whatever the latency outcome.
