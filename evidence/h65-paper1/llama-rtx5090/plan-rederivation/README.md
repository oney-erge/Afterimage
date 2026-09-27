# Re-deriving the frozen H6.5 plan

This makes the *method* itself checkable, not just the measurements it
produced. Everything the H6.5 search actually consumes -- the recorded
calibration traces, the RAM-preparation cost profile, and the exact search
settings -- is here, verbatim, sourced from the run's own recorded planning
report. `optimize_h65_plan()`'s code has not changed since this plan was
built (see the git log for `afterimage/runtime/h65_planner.py`).

## What's here

- `disk-trace-0.json`, `disk-trace-1.json`, `disk-trace-2.json`: the three raw,
  disjoint calibration traces (event-DAGs), verified free of secrets and
  personal paths before publishing.
- `ram-prepare-seconds.json`: the decoded-RAM tensor preparation cost profile,
  copied directly from the run's own `planning.placement.ram_prepare_seconds`
  field.
- `rederive_plan.py`: reruns `optimize_h65_plan()` with these traces and the
  exact recorded search settings, then compares the result tensor-by-tensor
  against [`../frozen-plans/h65-endpoint-plan.json`](../frozen-plans/h65-endpoint-plan.json)
  (SHA-256 `a31a0d4bbe2309998514c70182a04bce2024c2a385dc99bfd1c0585f0aa3ebb4`),
  the plan D1 and D2 confirmed against.

## What you need to supply

The Llama-3.3-70B compressed store's `manifest.json`, SHA-256
`abb500ea82e7b979f20a32ffd466de639e31773678091ad823586c5a6798ac92`. It is not
in this repository -- it was not captured in the evidence snapshot this
folder was built from, and is presumably still on the machine that ran the
paper's campaigns. It is small (metadata only, not the store's weight
bytes); if you have it or can regenerate it with `afterimage compress
meta-llama/Llama-3.3-70B-Instruct`, run:

```bash
python evidence/h65-paper1/llama-rtx5090/plan-rederivation/rederive_plan.py \
  --manifest /path/to/manifest.json
```

No GPU needed. Takes a minute or two (256 search iterations over 723
tensors, offline replay only).

## Why this matters more than re-checking the numbers

Everything else in `evidence/h65-paper1/` proves the paper's measurements
are correctly summarized from what was actually run. This proves the
*placement decision itself* -- which tensor goes where -- is reproducible
from the recorded evidence, not just asserted. A referee who wants to check
"did the search actually find this plan, or was it hand-tuned" can run this
script and see for themselves.
