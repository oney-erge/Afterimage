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
`abb500ea82e7b979f20a32ffd466de639e31773678091ad823586c5a6798ac92`. It is not in
this repository and was not kept from the RTX 5090 campaign, but it can be
regenerated: compression is deterministic (two independent compressions of
Qwen3-0.6B produced a byte-identical `manifest.json` and `weights.bin`), and the
compressor code has not changed since the Llama campaign's commit. So

```bash
afterimage compress meta-llama/Llama-3.3-70B-Instruct   # ~141 GB download
python evidence/h65-paper1/llama-rtx5090/plan-rederivation/rederive_plan.py \
  --manifest ~/.afterimage/stores/meta-llama__Llama-3.3-70B-Instruct/manifest.json
```

should reproduce it. This has not been verified for Llama itself; the script
checks the manifest's hash first and says so if it differs (for example, if the
Hugging Face checkpoint revision has changed since). The re-derivation itself
needs no GPU and takes a minute or two (256 search iterations over 723 tensors).

The laptop plans, whose manifests are published, re-derive without any of this:
see [`../../laptop-rtx3080/`](../../laptop-rtx3080/README.md).

## Why this matters more than re-checking the numbers

Everything else in `evidence/h65-paper1/` proves the paper's measurements
are correctly summarized from what was actually run. This proves the
*placement decision itself* -- which tensor goes where -- is reproducible
from the recorded evidence, not just asserted. A referee who wants to check
"did the search actually find this plan, or was it hand-tuned" can run this
script and see for themselves.
