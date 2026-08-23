# Experiment results

No result is checked in merely because a method was implemented. Hardware runs
belong here only after completing the protocol in
[`docs/RESEARCH_METHODS.md`](../docs/RESEARCH_METHODS.md).

The server writes immutable run JSON under the configured Afterimage store root,
in `_experiment_results/`. A publishable result copied here should retain:

- hypothesis and profile IDs;
- every trial, randomized order, paired repeat, and primary raw metric;
- resolved `EngineConfig` and its fingerprint;
- exactness verdict and output token IDs where deterministic;
- environment manifest and model/tokenizer revisions;
- cache regime and cold-cache procedure;
- failures and excluded trials, without overwriting the original JSON.

Recommended filename:

```text
YYYY-MM-DD_HYPOTHESIS_MODEL_HARDWARE_RUNID.json
```

An `inconclusive` or `falsified` result is still a result. Do not delete it or
replace it with a later run under the same identifier.

A `*.json.partial` file is a run still being written (interrupted, or a
background process not yet finished). It is not committed -- see
`.gitignore` -- and it is not evidence for any hypothesis until the run
completes and is renamed to `.json`. Never treat a `.partial` file's contents
as a result, and never delete one without confirming the run that owns it
has actually stopped.

The staged cross-family runner makes these checkpoints resumable and preserves
the interrupted file with an `.interrupted-*` suffix:

```bash
python scripts/run_cross_model_campaign.py \
  --config configs/cross_model_benchmark_v1.json \
  --out-dir results/cross_model_YYYY-MM-DD --workers 1
python scripts/run_cross_model_campaign.py \
  --config configs/cross_model_benchmark_v1.json \
  --out-dir results/cross_model_YYYY-MM-DD --workers 1 --resume
```

The controlling 2026-08-22 cross-family interpretation and direct links to
every valid raw artifact are in
[`docs/CROSS_MODEL_BENCHMARK_2026-08-22.md`](../docs/CROSS_MODEL_BENCHMARK_2026-08-22.md).

For a short exploratory screen before that full protocol, use the diverse,
disjoint calibration/evaluation suite with a hard wall-time cap:

```bash
python -u scripts/run_bounded_suite.py \
  --time-budget-minutes 58 \
  --max-new-tokens 4 \
  --out results/YYYY-MM-DD_bounded_qwen3-14b_rtx3080.json
```

The result labels itself `exploratory` and
`confirmatory_protocol_satisfied=false`; a quick screen must never be promoted
into a five-repeat hypothesis verdict merely because its point estimate is
large.
