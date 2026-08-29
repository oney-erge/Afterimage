# Reproducing the published results

One entry point per reported number. This page indexes commands and result
files; it does not restate protocol detail already documented elsewhere --
see [RESEARCH_METHODS.md](RESEARCH_METHODS.md) for the L0-L3 evidence levels
and per-hypothesis protocols, and
[ALL_HYPOTHESES_AND_BASELINES.md](ALL_HYPOTHESES_AND_BASELINES.md) for what
every number means.

## What needs no hardware

```bash
pip install -e ".[dev,server]"
python -m compileall -q afterimage
python -m pytest -q
```

`286` of the tests that can pass without CUDA pass on any machine; the rest
are skipped, not failed, and are named individually in the output (`-rs`
shows the reason). This proves the codec, config validation, planners,
protocols, and API logic are correct. It does not exercise the GPU decode
kernels, the streaming engine's live generation path, or any wall-clock
number in the paper -- those need the hardware runs below.

```bash
python -m pytest -q -rs   # -rs prints why each test was skipped
python -m ruff check afterimage
python scripts/check_prose.py
```

## What needs an NVIDIA GPU

Every command below refuses to run against an uncommitted working tree
(`git status --short` non-empty) unless `--allow-dirty-tree` is passed; the
result JSON records `reproducible_from_commit` accordingly. Do not cite a
`--allow-dirty-tree` result as evidence for a publishable claim -- see
[results/README.md](../results/README.md).

| Reported number | Model | Command | Result file |
|---|---|---|---|
| Headline table (README "Results") | Qwen3-14B | `python scripts/run_bounded_suite.py --time-budget-minutes 58 --max-new-tokens 4 --out results/YYYY-MM-DD_bounded_qwen3-14b_HARDWARE.json` | `results/2026-08-21_bounded_qwen3-14b_rtx3080_run1.json` |
| Cross-family Pareto table | Phi-4 Mini, Qwen3-14B, Mistral Small 24B | `python scripts/run_cross_model_campaign.py --config configs/cross_model_benchmark_v1.json --out-dir results/cross_model_YYYY-MM-DD --workers 1` (add `--resume` to continue an interrupted run) | `results/cross_model_2026-08-22_full_v2/`, interpreted in [CROSS_MODEL_BENCHMARK_2026-08-22.md](CROSS_MODEL_BENCHMARK_2026-08-22.md) |
| AirLLM baseline row | Qwen3-14B | `pip install -e ".[bench]"` then the AirLLM path inside `run_bounded_suite.py` / `run_cross_model_campaign.py` | see `packages.airllm` in each result's `environment` block |
| HF Accelerate baseline row | Qwen3-14B | `python scripts/run_hf_offload_baseline.py --gpu-memory 4000MB --cpu-memory 8GB --max-new-tokens 4 --out results/HF_ACCELERATE.json` | `results/2026-08-22_hf_accelerate_gpu_disk_qwen3-14b_rtx3080_full_l1.json` |
| H0/H3/H6/H7/H8 offline/artifact rows | -- (no live generation) | `python scripts/run_offline_hypotheses.py --manifest STORE/manifest.json --moe-shard PATH --out results/OFFLINE.json` | `results/2026-08-22_offline_h0_h3_h6_h7_h8.json` |
| H9 pinned-RAM overlay (the strongest lead) | Qwen3-0.6B (14B needs a host that can pin ≥1.6 GB; see below) | see the `systemd-run ... LimitMEMLOCK=infinity` command in [ALL_HYPOTHESES_AND_BASELINES.md](ALL_HYPOTHESES_AND_BASELINES.md#reproduce-the-new-runs) | `results/2026-08-22_h9_pinned_overlay_qwen3-0.6b_rtx3080_l1.json` |
| Any other H1-H18 regulated pair | -- | `python scripts/run_regulated_pair.py --hypothesis H<N> --blocks 2 --max-new-tokens 8 --time-budget-minutes 40 --out results/H<N>_L2.json` | filenames listed under "Raw evidence" in [ALL_HYPOTHESES_AND_BASELINES.md](ALL_HYPOTHESES_AND_BASELINES.md) |

For the canonical five-way Qwen3-14B comparison, prepare the Linux/WSL2
machine and run:

```bash
bash benchmark.sh canonical
```

This runs AirLLM, Hugging Face Accelerate disk offload, exact-min,
exact-resident, and fixed speculation for five cold-cache repeats. Override
`AFTERIMAGE_MODEL`, `AFTERIMAGE_STORE`, `AFTERIMAGE_HF_OFFLOAD_DIR`,
`AFTERIMAGE_BENCHMARK_REPEATS`, `AFTERIMAGE_BENCHMARK_TIME_MINUTES`, or
`AFTERIMAGE_BENCHMARK_OUT` to use another prepared model or campaign budget.

For the restartable paper matrix, including AirLLM, Accelerate, DeepSpeed
ZeRO-Inference, and the separate short-answer and long-generation workloads,
run:

```bash
bash paper_benchmark.sh
```

The wrapper refuses a dirty tree or missing benchmark dependency, resumes
partial non-capacity cells, and withholds incomplete matrices.

Its output goes to `results/paper-comparison/`, which is deliberately not
published (see `.gitignore`): a live campaign directory accumulates partial
and exploratory attempts, and only a completed, paper-eligible artifact
should be curated into the date-stamped published result set. Check
`paper_eligible` and `paper_eligibility_reason` in a result file before
citing it -- a matrix with missing or failed cells reports `false` and
says which cells are missing.

## Reproducing Paper 1's figures on a different GPU

`paper_benchmark.sh` above reproduces only the headline TTFT/Pareto
comparison. To reproduce Paper 1's full figure and table set (H6
representation-and-tier planning: Figures 2, 3, 4, 5, 7, 9 and Table 2) on
a second machine with different hardware, run:

```bash
AFTERIMAGE_MODEL=Qwen/Qwen3-14B \
AFTERIMAGE_DRAFT_MODEL=Qwen/Qwen3-0.6B \
  bash scripts/paper1_5090_guidance.sh
```

This is self-contained: it builds the compressed and raw stores, measures
the machine's own pinned-H2D bandwidth, generates the model's own H1
critical-path profile, runs a real single-token smoke gate before
committing to hours of GPU time, then runs all six experiments with the
same budgets/repeats/case-IDs used for the published Qwen3-14B and
Gemma-2-27B (RTX 3080) result sets, tagging every output file by GPU so
results from different machines never collide. See the script's own header
for the full parameter list, including `AFTERIMAGE_EXACT_MIN_VRAM_GB`,
which must be raised above its Qwen-tuned default for large-vocabulary
models such as Gemma. This does not cover H0-H18
([ALL_HYPOTHESES_AND_BASELINES.md](ALL_HYPOTHESES_AND_BASELINES.md)) or the
H19-H34 speculation-tree line
([SPECULATION_TREE_RESEARCH.md](SPECULATION_TREE_RESEARCH.md)).

## Exact software versions used for the published numbers

Every result JSON's `environment` block is authoritative and per-run (Python,
Torch, CUDA, driver, GPU, host memory, and `packages.{airllm,transformers,
accelerate,safetensors,numpy}`). There is no single pinned "the" version
across all results -- `uv.lock` pins the *application's* dependencies, not
the exact package snapshot behind any one historical benchmark row. When
comparing against a competitor, check the specific result file's
`environment.packages` field rather than assuming the current `uv.lock`.

## Replication: `--repeats`

`run_bounded_suite.py` historically ran each (method, case) cell exactly
once, so every headline seconds/token was a single unreplicated
observation. `--repeats N` runs N complete sweeps of every case per method,
re-dropping the page cache before each cell, and the summary then reports
spread across repeats:

```bash
python scripts/run_bounded_suite.py --repeats 3 --time-budget-minutes 170 \
  --out results/YYYY-MM-DD_bounded_qwen3-14b_HARDWARE.json
```

Each method's `summary` gains `per_repeat_seconds_per_token`,
`repeat_median_seconds_per_token`, `repeat_min/max_seconds_per_token`,
`all_repeats_complete`, and (from three repeats up)
`repeat_stdev_seconds_per_token` and `repeat_relative_stdev`. These are
deliberately not labelled confidence intervals: three repeats is far too
few for one to mean anything.

**Budget for it.** `--repeats N` multiplies wall time by roughly N; raise
`--time-budget-minutes` to match or the later repeats are truncated
(`all_repeats_complete` goes false and the run says so rather than
averaging a partial sweep against a full one).

**Expect the first repeat to be the slowest.** In validation on Qwen3-0.6B,
repeat 0 measured 0.165 s/token against 0.047 and 0.056 for repeats 1 and 2:
a 74% relative standard deviation driven almost entirely by first-pass
warmup (CUDA context, Triton JIT, allocator growth). Report repeat 0
separately or treat it as burn-in; do not average it in silently. This
effect is proportionally smaller on a 14B model, where per-token streaming
dominates initialization, but it does not vanish.

## Power analysis for a confirmatory (L3) run

```bash
python scripts/power_analysis.py
```

Reads every `results/*.json` produced by `run_regulated_pair.py` (anything
with paired `trials`), estimates the run-to-run variance of the paired
log-ratio from what already ran, and reports the paired sample count an L3
confirmation at that hypothesis's registered gate would need at 80%/90%
power, plus the power the completed screen actually had at its own n. Rows
built from fewer than 4 pairs are flagged: with 1-2 degrees of freedom the
variance estimate is not trustworthy. This is retrospective, not a
substitute for preregistering n before a specific confirmatory run.

## Environment facts a stranger's rerun will not match by default

- **Cache regime.** Every real result requires cold page cache before each
  timed cell (`cache_regime` in the result JSON). `drop_caches` writes
  `/proc/sys/vm/drop_caches`, which drops the Linux VM's page cache; it
  cannot reach a WSL2 host's Windows-side cache directly. Measured on this
  host (2026-08-26, `dd` reading the full 1.07 GB `weights.bin` of the
  0.6B store): a genuinely warm read (no intervening drop) hits **16.6
  GB/s**; a `drop_caches` read measures **2.3 GB/s**, and repeating
  `drop_caches` immediately after another cold read reproduces the same
  2.3 GB/s within 2%. `drop_caches` therefore does produce a stable,
  repeatable cold state on this host -- run-to-run *comparisons* are not
  confounded by warm-cache drift. Whether 2.3 GB/s is the NVMe's true
  from-metal cold speed or includes some Windows-side residual is not
  established (no fully-cold Windows-side baseline was measured); if that
  distinction matters for a claim, measure it on native Linux instead of
  assuming either answer.
- **Pinned memory ceiling.** WSL2 commonly caps `RLIMIT_MEMLOCK` well below
  what H9 needs at 14B scale; see
  [NVIDIA's CUDA-on-WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
  Native Linux or `systemd-run -p LimitMEMLOCK=infinity` removes the cap.
- **Clean tree.** See the table above -- a result written with
  `--allow-dirty-tree` is not reproducible from its `git_commit` alone; the
  environment block's `git_status` shows what was uncommitted.
