# scripts/

The scripts in this folder, grouped by what they're for. `install-utils.sh`
and `install-utils.ps1` are sourced by `run.sh`/`run.ps1`, not run directly.
Everything else is a standalone entry point; run any of them with `--help` (Python)
or read its header comment (shell) for its actual flags. `scripts/local/` is a
separate, gitignored personal workspace -- not indexed here.

## H6.5 (the paper)

See [`../docs/h65/README.md`](../docs/h65/README.md) for the full index and what each one does.
The short version:

| Script | Purpose |
|---|---|
| `benchmark_pinned_h2d.py` | Measure pinned H2D bandwidth, one input `h65-plan`/the paper's matrix needs. |
| `screen_h65_paper_budgets.py` | Offline: does a budget point produce a materially different placement-only vs. full plan? Gates whether a live cell is worth scheduling. |
| `run_h65_paper_matrix.py`, `run_h65_paper_worker.py` | The counterbalanced, cold-cache, gated causal matrix design (parent/worker pair). The paper's D1/D2 confirmations were driven by private wrapper scripts that reuse this worker unchanged with different frozen prompt sets -- see `docs/h65/README.md`'s protocol table for exactly which artifact backs which paper number. |
| `run_h65_direct_pair.py`, `run_h65_frozen_confirmation.py` | Earlier confirmation drivers, kept for traceability; check `docs/h65/README.md` and `evidence/h65-paper1/` before treating either as current. |
| `watch_h65_paper_matrix.py` | Progress/ETA for a running matrix. |
| `mark_paper1_result_validity.py` | Applies the deny-by-default validity grading (confirmatory/secondary/invalid/superseded) described in `evidence/h65-paper1/`. |

## H0-H18 (the opt-in research layer)

See [`../docs/RESEARCH_METHODS.md`](../docs/RESEARCH_METHODS.md) for the protocol
these implement (evidence levels L0-L3, per-hypothesis gates).

| Script | Purpose |
|---|---|
| `run_regulated_pair.py` | The general L1/L2 paired-block runner: `--hypothesis H<N>`. Start here for most H0-H18 work. |
| `run_offline_hypotheses.py` | H0/H3/H6/H7/H8: offline/artifact-only rows, no live generation. |
| `run_bounded_suite.py` | The five-way Qwen3-14B comparison behind the README's headline table (AirLLM, Accelerate, exact-min, exact-resident, fixed speculation). |
| `benchmark.sh` | One-command wrapper for `run_bounded_suite.py`'s canonical five-way Qwen3-14B comparison: `bash scripts/benchmark.sh canonical`. |
| `paper_benchmark.sh` | Restartable wrapper for `run_paper_comparison.py`'s external-package matrix (AirLLM, Accelerate, DeepSpeed). Not the H6.5 paper's protocol -- see `docs/h65/README.md` for that. |
| `run_hf_offload_baseline.py` | Hugging Face Accelerate baseline row. |
| `run_cross_model_campaign.py` | The Phi-4 Mini / Qwen3-14B / Mistral Small 24B cross-family campaign (`docs/CROSS_MODEL_BENCHMARK_2026-08-22.md`). |
| `run_headtohead.py` | Ad hoc two-method head-to-head, outside the regulated-pair protocol. |
| `run_h19_candidate_sweep.py`, `run_h21_collect_target_corpus.py`, `run_h21_combine_sources.py`, `run_h21_score_source.py`, `run_h22_disagreement_hmm.py` | The H19-H34 speculation-tree line (`docs/SPECULATION_TREE_RESEARCH.md`); CPU-only offline analysis, no GPU campaign has run for these yet. |
| `power_analysis.py` | Retrospective sample-size/power check for a completed L1/L2 screen, from its own `results/*.json`. |
| `cpu_decode_gate.py`, `h2_cpu_decode_sweep.py` | H2 (CPU decode path) gating and sweep. |
| `adaptive_bench.py` | Adaptive-policy (H3/H8-family) benchmark driver. |

## Baselines, comparisons, and campaign infrastructure

| Script | Purpose |
|---|---|
| `run_paper_comparison.py`, `run_paper_comparison_worker.py` | The general paper-comparison matrix runner (parent/worker); output is deliberately not published, see `.gitignore`. |
| `matched_vram_final.py`, `methods_vs_airllm.py` | Matched-memory and AirLLM-relative comparison summarizers. |
| `build_results_index.py` | Regenerates `results/INDEX.md` from the `results/` tree. |
| `campaign_status.py` | Status for a long-running campaign. |

## Diagnostics and CI

| Script | Purpose |
|---|---|
| `verify_odirect.py` | Checks whether this storage path supports O_DIRECT reads -- run it before trusting a cold-cache benchmark number, not after. |
| `run_probe_real.py` | A real (not synthetic) I/O probe, used to diagnose the storage-probe confound noted in `docs/h65/protocols/PROTOCOL-external-comparison-v6-20260911.md`. |
| `check_prose.py` | CI's em-dash/AI-prose check over every tracked Markdown file. |
| `install-utils.sh`, `install-utils.ps1` | Shared install mechanics (locking, retries, disk checks, CUDA-wheel-index detection) sourced by `run.sh`/`run.ps1`. Not run directly. |
| `paper1_5090_guidance.sh` | Self-contained reproduction of the earlier (pre-H6.5) H6 figure/table set on a second machine; see the note in `docs/REPRODUCE.md`. |
