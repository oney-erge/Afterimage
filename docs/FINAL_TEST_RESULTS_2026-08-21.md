# Superseded regulated results snapshot

This file captured the 2026-08-21 H9/H11-H15 checkpoint. It is retained so
older links do not break, but it is **not the current conclusion**.

The reevaluation on 2026-08-22:

- ran H0, H3, H6, H7, and H8 through an artifact/replay runner;
- raised the WSL memlock limit and ran a genuine pinned H9 mechanism screen on
  a model whose full output head fits the host's pinned-memory ceiling;
- added a real four-family Hugging Face Accelerate BF16 GPU/CPU/disk baseline;
- corrected H6's RAM-transfer cost with a measured 6.669 GB/s pinned H2D floor;
- kept fixed speculation's 3.15x AirLLM result separate from the H2/H11
  adaptive candidates that failed to beat it.

The same controlling report now also includes H16-H18. H16/H17 regressed;
H18 passed exact KV-rollback mechanics but stopped at its randomized L2
futility gate. This snapshot remains only a pointer.

Read the current controlling report:

## [All hypotheses, baselines, results, and ranking](ALL_HYPOTHESES_AND_BASELINES.md)

The raw 2026-08-21 JSON files remain immutable in [`results/`](../results/).
