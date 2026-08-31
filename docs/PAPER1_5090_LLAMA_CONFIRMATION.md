# Frozen Llama-70B RTX 5090 H6.5 confirmation protocol

Frozen before any confirmatory cell is launched. The exploratory pilot is not
pooled into this dataset.

## Fixed inputs

- Model: `meta-llama/Llama-3.3-70B-Instruct`.
- Store: the existing exact Afterimage Llama-70B store and immutable manifest.
- Hardware: RTX 5090 machine.
- Logical budget: 8 GiB VRAM, 16 GiB RAM, 0.5 GiB VRAM safety margin.
- Measured transfer artifact: pageable/blocking RTX 5090 H2D calibration.
- Calibration prompts: `summary-photosynthesis`, `logic-glippets`, and
  `copy-nonce`.
- Held-out prompts: `fact-gold`, `arithmetic-17x6`, `code-square`, and
  `retrieval-7319`.
- Workload: one generated token per held-out prompt (bounded cold-start/TTFT
  claim only).
- Seed: `20260901`.
- Blocks: eight complete four-position-balanced blocks.
- Hard timeout: 20 minutes per fresh-process cell; no optional stopping.

## Fixed methods

1. disk-only exact streaming;
2. traffic-density placement with decoded RAM;
3. H6.5 placement-only with compressed RAM disabled; and
4. full H6.5 joint representation/tier planning.

Plans are built only from the three disjoint all-disk calibration traces and
frozen before held-out execution. The candidate plan is evaluated even if an
offline deployment guard would select control, so causal efficacy and guarded
deployment are not conflated.

## Outcomes

The primary contrast is traffic-density placement versus H6.5 placement-only.
The primary estimand is the mean paired block log latency ratio, reported as a
geometric speedup with a two-sided 95% paired interval. Full H6.5 versus traffic
is secondary; full versus placement-only is a representation-selection
ablation. A negative or null estimate remains a valid confirmatory outcome.

The pilot's approximate paired-log calculation suggested fewer than eight
blocks for 80% power, but the previously frozen minimum of eight balanced
blocks is retained for robustness. Pilot values are used only for this sample
size check and are not combined with the confirmation.

## Validity gates

The artifact is procedurally confirmatory only if all requested cells and rows
complete; method position is balanced; calibration and evaluation are
disjoint; raw traces and source snapshots remain present; exact output token
IDs match disk-only; token counts match; cache, thermal, whole-process VRAM,
and matched-budget gates pass; and both H6.5 candidates causally differ from
traffic placement. Failed gates or timed-out cells invalidate the confirmatory
label but remain visible in the artifact.
