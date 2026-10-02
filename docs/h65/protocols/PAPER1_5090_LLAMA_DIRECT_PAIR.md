# Direct-pair Llama-70B RTX 5090 H6.5 confirmation protocol

Frozen before any cell in this experiment is launched. This is an independent,
noise-reduced confirmation of the primary placement contrast. No result from the
earlier pilot or four-method confirmation is pooled into this dataset.

## Fixed inputs

- Model: `meta-llama/Llama-3.3-70B-Instruct`.
- Store: the existing exact Afterimage Llama-70B store and immutable manifest.
- Hardware: RTX 5090 machine.
- Logical budget: 8 GiB VRAM, 16 GiB RAM, 0.5 GiB VRAM safety margin.
- Measured transfer artifact: pageable/blocking RTX 5090 H2D calibration.
- Held-out prompts: `fact-gold`, `arithmetic-17x6`, `code-square`, and
  `retrieval-7319`.
- Workload: one generated token per prompt. The claim is limited to bounded,
  cold-cache TTFT/one-token latency.
- Seed: `20260902`.
- Blocks: twelve complete adjacent pairs (24 fresh-process cells, 96 rows).
- Hard timeout: 20 minutes per cell; no optional stopping.

## Fixed methods and plan

The control is traffic-density placement with decoded RAM. The treatment is
H6.5 placement-only using the exact frozen candidate produced by the regulated
pilot:

- pilot artifact:
  `h65-paper-matrix-meta-llama-llama-3.3-70b-instruct-rtx5090-20260831-v1.json`;
- H6.5 candidate SHA-256:
  `28bcc25502025e0720912a6a1393eb9d9ff52fd51121a54c95d64e9f6d523838`.

The plan is not rebuilt, recalibrated, selected, or guarded using measurements
from this experiment. The runner must reject a candidate whose hash differs or
whose pilot record does not show causal divergence from traffic placement.

## Pairing and order

Each block runs the two methods consecutively with no intervening measurement
cell. The first method alternates by block, yielding six traffic-first pairs and
six H6.5-first pairs. Every cell is a fresh process. The page cache is dropped
before every timed prompt, and the existing thermal and whole-process VRAM gates
remain active.

## Outcomes

The primary estimand is the mean paired block log latency ratio,
`log(traffic / H6.5)`. It is reported as a geometric speedup with a two-sided
95% paired Student-t interval. Each block's latency is the median seconds per
token across the four fixed prompts. The median paired latency reduction is a
descriptive companion statistic. Results are reported regardless of direction;
the run is not extended based on the observed effect.

## Validity gates

The artifact is procedurally confirmatory only if all 24 cells and 96 rows
complete; pair order is exactly balanced; every pair is adjacent; the frozen
candidate hash remains unchanged; the pilot records a divergent candidate;
output token IDs match within every block/case pair; token counts match; cache,
thermal, logical-budget, whole-process VRAM, protocol, and source-snapshot gates
all pass. A timeout or failed gate remains visible and invalidates the
confirmatory label.
