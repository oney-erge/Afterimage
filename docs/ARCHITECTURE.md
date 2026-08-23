# Afterimage architecture

<p align="center">
  <img src="assets/afterimage-logo.png" width="120" alt="Afterimage logo">
</p>

Afterimage runs a model that is larger than available VRAM by keeping its exact
BF16 weights in a compressed store, assigning tensors to a memory tier, and
materializing streamed weights only for the layer currently executing. This
page contains the system and research-flow diagrams; the top-level README stays
focused on capabilities, measured results, and the quick start.

For the measurement-backed explanation of each method, see
[HOW_IT_WORKS.md](HOW_IT_WORKS.md). For implementation details and experiment
contracts, see [RESEARCH_METHODS.md](RESEARCH_METHODS.md).

## Lossless weight store

A BF16 word contains a sign bit, eight exponent bits, and seven mantissa bits.
The mantissa is nearly incompressible in the measured checkpoint, so Afterimage
stores sign plus mantissa as a raw byte and Huffman-codes the exponent in bounded
chunks. Reconstruction combines disjoint fields and is bit-exact.

```mermaid
flowchart LR
  A["BF16 tensor<br/>16 bits/weight"] --> B{split fields}
  B -->|"sign + mantissa"| C["packed raw<br/>8 bits/weight"]
  B -->|"exponent"| D["chunked Huffman<br/>~2.6 bits/weight"]
  C --> E["weights.bin<br/>flat, CRC32 per blob"]
  D --> E
  E --> F["manifest.json<br/>offset, bytes, shape, dtype"]
```

The measured Qwen3-14B store is 20.3 GB instead of 29.5 GB, a 1.453x exact
compression ratio. The manifest makes every blob addressable without loading
the complete store.

## Memory tiers and streamed execution

The planner assigns tensors to VRAM, pinned host RAM, disk, or row-gathered
embedding access under explicit budgets. Disk-tier tensors are prefetched while
the previous layer decodes and computes. Once a streamed layer finishes, its
temporary allocation is released before the next one is materialized.

```mermaid
flowchart TD
  M["manifest.json"] --> P["VRAM planner<br/>traffic avoided per VRAM byte"]
  P --> T1["VRAM tier<br/>resident, never re-read"]
  P --> T2["RAM tier<br/>pinned host memory"]
  P --> T3["disk tier<br/>re-streamed per target sweep"]
  P --> T4["row gather<br/>only requested embedding rows"]

  T3 --> R["prefetch pool<br/>reader threads, depth d"]
  R --> IO["pread weights.bin"]
  IO --> DEC["bounded GPU Huffman decode"]
  T2 --> H2D["host-to-device copy"]
  DEC --> L["decoder-layer forward"]
  H2D --> L
  T1 --> L
  L --> FREE["release streamed allocation"]
  FREE -.->|next layer| R
```

`vram_budget_gb` and `ram_budget_gb` define the operating point. The feasibility
planner rejects a configuration that cannot satisfy its declared exactness and
memory contract instead of silently choosing an approximate fallback.

## Speculative decoding

For a streamed target, the dominant unit of work is a complete pass through the
weight store. A small resident draft model proposes several tokens; the target
verifies the chain in one streamed pass. Rejection sampling preserves the
target distribution, while a good draft amortizes one expensive sweep across
multiple committed tokens.

```mermaid
sequenceDiagram
  participant D as Draft model (resident)
  participant T as Target model (streamed)
  D->>D: propose k tokens cheaply
  D->>T: submit the candidate chain
  Note over T: one complete weight-store sweep
  T->>T: verify k positions in parallel
  T-->>D: accept prefix; resample first rejection
  Note over D,T: one sweep commits 1..k+1 tokens
```

Fixed speculative decoding is the strongest measured full-suite operating
point. Adaptive stopping policies remain experiments and do not replace the
fixed default unless their registered evidence gate is met.

## Research evidence flow

The experiment lab separates invariant checks, mechanism checks, bounded
screens, and confirmation. A result cannot be called a performance failure if
the candidate never took a different action, the host lacked a required
capability, or the checkpoint did not contain the required structure.

```mermaid
flowchart LR
  L0["L0 invariant<br/>exactness + budget contracts"] --> L1["L1 mechanism<br/>did the action occur?"]
  L1 --> L2["L2 regulated screen<br/>advance / redesign / stop"]
  L2 --> L3["L3 confirmation<br/>fixed sample, 95% LCB > 0"]
  L1 -.->|no action or capability| G["gated / blocked"]
  L2 -.->|futility or regression| S["stop current candidate"]
```

The current H0–H18 interpretation is summarized in the
[README](../README.md#research-status-mixed-evidence-no-l3-confirmation), with
the controlling measurements in
[FINAL_TEST_RESULTS_2026-08-21.md](FINAL_TEST_RESULTS_2026-08-21.md).

## Main implementation boundaries

| Area | Primary modules | Responsibility |
|---|---|---|
| Store | `compressed_store.py`, `huffman_chunked.py`, `layout.py` | Exact encoding, manifests, checksums, and physical layout |
| Planning | `vram_planner.py`, `resident.py`, `tiers.py` | Budget feasibility and tensor placement |
| Streaming | `streamer.py`, `streaming_engine.py`, `gpu_decode.py` | Prefetch, decode, transfer, execution, and release |
| Speculation | `draft.py`, `spec_policy.py`, `controllers.py` | Draft generation, exact verification, and opt-in policies |
| Measurement | `bench/`, `experiments.py`, `scripts/run_regulated_pair.py` | Paired runs, evidence metadata, and immutable result output |
| Service | `server/app.py`, `server/jobs.py`, `server/static/` | OpenAI-compatible API, job control, and experiment UI |
