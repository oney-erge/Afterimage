# RTX 5090 Llama H6.5 external-framework comparison (descriptive, probe-confounded)

Frozen protocol dated 2026-09-11 (v6, correcting v5's measurement instrument). The
result artifact in this repository's evidence folder is **v5**, run under the prior
protocol version; no v6 measurement exists in the local evidence snapshot this
repository was built from.

**Status as of the 2026-09-26 arXiv draft: not cited.** The draft's external-package
comparison (Section 5.4, Figure 7) covers the RTX 3080 laptop runs (Qwen3-14B,
Gemma 2 27B), not this Llama/RTX 5090 comparison. This document and its artifact are
published for completeness and because the bundle's own summary presents the v5 table
descriptively; do not treat it as backing a specific paper table or figure unless the
paper is revised to cite it, and if it is, update this note to say where.

## Why this exists

Alongside the causal placement confirmations, a separate cross-framework comparison
measured H6.5-endpoint against Hugging Face Accelerate (8 GB GPU cap), AirLLM, and
Afterimage's own exact-residency-at-8GB control, on the same prompts, checkpoint, and
process-isolated protocol.

## The v5 probe confound

v4 and v5 added a per-cell storage-throughput probe and a rule: if probe throughput
across the run spans more than 2x, the comparison is reported as
environment-confounded and used for no comparative claim. The probe read 4 GB per
cell (about 0.35-0.86 s on this device). At that duration the reading is dominated by
start-up and scheduling noise: across v5's first sixteen cells it spanned 3.06-11.38
GB/s (3.7x), while the cells' own timings stayed stable (AirLLM 46-48 s, `exact-8gb`
88-92 s, `h65-endpoint` 35-43 s per request). **v5 trips its own pre-registered
confound rule and supports no cross-framework claim.** The threshold was deliberately
not relaxed to rescue a favorable-looking result.

v6 changes only the probe: it reads 20 GB (about 3 s) instead of 4 GB, on the
reasoning that a longer read is less dominated by start-up noise. Arms, prompts,
block count, engine settings, memory contract, inter-cell rest, cold-cache policy,
thermal gate, and the analysis are otherwise identical between v5 and v6. **No v6
run exists in this repository's evidence.** Anyone who wants a citable Llama external
comparison needs to run v6 (or a later corrected version) and add its result here.

## Arms, design, and predeclared analysis (unchanged since v4)

Eight randomized blocks, four prompts per block, `h65-endpoint`, `accelerate-8gb`,
`airllm`, and `exact-8gb` as the compared arms. Per block, the geometric mean over the
four prompts of `arm_wall / h65-endpoint_wall`; across the eight blocks, geometric
mean and 95% paired-t interval (df 7) per arm; peak VRAM as measured; output token IDs
checked against `exact-8gb`. H6.5-endpoint is faster than an arm if that arm's
interval lies above 1.0, slower if below, otherwise indistinguishable. AirLLM's lower
peak VRAM is reported as a trade-off whatever the latency outcome, regardless of the
probe-confound status.

## What the v5 artifact still shows

Even discounted for a comparative claim, v5's own per-cell numbers are internally
consistent and can be read descriptively: H6.5-endpoint measured 7.97 GB peak VRAM at
roughly 42.6 s/token; Accelerate at the same 8 GB cap measured 8.42 GB at roughly
40.3 s/token; AirLLM measured 2.86 GB at roughly 47.4 s/token; the exact-residency
control at 8 GB measured 10.42 GB at roughly 91.0 s/token. Report these as context, not
as a confirmed comparative claim, and say why (the probe confound above) wherever they
are quoted.

## Scope

Descriptive-only evidence, not a primary or secondary confirmatory claim. See
[../evidence/h65-paper1/README.md](../evidence/h65-paper1/README.md) for the v5
artifact and how its numbers were checked.
