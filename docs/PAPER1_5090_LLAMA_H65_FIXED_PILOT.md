# RTX 5090 Llama H6.5 corrected-placement viability pilot

Status: frozen before calibration or live timing. This is a regulated pilot,
not primary confirmation, and is not pooled with historical measurements.

## Question and scope

Does the corrected H6.5 placement search produce a reproducible latency benefit
over Afterimage traffic-density placement near the earlier positive Llama-70B
memory-pressure region, under corrected physical-memory accounting?

The claim is limited to exact, cold-cache, one-token request latency on this
RTX 5090. Speculation is disabled. Full H6.5 is checked offline for search-space
consistency but is not a live arm. HF Accelerate and AirLLM are not part of this
planner-isolation pilot.

## Frozen setting

- Model: `meta-llama/Llama-3.3-70B-Instruct` and the existing exact store.
- GPU ceiling: 9.0 decimal GB sampled whole-process device-memory delta.
- Logical weight RAM: 16.0 decimal GB.
- Additional planner safety allowance: 0.5 decimal GB.
- Decoder-table reuse: enabled.
- Decode slice: 4,194,304 elements.
- Fixed prefetch depth: 2; per-blob storage reads.
- Full exact output head; batch size one; no speculation.
- One untimed warm-up token and one timed output token per fresh process.
- Page cache dropped and CUDA allocator cache emptied before each timed request.
- Pageable/blocking RTX 5090 H2D calibration artifact is frozen by hash.

Nine decimal GB is fixed from the historical memory record before this run:
correcting the old MiB conversion gives an approximately 8.406 GB traffic peak,
leaving about 0.594 GB rather than silently accepting an over-cap nominal 8 GB
measurement. If either arm exceeds 9.0 GB, the pilot stops and the failure is
reported; the budget is not changed after timing is observed.

## Fresh calibration and plan selection

Collect three separate causal disk traces from these calibration prompts:

1. `summary-photosynthesis`
2. `logic-glippets`
3. `copy-nonce`

Collect an execution-matched decoded-RAM preparation profile using
`json-status` and `code-squares`. Calibration prompts are disjoint from the
live pilot prompts. The first two disk traces are training traces; the third is
a held-out calibration trace. It may accept or reject the training-selected
candidate, but it may not select another candidate.

Run corrected placement-only search with seed `202609093`, 256 search
iterations, nonnegative required improvement on every training and held-out
calibration trace, and the existing requirement for live validation. Run full
search offline using placement's selected candidate as an incumbent. Full must
not return a worse offline objective than that incumbent. Freeze every plan,
profile, trace, source file, configuration and hash before live timing.

If placement does not differ physically from traffic or fails replay, stop
without live performance cells. This is a valid negative pilot outcome.

## Live comparison

The four paired blocks use one prompt each and the following fixed adjacent
order. Each method/prompt is a fresh process.

| Block | Prompt | Order |
|---|---|---|
| 0 | `fact-gold` | traffic, H6.5 |
| 1 | `arithmetic-17x6` | H6.5, traffic |
| 2 | `code-square` | traffic, H6.5 |
| 3 | `retrieval-7319` | H6.5, traffic |

Each cell has a six-minute hard process-group timeout. Calibration plus live
timing has a 50-minute total cap. No automatic retry, optional stopping, prompt
replacement, outlier deletion or budget change is allowed. A technical failure
remains in the artifact; any separately authorized rerun must be labeled and
must preserve the original attempt.

## Required gates and outcome

All eight live cells must finish with captured zero exits, exact paired token
IDs, successful cache drops, one completed warm-up token, observable clean
thermal windows, frozen runtime plan equality, expected tier fingerprints and
read counters, and valid whole-process memory measurements no greater than
9.0 GB. Initialization and warm-up are recorded but excluded from request
latency. I/O durations may overlap and are not summed as wall time.

Primary pilot estimand: geometric mean of the four paired block ratios
`traffic_seconds / h65_seconds`. Report all four ratios and the number favoring
H6.5 regardless of direction.

This pilot justifies an independent confirmation only when:

1. all procedural and measurement gates pass;
2. the geometric-mean speedup is at least 1.03x; and
3. H6.5 is faster in at least three of four blocks.

The guarded deployment rule is reported separately: it clears live validation
only if every eligible paired block improves by at least the planner's frozen
1% threshold. The pilot advancement rule does not silently weaken that guard.

No additional model, broader budget sweep, multi-token run, external-framework
comparison or confirmation is automatically launched by this protocol.
