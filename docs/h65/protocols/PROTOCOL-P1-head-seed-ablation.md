# Frozen protocol: P1 head-seed ablation (four arms, eight blocks, one token)

Frozen 2026-09-12 after P0 and before any P1 measurement. The SHA-256 of this file is
recorded in the run's `status.json` at prepare. Governing plan:
`paper/draft-paper-2026-09-12/revised/PRIORITY-RUN-PLAN.md`, section P1.

## Question

Does the complete H6.5 search improve live latency over its own best endpoint-promoting
initial seed? P0 established that the chosen seed differs from the final plan in 107
tensors, while predicting a difference of well under 1%. Only a live measurement can
decide.

## Arms (fixed before timing)

| Arm | Assignment | Frozen input |
|---|---|---|
| T | archived traffic plan | pilot `traffic-plan.json`, SHA-256 `926b0adb…` |
| K | archived measured-cost greedy placement | `profiled_knapsack` with profile `b88db4b5…` from mechanism control v2; its runtime tier fingerprint must equal that run's |
| S | best valid initial endpoint seed from P0 | `seed-endpoint-vram-first-plan.json`, SHA-256 `59ce58e7…` |
| H | archived final H6.5 endpoint plan | pilot `h65-endpoint-plan.json`, SHA-256 `a31a0d4b…` |

Plans are frozen files; nothing is replanned per block. S was produced by the pilot's
planner at commit `1c6a02cf`; P1 executes on the runtime at `637bbd55`. Between those
commits only the prompt suite, two docs and a test changed; no runtime or planner code.

## Engine contract, identical for every arm

Llama-3.3-70B, batch one, greedy decoding, full output head, no speculative draft, no
quantization. 8 GB logical GPU budget, 10 GB allocator cap, 12 GB whole-process peak
threshold, 16 GB persistent host budget, 0.5 GB safety margin, `decode_slice_elems`
4,194,304 (elements), prefetch depth 2, per-blob storage reads, `reuse_decode_tables`
true, decoded RAM tier, static execution. One warm-up token, 10 s cooldown with GPU at
or below 75 °C, cold page cache per request, fresh process per cell.

## Prompts

Eight new prompts in `p1-prompts.json`, one per block, written before any P1 timing and
without seeing method performance. Prepare verifies that no ID or text matches any
calibration, pilot, confirmation, confirmation-2, mechanism-control or external prompt,
and that every rendered prompt has 72-110 input tokens. Exact rendered text, token IDs,
lengths and tokenizer provenance are saved before timing. Each request generates exactly
one output token.

## Order (four-arm Williams design, used twice)

| Block | Order |
|---|---|
| 1 | T K H S |
| 2 | K S T H |
| 3 | S H K T |
| 4 | H T S K |
| 5 | S H K T |
| 6 | H T S K |
| 7 | T K H S |
| 8 | K S T H |

## Eligibility and stop rules

- A cell is measured under the archived gates: worker exit, row set, 12 GB whole-process peak, cold page cache, thermal and power windows, resolved engine settings, one-token output and warm-up.
- A thermal- or power-window failure, a failed cell, or differing output tokens makes that **block** ineligible for the paired comparison. The run continues to the next cell; no replacement blocks are added.
- The run **stops** if a frozen input changes, a T/S/H runtime plan does not match its frozen file, or the K tier fingerprint differs from mechanism control v2, because old plans could no longer be treated as equivalent.
- P1 has a 3-hour limit from the end of P0, including setup, cooling, and failed cells. At the deadline the run stops and all partial rows are kept.
- No sample extension after seeing results.

## Predeclared analysis

- **Primary:** S/H request-latency ratio (`S_wall / H_wall`) over eligible paired blocks. Geometric mean and Student-t interval on block log ratios (df = eligible blocks − 1). Every pair is shown.
- **Context, not alternative endpoints:** T/H and K/H latency ratios.
- **Alongside:** whole-process and torch peak GPU memory, host RSS, initialization, output tokens, read bytes and calls, I/O and prefetch wait, thermal and power summaries, eligibility.
- A 5% latency reduction is a practical-effect reference, not a significance threshold or power calculation.

Reading:

- Interval lies above 1.0: H improves on S; supports an incremental search benefit for this workload.
- Interval includes 1.0: S and H are indistinguishable; attribute the benefit to the selected placement and do not claim search was necessary.
- Interval lies below 1.0: S wins; report it and recommend the simpler seed for this measured case.
- Differing tokens in any block: no performance interpretation for that block until diagnosed.
