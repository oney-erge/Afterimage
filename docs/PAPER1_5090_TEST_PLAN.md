# Paper 1 RTX 5090 test plan

Status: protocol outline. The first Llama pilot is exploratory and is used only
to estimate effect size and variance. It is not the confirmatory dataset.

## Evidence policy

All result consumers must consult `VALIDITY_LEDGER.json` and the immutable
artifact's adjacent `.validity.json` sidecar. Missing/unreviewed, invalid,
incomplete, diagnostic, and superseded artifacts are rejected. A
`regulated_pilot` artifact can support experiment design and visibly labelled
exploratory discussion, but not a primary confirmatory claim.

Every live paper cell must retain:

- the exact source snapshot and hashes;
- model-store manifest and measured H2D artifact hashes;
- a fresh process per block/method cell;
- cold-page-cache and thermal/throttling checks;
- a hard per-cell process-group timeout and resumable checkpoint;
- whole-process VRAM, not generation-only allocator memory;
- generated token IDs and exact equality against disk-only execution;
- matched logical VRAM/RAM budgets; and
- counterbalanced method position with paired block-level estimates.

Calibration prompts and evaluation prompts are disjoint. Plans are frozen
before held-out live evaluation.

## Stage 1: Llama-70B causal pilot

Budget: 8 GiB logical VRAM, 16 GiB RAM, 0.5 GiB VRAM safety margin. Workload:
one generated token for each of four held-out short prompts. Three distinct
all-disk calibration prompts feed the scheduler-aware H6.5 trace replay.

Four methods run in a four-position balanced order for four blocks:

1. disk-only exact streaming;
2. traffic-density placement with decoded RAM;
3. H6.5 placement-only (compressed RAM disabled); and
4. full H6.5 representation/tier planning.

This stage estimates paired variance, distinguishes H6.5 placement from
representation selection, and establishes whether the treatment plans actually
diverge. Expected runtime on the RTX 5090: 55--85 minutes.

## Stage 2: freeze the confirmation

After Stage 1, record the paired log-latency effect and block variance. Freeze a
new seed and independent block count before running more live cells. Use at
least eight balanced blocks unless a larger power-derived count is required.
Do not merge Stage 1 into the confirmatory estimate.

Proceed only if Stage 1 passes all scientific gates and the candidate differs
from traffic placement. A null or negative effect is still a valid result; do
not change prompts or budgets after inspecting it to manufacture a gain.

Expected analysis/protocol-freeze time: 10 minutes. Expected independent Llama
confirmation time: 1--3 hours, depending on the power-derived block count.

## Stage 3: Llama multi-token validation

Run four generated tokens per held-out prompt to test whether the observed
effect extends beyond TTFT/one-token execution. Compare traffic placement,
H6.5 placement-only, and full H6.5; retain disk-only exactness cells. Keep the
20-minute cell cap. Expected runtime: 2--3 hours.

## Stage 4: external Llama baselines

Compare the selected, guarded Afterimage deployment with HF Accelerate and
AirLLM using the same prompts, token count, cache regime, process isolation, and
whole-process VRAM measurement. Report failures/timeouts as compatibility or
bounded-runtime outcomes, never as invented latency values. Expected runtime:
1--2.25 hours with 15--20 minute hard cell caps.

## Stage 5: budget sensitivity

Screen tighter and more relaxed VRAM/RAM points offline first. Run live cells
only where the frozen H6.5 candidate differs from traffic placement. This tests
the intended claim that H6.5 matters under heterogeneous memory pressure and
that its advantage should shrink when ordinary placement is already adequate.
Expected live runtime: 45--90 minutes.

## Stage 6: Qwen-14B 5090 disambiguation

If the Llama result conflicts with the RTX 3080 Qwen result, repeat the same
four-method causal matrix for Qwen on the RTX 5090 at 4 GiB VRAM / 8 GiB RAM.
This separates model topology from machine/storage effects. Expected runtime:
20--35 minutes.

If Paper 1 makes a full-system speculation claim, also run a crossed Qwen
factorial: H6.5 off/on by speculation off/on on realistic generation prompts.
This prevents gains from non-novel speculation from being attributed to H6.5.
Expected runtime: 1.5--2.5 hours.

Gemma and the main Qwen hardware replication remain assigned to the RTX 3080;
duplicating them on the RTX 5090 is not required unless the cross-machine
results disagree.
