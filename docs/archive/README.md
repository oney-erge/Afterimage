# Archive

Superseded planning, hypothesis, and interim-results documents. Kept for
traceability — nothing here is deleted, and nothing here is deleted for
being wrong; most of it is exactly right for the moment it was written and
overtaken by later, more rigorous measurement.

**Do not treat anything in this folder as current.** For the current,
accurate picture, start with [the top-level README](../../README.md), then:

- **[HOW_IT_WORKS.md](../HOW_IT_WORKS.md)** — what the engine does, method by
  method, next to AirLLM.
- **[BOUNDED_RESEARCH_REPORT_2026-08-21.md](../BOUNDED_RESEARCH_REPORT_2026-08-21.md)** —
  the current multi-prompt evidence and H0-H8 verdicts.
- **[RESULTS_LOG.md](../RESULTS_LOG.md)** — the append-only evidence ledger
  every number in HOW_IT_WORKS.md traces back to, including corrections.
- **[HYPOTHESIS_LINEAGE.md](../HYPOTHESIS_LINEAGE.md)** — every hypothesis
  H0-H15, its source, and its current verdict, in one table.
- **[LITERATURE.md](../LITERATURE.md)** — the research survey.

## What's here, roughly chronological

**Phase 0 — the subspace activation cache (abandoned).** `HYPOTHESIS.md`,
`EXECUTION_PLAN.md`, `IMPLEMENTATION_PLAN.md`, `VALIDATION_PLAN.md`,
`IMPLEMENTATION_STATUS.md`, `PHASE0_RESULTS.md`, plus the raw
`phase0_real_results.json`. A different idea than the current engine:
caching linear-layer outputs for previously-seen activation directions.
Real math, 67 tests, and a real-model measurement that came in 250-450x
above the success threshold. Correctly killed per its own pre-registered
gate. The code (`basis.py`, `gate.py`, `sketch.py`, etc.) still lives in
`afterimage/`, marked archived in its own docstrings.

**Early streaming-engine results.** `BASELINE_RESULTS.md`,
`CAPACITY_RESULTS.md`, `SHOOTOUT_RESULTS.md`, `ollama_capacity_results.json`,
`shootout.json`, `ENGINE_EVALUATION.md`, `STREAMING_ENGINE_STATUS.md`,
`IMPROVEMENT_PLAN.md` — the layer-streaming engine's early measurements,
before three-tier residency, speculative decoding, or the chunked head
existed.

**`MASTER_PLAN.md`** — an overarching roadmap and competitive landscape
write-up. Its headline VRAM/speed table is now **superseded and known
inaccurate** — see RESULTS_LOG.md's "CORRECTION" section for why the
matched-VRAM comparison it was built on was wrong, and HOW_IT_WORKS.md for
the corrected numbers. The competitive-landscape section (DFloat11, ZipServ,
NeuZip, FlexGen, AirLLM) is still a reasonable orientation read.

**`PROPOSAL.md`, `PROPOSAL_ADAPTIVE.md`, `ADAPTIVE_TEST_PLAN.md`** — the
hypothesis-and-test-plan documents for the compression/residency levers
(labelled H1-H4 in that document's own numbering -- **not** the same H1-H4
in the current `docs/RESEARCH_METHODS.md` registry, which restarted the
numbering for a different set of hypotheses; code comments that cite one of
these say so explicitly) and the adaptive-speculation mechanisms
(self-drafting, bandit-tuned k, draft-aware VRAM planning). Every hypothesis
in them was actually run; the outcomes are in RESULTS_LOG.md and summarized
in HOW_IT_WORKS.md. Useful if you want the reasoning and kill-criteria
behind a result, not just the number.

**`LOSSLESS_ENGINE.md`** — an earlier design doc for the compression +
streaming approach, written before most of it was built and measured.
