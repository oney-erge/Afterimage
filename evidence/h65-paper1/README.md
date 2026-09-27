# H6.5 Paper 1 evidence

Raw, immutable measurement artifacts behind the numbers in "Schedule-Aware Exact
Weight Placement for Large Language Models with Limited GPU Memory" (Oney Erge,
arXiv draft frozen 2026-09-26). Every JSON file here is copied byte-for-byte from
the original private campaign output; `MANIFEST.sha256` records the exact bytes, and
those hashes match both each artifact's own `.validity.json` sidecar and the frozen
plan hashes recorded in the protocol docs under [`docs/`](../../docs). Nothing in
this folder was recomputed, filtered, or reweighted before being copied here.

**Run [`verify_llama_confirmations.py`](verify_llama_confirmations.py) yourself:**

```bash
python evidence/h65-paper1/verify_llama_confirmations.py
```

It recomputes the paper's Table 7, Table 8, and pooled-20-pair numbers directly from
the paired per-block data in this folder and checks every one against the value
printed in the draft. It prints `ALL CHECKS PASSED` or a list of exactly which number
did not match. This is the fastest way to check this paper's headline claims without
a GPU.

## Index: paper claim -> artifact -> protocol

| Paper reference | Claim | Artifact | Protocol | Status |
|---|---|---|---|---|
| Abstract; Table 7, row "1" | D1: 8 pairs, 1.018x [0.884, 1.173] | [`llama-rtx5090/D1-confirmation-1/status.json`](llama-rtx5090/D1-confirmation-1/status.json) | [PAPER1_5090_LLAMA_H65_ENDPOINT_CONFIRMATION.md](../../docs/PAPER1_5090_LLAMA_H65_ENDPOINT_CONFIRMATION.md) | confirmatory |
| Abstract; Table 7, row "2" | D2: 12 pairs, 1.239x [1.128, 1.361] | [`llama-rtx5090/D2-confirmation-2/status.json`](llama-rtx5090/D2-confirmation-2/status.json) | [PAPER1_5090_LLAMA_H65_ENDPOINT_CONFIRMATION2.md](../../docs/PAPER1_5090_LLAMA_H65_ENDPOINT_CONFIRMATION2.md) | confirmatory |
| Abstract; Table 7, row "All" | Pooled 20 pairs, 1.145x [1.052, 1.247]; ~23% lower peak GPU | derived from D1 + D2 above | pooling rule fixed in the D2 protocol before D2 ran | confirmatory |
| Section 5.3 / Table 8 | H6.5 vs. greedy-rule control: ~16% lower latency, ~20% lower peak GPU | [`llama-rtx5090/D4-greedy-mechanism/status.json`](llama-rtx5090/D4-greedy-mechanism/status.json) | [PAPER1_5090_LLAMA_H65_MECHANISM_CONTROL.md](../../docs/PAPER1_5090_LLAMA_H65_MECHANISM_CONTROL.md) | secondary |
| Not currently in the draft's body; appendix candidate | Four-token ablation: placement-only vs. traffic 1.041x | [`llama-rtx5090/four-token-ablation/status.json`](llama-rtx5090/four-token-ablation/status.json) | [PAPER1_5090_LLAMA_H65_4TOKEN_ABLATION.md](../../docs/PAPER1_5090_LLAMA_H65_4TOKEN_ABLATION.md) | secondary |
| Not cited in the 2026-09-26 draft | Llama external-framework comparison (Accelerate/AirLLM/exact-8GB) | [`llama-rtx5090/external-comparison/status.json`](llama-rtx5090/external-comparison/status.json) (v5) | [PAPER1_5090_LLAMA_H65_EXTERNAL_COMPARISON.md](../../docs/PAPER1_5090_LLAMA_H65_EXTERNAL_COMPARISON.md) | **descriptive only -- trips its own probe-confound gate; v6 correction never ran** |
| Section 5.4 / Figure 7, Table 9 | Qwen3-14B and Gemma 2 27B on RTX 3080, one-token, "regulated exploratory" | not yet located with certainty -- see [`laptop-rtx3080/UNRESOLVED.md`](laptop-rtx3080/UNRESOLVED.md) | none published yet | **unresolved** |

## Frozen plans

[`llama-rtx5090/frozen-plans/`](llama-rtx5090/frozen-plans/) holds the three
immutable placement plans every Llama study above replays against:

- `traffic-plan.json` -- traffic-density control, SHA-256 `926b0adb51275de35473a76ac3fc7dc7d82c58d3669027118f5bd433bb585563`
- `h65-endpoint-plan.json` -- H6.5 placement-only endpoint plan, SHA-256 `a31a0d4bbe2309998514c70182a04bce2024c2a385dc99bfd1c0585f0aa3ebb4`
- `h65-full-plan.json` -- full H6.5 endpoint/representation plan (four-token ablation only), SHA-256 `99d0869e6be6b0e2e55249a3a2a189d8ef24889581b951163dd39e0bbe762f67`

These match the hashes printed in the protocol docs above; `verify_llama_confirmations.py`
does not independently re-derive a plan from calibration traces (that requires the
original model checkpoint and GPU), but it does confirm the *result* artifacts are
paired against the plan the protocol committed to before any measurement ran.

## What this folder deliberately leaves out

- The multi-gigabyte raw calibration traces, controller logs, and per-cell source
  snapshots the original campaign wrote. Those live in the author's private
  `paper1-5090-bundle-20260911/`; this folder carries only the frozen `status.json`
  result for each study, which is what the paper's numbers are computed from.
- Anything graded `invalid`, `superseded`, or `diagnostic_only` by the campaign's own
  validity ledger. The full grading history (what ran, what failed, and why) is
  summarized in each protocol doc; ask the author if you want the discarded runs too.
- The RTX 3080 laptop (Qwen3-14B, Gemma 2 27B) result files -- see the unresolved
  note below.

## Known open item: laptop (RTX 3080) evidence

See [`laptop-rtx3080/UNRESOLVED.md`](laptop-rtx3080/UNRESOLVED.md). Short version:
during this audit, the numeric search that located every Llama artifact above could
not identify a single, unambiguous source directory for the paper's Table 9
(Qwen3-14B, Gemma 2 27B on RTX 3080). The paper's own numbers point most strongly at
a `paper-h65-gemma-practical-20260909-r2b` / `paper-h65-qwen-practical-20260909-r2`
pair of campaign directories (both post-date the private evidence bundle's own
`corpus-3080/` snapshot, which the bundle's README says has **no Gemma runs at all**
-- a direct contradiction with the paper that needs the author's confirmation before
publishing this half of the evidence).
