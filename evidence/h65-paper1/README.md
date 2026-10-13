# H6.5 Paper 1 evidence

Raw, immutable measurement artifacts behind the numbers in "Schedule-Aware Exact
Weight Placement for Large Language Models with Limited GPU Memory" (Oney Erge,
arXiv draft frozen 2026-09-26). Every JSON file here is copied byte-for-byte from
the original campaign output; `MANIFEST.sha256` records the exact bytes, and
those hashes match both each artifact's own `.validity.json` sidecar and the frozen
plan hashes recorded in the protocol docs under [`docs/`](../../docs). Nothing in
this folder was recomputed, filtered, or reweighted before being copied here.

**Run [`verify_paper_evidence.py`](verify_paper_evidence.py) yourself:**

```bash
python evidence/h65-paper1/verify_paper_evidence.py
```

It recomputes every number in the paper's Tables 7 to 11 (Llama on the RTX 5090;
Qwen3-14B and Gemma 2 27B on the RTX 3080 Laptop), the three repeats, the 12 GB
budget follow-up, the head-seed ablation, the four-token ablation, the
external-package claims, and the two earlier (pre-head-seed) method-history
confirmations, directly from the per-block data in this folder, and checks
every one against the value printed in the draft or documented in
[`docs/h65/README.md`](../../docs/h65/README.md#method-history). It prints `ALL CHECKS PASSED` or
a list of exactly which number did not match. This is the fastest way to check this
paper's headline claims without a GPU.

That checks the *measurements*. To check the *method* -- whether the H6.5 search
actually finds the published placement from its calibration traces, rather than the
plan having been hand-tuned -- see
[`llama-rtx5090/plan-rederivation/`](llama-rtx5090/plan-rederivation/README.md).
Also no GPU needed, though it does need one file this repository doesn't have (a
store metadata file, not model weights).

To rerun a measurement rather than recheck one, match the software environment
first: [`ENVIRONMENT.md`](ENVIRONMENT.md) records the exact package versions
behind the RTX 5090 numbers, with a pinned
[`constraints-rtx5090.txt`](constraints-rtx5090.txt).

## Index: paper claim -> artifact -> protocol

| Paper reference | Claim | Artifact | Protocol | Status |
|---|---|---|---|---|
| Abstract; Table 7, row "1" | D1: 8 pairs, 1.018x [0.884, 1.173] | [`llama-rtx5090/D1-confirmation-1/status.json`](llama-rtx5090/D1-confirmation-1/status.json) | [PAPER1_5090_LLAMA_H65_ENDPOINT_CONFIRMATION.md](../../docs/h65/protocols/PAPER1_5090_LLAMA_H65_ENDPOINT_CONFIRMATION.md) | confirmatory |
| Abstract; Table 7, row "2" | D2: 12 pairs, 1.239x [1.128, 1.361] | [`llama-rtx5090/D2-confirmation-2/status.json`](llama-rtx5090/D2-confirmation-2/status.json), prompts [`h65-confirmation2-prompts-20260911.json`](llama-rtx5090/D2-confirmation-2/h65-confirmation2-prompts-20260911.json) | [PROTOCOL-h65-endpoint-confirmation2-20260911.md](../../docs/h65/protocols/PROTOCOL-h65-endpoint-confirmation2-20260911.md) | confirmatory |
| Abstract; Table 7, row "All" | Pooled 20 pairs, 1.145x [1.052, 1.247]; ~23% lower peak GPU | derived from D1 + D2 above | pooling rule fixed in the D2 protocol before D2 ran | confirmatory |
| Section 5.3 / Table 8 | H6.5 vs. greedy-rule control: ~16% lower latency, ~20% lower peak GPU | [`llama-rtx5090/D4-greedy-mechanism/status.json`](llama-rtx5090/D4-greedy-mechanism/status.json) | [PROTOCOL-h65-mechanism-control-20260911.md](../../docs/h65/protocols/PROTOCOL-h65-mechanism-control-20260911.md) | secondary |
| Section 5.3 | Three repeats of D4 on new prompts, 35 blocks: ~18% lower latency than traffic, ~15% lower than greedy, 20-23% lower peak GPU, lower in every block | [`llama-rtx5090/D4-repeat-1/`](llama-rtx5090/D4-repeat-1/), [`llama-rtx5090/D4-repeat-2/`](llama-rtx5090/D4-repeat-2/), [`llama-rtx5090/D4-repeat-3/`](llama-rtx5090/D4-repeat-3/) | [PROTOCOL-d4-replication-20261005.md](../../docs/h65/protocols/PROTOCOL-d4-replication-20261005.md) with [amendment 1](../../docs/h65/protocols/AMENDMENT-1-resume-after-controller-stop-20261005.md), [PROTOCOL-d4r2-20261005.md](../../docs/h65/protocols/PROTOCOL-d4r2-20261005.md), [PROTOCOL-d4r3-20261005.md](../../docs/h65/protocols/PROTOCOL-d4r3-20261005.md) | repeat |
| Section 5.3 | D12: 12 GB GPU budget, 8 blocks: ~14% lower peak GPU than traffic; latency difference unresolved | [`llama-rtx5090/D12-budget-12gb/`](llama-rtx5090/D12-budget-12gb/) | [PROTOCOL-budget-followup-amendment1-20261004.md](../../docs/h65/protocols/PROTOCOL-budget-followup-amendment1-20261004.md) | follow-up |
| Section 6 | D11: head-seed ablation, 8 blocks: no clear additional latency benefit from refinement beyond the head-promoting seed | [`llama-rtx5090/D11-seed-ablation/`](llama-rtx5090/D11-seed-ablation/) | [PROTOCOL-P1-head-seed-ablation.md](../../docs/h65/protocols/PROTOCOL-P1-head-seed-ablation.md) with [amendment 1](../../docs/h65/protocols/PROTOCOL-P1-amendment1-pause-resume.md) | ablation |
| Not currently in the draft's body; appendix candidate | Four-token ablation: placement-only vs. traffic 1.041x | [`llama-rtx5090/four-token-ablation/status.json`](llama-rtx5090/four-token-ablation/status.json) | [PAPER1_5090_LLAMA_H65_4TOKEN_ABLATION.md](../../docs/h65/protocols/PAPER1_5090_LLAMA_H65_4TOKEN_ABLATION.md) | secondary |
| Not cited in the 2026-09-26 draft | Llama external-framework comparison (Accelerate/AirLLM/exact-8GB) | [`llama-rtx5090/external-comparison/status.json`](llama-rtx5090/external-comparison/status.json) (v5) | [PROTOCOL-external-comparison-v4-20260911.md](../../docs/h65/protocols/PROTOCOL-external-comparison-v4-20260911.md) | **descriptive only -- trips its own probe-confound gate; v6 correction never ran** |
| Table 9 | Qwen3-14B (D6) and Gemma 2 27B (D9) on the RTX 3080 Laptop: H6.5 vs. traffic, disk, and minimum-memory controls | [`laptop-rtx3080/D6-qwen-matched/`](laptop-rtx3080/D6-qwen-matched/), [`laptop-rtx3080/D9-gemma-matched/`](laptop-rtx3080/D9-gemma-matched/) | no separate frozen document; see [`laptop-rtx3080/`](laptop-rtx3080/README.md) | regulated exploratory |
| Section 5.4 / Figure 7 | External packages on the laptop (D7 Qwen, D10 Gemma) | [`laptop-rtx3080/D7-qwen-external/`](laptop-rtx3080/D7-qwen-external/), [`laptop-rtx3080/D10-gemma-external/`](laptop-rtx3080/D10-gemma-external/) | as above | regulated exploratory |
| Section 5.5 / Table 10 / Figure 8 | Qwen3-14B at 32 output tokens (D8) | [`laptop-rtx3080/D8-qwen-32token/`](laptop-rtx3080/D8-qwen-32token/) | as above | regulated exploratory |
| Section 5.5 / Table 11 | Qwen3-14B 32-token follow-ups: placement screen (D13) and confirmation on new prompts (D14) | [`laptop-rtx3080/D13-qwen-32token-screen/`](laptop-rtx3080/D13-qwen-32token-screen/), [`laptop-rtx3080/D14-qwen-32token-confirmation/`](laptop-rtx3080/D14-qwen-32token-confirmation/) | [PROTOCOL-h65-32token-calibration-followup-v3-20261003.md](../../docs/h65/protocols/PROTOCOL-h65-32token-calibration-followup-v3-20261003.md), [PROTOCOL-h65-32token-confirmation-20261003.md](../../docs/h65/protocols/PROTOCOL-h65-32token-confirmation-20261003.md) | regulated exploratory |
| Not cited; disclosed in [docs/h65/README.md](../../docs/h65/README.md#method-history) | Two earlier Llama confirmations (2026-08-31 8-block, 2026-09-02 12-pair) of the same causal claim, on a plan built before the 2026-09-09/09-10 planner fixes -- both completed and passed their own protocol's gates | [`llama-rtx5090/method-history/`](llama-rtx5090/method-history/) | [PAPER1_5090_LLAMA_CONFIRMATION.md](../../docs/h65/protocols/PAPER1_5090_LLAMA_CONFIRMATION.md), [PAPER1_5090_LLAMA_DIRECT_PAIR.md](../../docs/h65/protocols/PAPER1_5090_LLAMA_DIRECT_PAIR.md) | method history, not confirmatory for the current plan |

## Frozen plans

[`llama-rtx5090/frozen-plans/`](llama-rtx5090/frozen-plans/) holds the three
immutable placement plans every Llama study above replays against:

- `traffic-plan.json` -- traffic-density control, SHA-256 `926b0adb51275de35473a76ac3fc7dc7d82c58d3669027118f5bd433bb585563`
- `h65-endpoint-plan.json` -- H6.5 placement-only endpoint plan, SHA-256 `a31a0d4bbe2309998514c70182a04bce2024c2a385dc99bfd1c0585f0aa3ebb4`
- `h65-full-plan.json` -- full H6.5 endpoint/representation plan (four-token ablation only), SHA-256 `99d0869e6be6b0e2e55249a3a2a189d8ef24889581b951163dd39e0bbe762f67`
- `h65-plan-12v16r.json` and `traffic-plan-12v16r.json` -- the two D12 plans at the 12 GB GPU budget, SHA-256 `edfed766545ebb17a706a68642c07af4df2c7f5f19aa8666af67d095458ffdcf` and `8ef4129b34a83540c9eb330995edfbc558cd5e18d8ea50f11abd6894279d3c34`
- `head-seed-plan.json` -- the head-promoting seed placement, D11's fourth arm, SHA-256 `59ce58e74ab8ed4b7207f19d21abaf9c90b3c34d157732f69fcdf2e8092192e5`
- `knapsack-critical-path-profile.json` -- the measured-cost profile the greedy control ranks by (D4, its repeats and D11), SHA-256 `b88db4b579aea46dfd72572fb2a4222c5e03b25d64b317c1cd3043ec02ea083d`

The three D4 repeats and D11 reuse `traffic-plan.json` and `h65-endpoint-plan.json` unchanged.

These match the hashes printed in the protocol docs above; `verify_paper_evidence.py`
confirms the *result* artifacts are paired against the plan the protocol committed to
before any measurement ran. To independently re-derive the placement-only endpoint
plan itself from its calibration traces (no GPU needed, one input file missing --
see below), use [`llama-rtx5090/plan-rederivation/`](llama-rtx5090/plan-rederivation/README.md).

## What this folder deliberately leaves out

- The controller logs and per-cell source snapshots the original campaign wrote.
  Those live in the author's private `paper1-5090-bundle-20260911/`; most studies
  here carry only the frozen `status.json` result, which is what the paper's
  numbers are computed from. The exception is
  [`llama-rtx5090/plan-rederivation/`](llama-rtx5090/plan-rederivation/README.md),
  which does publish the three raw calibration traces behind the endpoint plan
  (small, event-DAGs only, no model weights) specifically so the placement
  decision itself, not only the measurements, can be checked.
- Anything graded `invalid`, `superseded`, or `diagnostic_only` by the campaign's own
  validity ledger. The full grading history (what ran, what failed, and why) is
  summarized in each protocol doc; ask the author if you want the discarded runs too.
  (The two `method-history/` runs above are the exception: both are genuinely
  *completed, passing* runs of an earlier plan, not invalid or superseded artifacts,
  so they are published rather than left out.)

## Re-deriving the laptop H6.5 plans

The laptop plans can be rebuilt from scratch, not just checked:
[`laptop-rtx3080/plan-rederivation/rederive_laptop_plans.py`](laptop-rtx3080/plan-rederivation/rederive_laptop_plans.py)
reruns the H6.5 search on a CPU from the published store manifests and
calibration traces, using the planner from the commit those plans were built
with, and reproduces both plans byte for byte. See
[`laptop-rtx3080/README.md`](laptop-rtx3080/README.md).
