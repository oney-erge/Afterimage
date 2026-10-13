# RTX 3080 Laptop studies (Qwen3-14B, Gemma 2 27B)

The raw artifacts behind the paper's Tables 9 to 11 and Figures 7 and 8, copied
byte for byte from the campaign output on the laptop that ran them. These are
the exact files the paper's own analysis code reads. `verify_paper_evidence.py`
(one folder up) recomputes every published number from them.

| Study | Paper | Artifact |
|---|---|---|
| D6 | Table 9, Qwen3-14B: H6.5 vs. traffic, disk, and minimum-memory controls; 8 blocks x 4 prompts, 1 token | [`D6-qwen-matched/`](D6-qwen-matched/) |
| D7 | Figure 7, Qwen3-14B external packages (AirLLM, Accelerate, DeepSpeed) | [`D7-qwen-external/`](D7-qwen-external/) |
| D8 | Table 10, Figure 8, Section 5.5: Qwen3-14B at 32 output tokens; 3 blocks x 4 prompts | [`D8-qwen-32token/`](D8-qwen-32token/) |
| D9 | Table 9, Gemma 2 27B, same design as D6 | [`D9-gemma-matched/`](D9-gemma-matched/) |
| D10 | Figure 7, Gemma 2 27B external packages | [`D10-gemma-external/`](D10-gemma-external/) |
| D13 | Table 11, Qwen3-14B at 32 output tokens: placement screen, one- or 32-token calibration with 0 or 8 GB of host-resident weights; 3 blocks x 4 prompts | [`D13-qwen-32token-screen/`](D13-qwen-32token-screen/) |
| D14 | Table 11, Figure 8: confirmation of the 0 GB host placement on new prompts; 7 blocks x 4 prompts | [`D14-qwen-32token-confirmation/`](D14-qwen-32token-confirmation/) |

In the artifacts, `simple-v4-r8` is the paper's traffic control, `disk-frozen`
its disk control, `exact-min` its minimum-memory control, and `h65-selected` its
H6.5 arm. In D13 and D14, `simple-v4-r0` is the traffic control with no
host-resident weights, and `h65-cal1-ram8`, `h65-cal32-ram8`, `h65-cal1-ram0`
and `h65-cal32-ram0` are H6.5 placements calibrated on one or 32 tokens with 8
or 0 GB of host-resident weights.

**Evidence grade.** These are "regulated exploratory" runs, as the paper labels
them: complete, cache- and thermal-gated, paired and counterbalanced, but not
governed by a separately frozen protocol document the way the RTX 5090
confirmations are. D6 to D10 each ran from a clean tree at public commit
`6c37700`. D13 and D14 ran later, each from a clean tree and each under a
protocol frozen before measurement
([D13](../../../docs/h65/protocols/PROTOCOL-h65-32token-calibration-followup-v3-20261003.md),
[D14](../../../docs/h65/protocols/PROTOCOL-h65-32token-confirmation-20261003.md)).
Every study is marked paper-eligible by its own runner and records the software
environment in [`../ENVIRONMENT.md`](../ENVIRONMENT.md).

**Planner version.** The H6.5 plans here were built by the planner at commit
`6c37700`, before two later planner changes (the 2026-09-09 candidate-retention
fix and the 2026-09-10 output-head search seed) that the RTX 5090 plans used.
See [`docs/h65/README.md`](../../../docs/h65/README.md#which-planner-built-which-plan).

## Frozen plans and re-deriving them

[`frozen-plans/`](frozen-plans/) holds the plans the timed runs loaded; each
matches the SHA-256 its run recorded. [`plan-rederivation/`](plan-rederivation/)
holds everything the H6.5 search consumed: each model's store manifest, its
three calibration traces, and the measured pageable H2D rate. Then:

```bash
python evidence/h65-paper1/laptop-rtx3080/plan-rederivation/rederive_laptop_plans.py
```

reruns the search on a CPU (about 20 seconds, no model weights, no GPU) with the
planner from the recorded commit, and checks both H6.5 plans byte for byte
against the frozen ones. CI runs it on every change.
