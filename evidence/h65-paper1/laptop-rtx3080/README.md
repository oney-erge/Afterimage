# RTX 3080 Laptop studies (Qwen3-14B, Gemma 2 27B)

The raw artifacts behind the paper's Tables 9 and 10 and Figures 7 and 8, copied
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

In the artifacts, `simple-v4-r8` is the paper's traffic control, `disk-frozen`
its disk control, `exact-min` its minimum-memory control, and `h65-selected` its
H6.5 arm.

**Evidence grade.** These are "regulated exploratory" runs, as the paper labels
them: complete, cache- and thermal-gated, paired and counterbalanced, but not
governed by a separately frozen protocol document the way the RTX 5090
confirmations are. Every one of them ran from a clean tree at public commit
`6c37700`, is marked paper-eligible by its own runner, and records the software
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
