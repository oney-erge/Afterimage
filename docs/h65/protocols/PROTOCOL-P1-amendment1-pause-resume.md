# P1 protocol amendment 1: pause and resume

Written 2026-09-12 after the user paused P1 and before any further P1 measurement.
Its SHA-256, and that of the resume driver, are added to the run's frozen inputs when the
run is prepared for resume. Everything in `PROTOCOL-P1-head-seed-ablation.md` not changed
below still applies.

## What happened

The user paused P1 to continue the next day. Block 1 (`seedabl-explain-auroras`, order
T K H S) was complete. Block 2 (`seedabl-arithmetic-library`, order K S T H) had
finished K, S and T; its H cell was stopped mid-run. No P1 result had been reported, so
the pause was not outcome-driven.

## Rules

1. **Complete blocks are kept** exactly as measured, including their eligibility.
2. **Interrupted blocks are retired.** Their finished cells are moved to `cells-abandoned/<pause-tag>/` and listed under `abandoned_cells` in `status.json`. They are preserved but never paired, pooled or reported as results: a block's four arms are compared only within one uninterrupted run.
3. **An interrupted block is rerun in full** on resume, with its frozen prompt and its frozen arm order.
4. **No sample change.** The run still has exactly eight blocks with the frozen prompts and orders. Complete but ineligible blocks are not replaced, and there is no extension.
5. **Budget.** The three-hour P1 limit counts time from the end of P0 to each pause, plus time from each resume launch. Paused time is not counted. At each pause the remaining budget is the session deadline minus the pause time. The pause time is taken from the controller's own checkpoint; if the controller was stopped externally, it is the latest heartbeat, controller-log, cell-file or session-launch time, which slightly overstates usage. For the first pause this is 16:23:06 CST (the stopped cell's log), so 40.5 of 180 minutes were used and 139.5 remain.
6. **Sessions are disclosed.** Every block result records the session it ran in, and results are reported by session. All ratios are within block, so a between-session change cannot bias them, but any session difference in absolute times is reported.
7. **Refusals.** Resume is refused if any frozen input changed, if the runtime is not the clean frozen commit, if a P1 process is still running, or if less than 15 minutes of budget remains. Runs that stopped on a failed compatibility check, or that reached their deadline, are not resumable.
8. **Further pauses.** A resumed controller checkpoints itself as `paused` on SIGTERM. The same retire-and-rerun procedure applies to any later pause or crash.

## Implementation

`p1_head_seed_ablation_resume_20260912.py` imports the frozen P1 driver unchanged and
reuses its engine overrides, runtime-plan and knapsack-fingerprint checks, eligibility
rules and statistics. Its per-block body reproduces the driver's without behavioural
change; the only additions are session labels and pause handling.
