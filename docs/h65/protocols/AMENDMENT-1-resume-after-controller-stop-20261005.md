# Amendment 1: resume after a controller stop (D4 replication)

Written 2026-10-05, after the stop and before the two measured cells of the interrupted block
were examined. The original protocol file is unchanged (its hash is checked at every cell).

## What happened

At 10:06:30 the controller stopped. It raised `PermissionError` while atomically replacing
`status.json`, which happens when another Windows process has the file open. The most likely
cause is the assistant's own monitoring scripts, which opened `status.json` to read progress.
This is unconfirmed. It was not a measurement failure, a gate failure, or a data problem.

State at the stop:
- **Blocks 1-7** (block indexes 0-6): complete, all eligible, 21 cells.
- **Block 8** (index 7, order K H T): its first two cells, greedy (K) and H6.5 (H), were
  measured and passed their gates. The T cell had not started.
- No other process was running, and the GPU lock was free.

## Rules for the resumption (fixed before the two block-8 cells were examined)

1. **Block 8 is retired.** It is excluded from every analysis, because its comparison is
   incomplete. Its two measured cells stay in the raw data and are reported as retired. T is
   not measured later for it. This follows P1's amendment 1, which retired an interrupted block.
2. **Blocks 9-12 run unchanged:** same prompts, orders, arms, plans, engine settings, 120 s
   storage idle before each cell, keep-awake and gates.
3. **No other cell is rerun,** and no replacement block is added. The analysis therefore uses
   up to **11 blocks** (1-7 and 9-12). Order balance is reported: each order appears twice
   except KHT, which appears once.
4. **Mechanical fix only.** The new controller process retries the atomic `status.json` write
   when it is blocked, and nothing else changes. The monitoring will read the controller log
   and the cell result files rather than opening `status.json` while the run is active.
5. **Time cap:** 150 more minutes from the resume, in addition to the 111 minutes already used.
6. **Predeclared analysis:** unchanged. It applies to the 11 eligible blocks, including the
   stall-excluded secondary and the 1-6 / 7-12 split. The split now has blocks 1-6 and 7-12
   minus the retired block 8.

## Disclosure

The interruption, its likely cause and the retired block are reported with the results. The
retired block's two cells are listed but not analyzed.
