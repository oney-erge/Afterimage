# Amendment 1 to the H6.5 mechanism-control protocol (infrastructure only)

Written 2026-09-11 after v1 stopped and before any v2 measurement. Its SHA-256 is
recorded in the v2 run's `status.json` at prepare time.

## What happened

v1 (`h65-mechanism-control-knapsack-llama70b-8logical-10allocator-12physical-rtx5090-20260911-v1`)
completed its first two cells (`traffic`, `h65-endpoint`) and stopped on its first
`knapsack` cell with `placement_policy='profiled_knapsack' requires critical_path_profile`.
The cell config contained the correct profile path. The frozen matrix worker calls
`run_afterimage(..., critical_profile=None)`, and `engine_for` replaces the configured
profile path with that argument whenever the placement policy needs a profile. Every
`profiled_knapsack` cell run through this worker fails the same way, which is also why
the September 9 greedy control never produced a comparison.

## Change

v2 runs every cell through `h65_worker_profile_passthrough_20260911.py`. It loads the
frozen worker unchanged and forwards the cell's own `critical_path_profile` when the
worker passes none. Cells without a profile (`traffic`, `h65-endpoint`) take exactly
the frozen code path. Preflight verifies the forwarding before any GPU cell runs.

## Unchanged

The question, arms, frozen inputs, engine settings, memory contract, six blocks and
their arm orders, gates and predeclared analysis are exactly those of
`PROTOCOL-h65-mechanism-control-20260911.md`. v1 is preserved as invalid; none of its
cells are reused.
