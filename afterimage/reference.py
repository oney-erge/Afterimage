"""One measured reference point, shared by everything that extrapolates
from it.

Qwen3-14B on an RTX 3080 Laptop (8 GB), WSL2/CUDA, cold page cache -- the
README's benchmark table. Every other model-size or hardware estimate in
this project (the web UI's /api/capability card, `afterimage doctor`'s
disk-speed translation) scales from these per-billion-parameter ratios
rather than restating its own copy, so there is exactly one place that can
drift out of sync with the actual measured numbers in results/.
"""
from __future__ import annotations

MEASURED_REFERENCE = {
    "model": "Qwen/Qwen3-14B",
    "params_b": 14.0,
    "bf16_gb_per_b_params": 29.536 / 14.0,
    "compressed_gb_per_b_params": 20.328 / 14.0,
    "min_memory_s_per_token_per_b": 32.514 / 14.0,
    "min_memory_vram_gb": 1.723,
    "fast_s_per_token_per_b": 9.150 / 14.0,
    "fast_vram_floor_gb": 3.813,
    # exact-min's gb_read_per_token from results/2026-08-21_bounded_
    # qwen3-14b_rtx3080_run1.json: 18.705 GB read of a 20.328 GB store --
    # close enough to "the whole store" that "read the whole compressed
    # store every token" is a fair worst-case approximation for the
    # min-memory profile specifically (balanced/fast read far less).
    "min_memory_store_fraction_read_per_token": 18.705480996 / 20.328,
}
