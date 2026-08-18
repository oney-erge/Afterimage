"""llama.cpp partial-offload baseline (also the base ATSInfer, LITERATURE.md
#8, was built on top of).

NOT EXECUTED OR VERIFIED IN THIS DEVELOPMENT ENVIRONMENT: shells out to a
`llama-cli` / `llama-server` binary that must be built separately and is not
present here (IMPLEMENTATION_STATUS.md). This wraps the documented
`--n-gpu-layers` partial-offload flag; it has not been run.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time


def is_available(binary: str = "llama-cli") -> bool:
    return shutil.which(binary) is not None


def run_llamacpp_baseline(binary: str, model_path: str, prompt: str, max_new_tokens: int,
                           n_gpu_layers: int, timeout_seconds: int = 600) -> dict:
    if not is_available(binary):
        raise FileNotFoundError(
            f"'{binary}' not found on PATH -- build llama.cpp on the real "
            f"benchmarking rig (see IMPLEMENTATION_STATUS.md)."
        )

    cmd = [
        binary, "-m", model_path, "-p", prompt,
        "-n", str(max_new_tokens), "-ngl", str(n_gpu_layers),
        "--no-display-prompt",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    dt = time.perf_counter() - t0

    match = re.search(r"([\d.]+)\s*tokens per second", proc.stdout + proc.stderr)
    reported_tps = float(match.group(1)) if match else None

    return {
        "wall_seconds": dt,
        "tokens_per_second_reported": reported_tps,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
    }
