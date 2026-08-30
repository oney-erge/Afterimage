#!/usr/bin/env python3
"""Run a diverse, cold-cache comparison under a hard wall-time budget.

This is an exploratory screening run, not the five-repeat confirmatory protocol
in docs/RESEARCH_METHODS.md.  It is designed to answer the practical question
"which hypotheses deserve the expensive run?" in roughly 30-60 minutes on the
reference RTX 3080 Laptop GPU.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from afterimage.bench.memory import MemoryProbe
from afterimage.bench.prompt_suite import (
    PROMPT_SUITE_VERSION,
    PromptCase,
    prompt_cases,
    render_chat_prompt,
)
from afterimage.runtime.config import EngineConfig


MODEL = "Qwen/Qwen3-14B"
DRAFT_MODEL = "Qwen/Qwen3-0.6B"
STORE = "/root/afterimage/store_14b"
# DFloat11 (arXiv:2504.11651, NeurIPS'25) ships pre-compressed checkpoints
# under its own Hub org rather than compressing an arbitrary --model at
# runtime; this is the real DFloat11/Qwen3-14B-DF11 repo, verified to exist
# and to match the canonical MODEL parameter-for-parameter, not guessed.
DFLOAT11_MODEL = "DFloat11/Qwen3-14B-DF11"
DEEPSPEED_OFFLOAD_DIR = os.environ.get(
    "AFTERIMAGE_DEEPSPEED_OFFLOAD_DIR", "/root/afterimage/deepspeed_offload_14b")
H6_PLAN_STATE = os.environ.get(
    "AFTERIMAGE_H6_PLAN_STATE", "/root/afterimage/plans/qwen3-14b-h6-v4-r8.json")
DISK_PLAN_STATE = os.environ.get(
    "AFTERIMAGE_DISK_PLAN_STATE", "/root/afterimage/plans/qwen3-14b-disk-v4-r8.json")

# Inter-cell cooldown, set once from --cooldown-seconds / --cooldown-max-temp-c.
# Module-level because every timed cell in every runner must observe the same
# policy; a per-runner argument would silently diverge between methods, which
# is exactly the kind of asymmetry that makes a comparison invalid.
COOLDOWN_SECONDS = 0.0
COOLDOWN_MAX_TEMPERATURE_C: float | None = None


@dataclasses.dataclass(frozen=True)
class Method:
    id: str
    title: str
    kind: str
    overrides: dict
    exactness: str
    estimated_s_per_token: float


def _installed_airllm_title() -> str:
    try:
        from importlib.metadata import version
        return "AirLLM %s" % version("airllm")
    except Exception:
        return "AirLLM (version unknown -- not importable)"


def _installed_accelerate_title() -> str:
    try:
        from importlib.metadata import version
        return "Hugging Face Accelerate %s" % version("accelerate")
    except Exception:
        return "Hugging Face Accelerate (version unknown -- not importable)"


def _installed_dfloat11_title() -> str:
    try:
        from importlib.metadata import version
        return "DFloat11 %s" % version("dfloat11")
    except Exception:
        return "DFloat11 (version unknown -- not importable)"


def _installed_deepspeed_title() -> str:
    try:
        from importlib.metadata import version
        return "DeepSpeed ZeRO-Inference %s" % version("deepspeed")
    except Exception:
        return "DeepSpeed ZeRO-Inference (version unknown -- not importable)"


METHODS = {
    "airllm": Method("airllm", _installed_airllm_title(), "airllm", {},
                     "reference_greedy", 30.0),
    "accelerate": Method(
        "accelerate", _installed_accelerate_title(), "accelerate",
        {"gpu_memory": "1500MB", "cpu_memory": "8GB"},
        "reference_greedy", 30.0),
    # cpu_offload=True is the usable default on this project's reference
    # hardware: Qwen3-14B-DF11 compressed is DFloat11's own claimed ~70% of
    # bf16 size, i.e. roughly 19-20 GB, which does not fit an 8 GB RTX 3080
    # at all without CPU offload. dfloat11-gpu-resident (cpu_offload=False)
    # is registered separately below for hosts with enough VRAM to hold the
    # compressed model outright; on this reference card it is expected to
    # fail loudly (CUDA OOM) rather than silently degrade, which is itself
    # a real, reportable data point about DFloat11's applicability regime.
    "dfloat11": Method(
        "dfloat11", _installed_dfloat11_title(), "dfloat11",
        {"model_id": DFLOAT11_MODEL, "cpu_offload": True},
        "reference_greedy", 20.0),
    "dfloat11-gpu-resident": Method(
        "dfloat11-gpu-resident", _installed_dfloat11_title(), "dfloat11",
        {"model_id": DFLOAT11_MODEL, "cpu_offload": False},
        "reference_greedy", 20.0),
    # DeepSpeed ZeRO-Inference: the same GPU/CPU-RAM/(optionally NVMe)
    # tiered-offload category AirLLM and Accelerate occupy (confirmed
    # against DeepSpeed's own zero_inference reference implementation,
    # not assumed from its training-focused ZeRO-Infinity sibling's
    # marketing) -- a real, general-purpose, actively-maintained direct
    # competitor, not an architecture-specific one like KTransformers/
    # Fiddler (MoE-only) or FlexGen (OPT-family only, ruled out here for
    # exactly that reason). cpu offload is the default for the same
    # reason as dfloat11/accelerate above: this project's reference 8 GB
    # card cannot hold a 14B bf16 model's ZeRO-3-partitioned live
    # parameters without it.
    "deepspeed-zero-inference": Method(
        "deepspeed-zero-inference", _installed_deepspeed_title(), "deepspeed",
        # The reference host has 19 GiB of RAM, less than the 29.5 GB BF16
        # checkpoint. CPU-only offload therefore cannot be the reproducible
        # default for this comparison. ZeRO-3's documented NVMe parameter
        # offload keeps the baseline applicable on the same machine. Pinned
        # memory is disabled because WSL2's default memlock limit is commonly
        # only 64 MiB; the result records the choice explicitly.
        {"offload_device": "nvme", "offload_path": DEEPSPEED_OFFLOAD_DIR,
         "pin_memory": False},
        "reference_greedy", 20.0),
    "exact-min": Method(
        "exact-min", "Afterimage exact streaming, minimum-memory control", "afterimage",
        {"vram_budget_gb": 1.80, "decode_slice_elems": 1 << 20,
         "io_prefetch_depth": 2}, "reference_execution_equivalent", 31.0),
    "exact-resident": Method(
        "exact-resident", "Afterimage exact streaming + 4 GB residency", "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2}, "reference_execution_equivalent", 20.0),
    "ram-overlay-head": Method(
        "ram-overlay-head", "Afterimage exact pinned-RAM lm_head overlay",
        "afterimage",
        {"vram_budget_gb": 1.80, "ram_budget_gb": 1.60,
         "decode_slice_elems": 1 << 20, "io_prefetch_depth": 2,
         "lm_head_policy": "ram_overlay", "require_pinned_ram": True},
        "reference_execution_equivalent", 24.0),
    "full-head-control": Method(
        "full-head-control", "Afterimage legacy streaming with resident full head",
        "afterimage", {"io_prefetch_depth": 2},
        "reference_execution_equivalent", 30.0),
    "chunked-head": Method(
        "chunked-head", "Afterimage chunked output head", "afterimage",
        {"vram_budget_gb": 0.50, "decode_slice_elems": 1 << 20,
         "io_prefetch_depth": 2, "lm_head_slice_rows": 2048},
        "approximate", 30.0),
    "pi-prefetch": Method(
        "pi-prefetch", "Afterimage PI-controlled prefetch", "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "io_prefetch_max_depth": 8,
         "prefetch_policy": "pi"}, "reference_execution_equivalent", 20.0),
    "mpc-prefetch": Method(
        "mpc-prefetch", "Afterimage one-step MPC prefetch", "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "io_prefetch_max_depth": 8,
        "prefetch_policy": "mpc"}, "reference_execution_equivalent", 20.0),
    "bayes-prefetch": Method(
        "bayes-prefetch", "Afterimage Bayesian probit prefetch", "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "io_prefetch_max_depth": 8,
         "prefetch_target_ready": 0.90,
         "prefetch_policy": "bayes_probit"},
        "reference_execution_equivalent", 20.0),
    "profiled-knapsack": Method(
        "profiled-knapsack", "Afterimage measured-cost residency", "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "placement_policy": "profiled_knapsack"},
        "reference_execution_equivalent", 20.0),
    "critical-path": Method(
        "critical-path", "Afterimage event-DAG critical-path residency", "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "placement_policy": "critical_path"},
        "reference_execution_equivalent", 20.0),
    "replay-cem": Method(
        "replay-cem", "Afterimage digital-twin CEM residency", "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "placement_policy": "replay_cem"},
        "reference_execution_equivalent", 20.0),
    "replay-qubo": Method(
        "replay-qubo", "Afterimage event-interference QUBO residency", "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "placement_policy": "replay_qubo"},
        "reference_execution_equivalent", 20.0),
    "coalesced-storage": Method(
        "coalesced-storage", "Afterimage bounded contiguous storage reads",
        "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2,
         "storage_read_policy": "coalesced_extents",
         "storage_extent_max_bytes": 1 << 28,
         "storage_extent_max_gap_bytes": 0},
        "reference_execution_equivalent", 20.0),
    "tensor-extents": Method(
        "tensor-extents", "Afterimage tensor-scoped micro-extents", "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2,
         "storage_read_policy": "tensor_extents",
         "storage_extent_max_bytes": 1 << 23,
         "storage_extent_max_gap_bytes": 0},
        "reference_execution_equivalent", 20.0),
    "replay-extent-qubo": Method(
        "replay-extent-qubo", "Afterimage physical-extent QUBO residency",
        "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "placement_policy": "replay_extent_qubo"},
        "reference_execution_equivalent", 20.0),
    "certified-mips": Method(
        "certified-mips", "Afterimage certified greedy MIPS head", "afterimage",
        {"io_prefetch_depth": 2, "lm_head_policy": "certified_mips"},
        "greedy_token_exact", 30.0),
    "spec-fixed": Method(
        "spec-fixed", "Afterimage fixed-k speculative decoding", "afterimage",
        {"vram_budget_gb": 2.70, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "draft_mode": "model", "spec_k": 8,
         "spec_k_policy": "fixed"}, "greedy_token_exact_at_temperature_zero", 8.0),
    # Fixed-k speculation ablation series, all at spec-fixed's own 2.70 GB
    # budget so the *only* variable across this series is k (see engine.
    # stats.spec_sweeps/spec_accepted_tokens, already recorded per row by
    # run_afterimage, for accepted-tokens/sweep and sweeps/committed-token).
    # spec-k0 (draft_mode="none") is the no-speculation control at that same
    # budget -- exact-min/exact-resident are NOT that control, since they
    # sit at different budgets (1.80/4.00 GB) and would confound the memory
    # variable with the speculation variable.
    "spec-k0": Method(
        "spec-k0", "Afterimage no speculation (k-ablation control)", "afterimage",
        {"vram_budget_gb": 2.70, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "draft_mode": "none"},
        "reference_execution_equivalent", 20.0),
    "spec-k2": Method(
        "spec-k2", "Afterimage fixed k=2 speculative decoding", "afterimage",
        {"vram_budget_gb": 2.70, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "draft_mode": "model", "spec_k": 2,
         "spec_k_policy": "fixed"}, "greedy_token_exact_at_temperature_zero", 10.0),
    "spec-k4": Method(
        "spec-k4", "Afterimage fixed k=4 speculative decoding", "afterimage",
        {"vram_budget_gb": 2.70, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "draft_mode": "model", "spec_k": 4,
         "spec_k_policy": "fixed"}, "greedy_token_exact_at_temperature_zero", 9.0),
    "spec-k16": Method(
        "spec-k16", "Afterimage fixed k=16 speculative decoding", "afterimage",
        {"vram_budget_gb": 2.70, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "draft_mode": "model", "spec_k": 16,
         "spec_k_policy": "fixed"}, "greedy_token_exact_at_temperature_zero", 7.0),
    "spec-critical": Method(
        "spec-critical", "Afterimage critical-path residency + fixed speculation",
        "afterimage",
        {"vram_budget_gb": 2.70, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "placement_policy": "critical_path",
         "draft_mode": "model", "spec_k": 8, "spec_k_policy": "fixed"},
        "greedy_token_exact_at_temperature_zero", 8.0),
    "spec-cached": Method(
        "spec-cached", "Afterimage rollback-cached target speculation", "afterimage",
        {"vram_budget_gb": 2.70, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "draft_mode": "model", "spec_k": 8,
         "spec_k_policy": "fixed", "spec_target_cache": True},
        "greedy_token_exact_at_temperature_zero", 8.0),
    "breakdown-exact": Method(
        "breakdown-exact", "Afterimage traced exact runtime breakdown", "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "trace_events": True},
        "reference_execution_equivalent", 20.0),
    "simple-v4-r8": Method(
        "simple-v4-r8", "Afterimage simple tier placement at 4 GB VRAM / 8 GB RAM",
        "afterimage",
        {"vram_budget_gb": 4.00, "ram_budget_gb": 8.00,
         "decode_slice_elems": 1 << 22, "io_prefetch_depth": 2,
         "placement_policy": "traffic_density", "ram_tier_format": "decoded"},
        "reference_execution_equivalent", 20.0),
    "h1-v4-r8": Method(
        "h1-v4-r8", "Afterimage H1 placement at 4 GB VRAM / 8 GB RAM",
        "afterimage",
        {"vram_budget_gb": 4.00, "ram_budget_gb": 8.00,
         "decode_slice_elems": 1 << 22, "io_prefetch_depth": 2,
         "placement_policy": "critical_path", "ram_tier_format": "decoded"},
        "reference_execution_equivalent", 20.0),
    "h6-disk-v4-r8": Method(
        "h6-disk-v4-r8", "Afterimage compressed SSD streaming control",
        "afterimage",
        {"vram_budget_gb": 4.00, "ram_budget_gb": 8.00,
         "decode_slice_elems": 1 << 22, "io_prefetch_depth": 2,
         "representation_policy": "per_tensor",
         "representation_plan_state": DISK_PLAN_STATE},
        "reference_execution_equivalent", 31.0),
    "h6-live-v4-r8": Method(
        "h6-live-v4-r8", "Afterimage live H6 representation plan",
        "afterimage",
        {"vram_budget_gb": 4.00, "ram_budget_gb": 8.00,
         "decode_slice_elems": 1 << 22, "io_prefetch_depth": 2,
         "representation_policy": "per_tensor",
         "representation_plan_state": H6_PLAN_STATE},
        "reference_execution_equivalent", 16.0),
    "breakdown-spec": Method(
        "breakdown-spec", "Afterimage traced speculative runtime breakdown",
        "afterimage",
        {"vram_budget_gb": 2.70, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "draft_mode": "model", "spec_k": 8,
         "spec_k_policy": "fixed", "trace_events": True},
        "greedy_token_exact_at_temperature_zero", 8.0),
    "spec-hazard": Method(
        "spec-hazard", "Afterimage frozen rejection-hazard stopping", "afterimage",
        {"vram_budget_gb": 2.70, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "draft_mode": "model", "spec_k": 8,
         "spec_k_policy": "hazard_cost", "spec_policy_learn": False},
        "greedy_token_exact_at_temperature_zero", 8.0),
    "spec-neural": Method(
        "spec-neural", "Afterimage frozen tiny neural utility stopping",
        "afterimage",
        {"vram_budget_gb": 2.70, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "draft_mode": "model", "spec_k": 8,
         "spec_k_policy": "neural_utility", "spec_policy_learn": False},
        "greedy_token_exact_at_temperature_zero", 8.0),
    "chunked-spec": Method(
        "chunked-spec", "Afterimage chunked head + fixed-k speculation", "afterimage",
        {"vram_budget_gb": 0.50, "decode_slice_elems": 1 << 20,
         "io_prefetch_depth": 2, "lm_head_slice_rows": 2048,
         "draft_mode": "model", "spec_k": 8, "spec_k_policy": "fixed"},
        "approximate", 12.0),
}

DEFAULT_METHODS = (
    "airllm", "exact-min", "exact-resident", "chunked-head", "pi-prefetch",
    "critical-path", "certified-mips", "spec-fixed", "spec-hazard",
)


def log(message: str) -> None:
    print(message, flush=True)


def drop_caches() -> tuple[bool, str | None]:
    """Drop the Linux page cache or make the failure explicit in the result."""
    try:
        subprocess.run(["sync"], check=True, timeout=60)
        pathlib.Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return True, None
    except Exception as exc:  # benchmark must continue, but never hide this
        return False, repr(exc)


def process_read_bytes() -> int:
    """Cumulative bytes this process has asked the kernel to read, from
    /proc/self/io's read_bytes counter (ported from scripts/
    matched_vram_final.py, which established this measurement first).

    This is process/storage read traffic as accounted by the kernel's own
    block-layer I/O statistics for this process -- not confirmed physical
    NVMe traffic. Under WSL2 in particular, /proc/self/io reflects the
    virtualized 9p/vhdx storage stack the guest sees, which does not
    necessarily correspond 1:1 with physical device reads on the Windows
    host (page cache behavior, the vhdx's own caching, and virtio-9p
    request coalescing can all sit between this number and an actual NVMe
    read). Call it what it measures.
    """
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def reset_cuda_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def canonical_peak_vram(report) -> tuple[float | None, str | None]:
    """The one peak-VRAM number aggregate()/pareto_frontier() should trust,
    plus which measurement it came from.

    torch.cuda.max_memory_allocated() alone is not just incomplete, it is
    actively wrong for some methods: HF Accelerate's device_map="auto"
    dispatch reports exactly 0.0 to it -- not "unavailable", a real float
    that then won a Pareto-frontier comparison outright, because 0 GB
    looked like the best memory result in the campaign. Prefer the
    nvidia-smi delta (afterimage.bench.memory.MemoryProbe), which observes
    actual device usage regardless of which allocator path put it there,
    and fall back to the torch figure only if nvidia-smi itself was
    unavailable. Never fabricate a 0.0 -- an unmeasurable cell must report
    None so aggregate() and pareto_frontier() can exclude it, not rank it
    as the cheapest.
    """
    smi_delta = report.smi_delta_gb
    if smi_delta is not None:
        return smi_delta, "nvidia_smi_delta"
    torch_peak = report.torch_peak_vram_gb
    if torch_peak is not None:
        return torch_peak, "torch_allocator"
    return None, None


def memory_probe_extra_fields(report) -> dict:
    """Auxiliary memory readings alongside the canonical peak_vram_gb, so a
    reader can see torch/nvidia-smi agree or disagree rather than trusting
    one silently-chosen number."""
    return {
        "torch_peak_vram_gb": report.torch_peak_vram_gb,
        "smi_peak_vram_gb": (report.smi_peak_used_mb / 1000.0
                             if report.smi_peak_used_mb is not None else None),
        "smi_baseline_vram_gb": (report.smi_baseline_used_mb / 1000.0
                                 if report.smi_baseline_used_mb is not None else None),
        "host_rss_peak_gb": report.host_rss_peak_gb,
    }


def release_cuda(*objects) -> None:
    for obj in objects:
        with contextlib.suppress(Exception):
            if hasattr(obj, "close"):
                obj.close()
    del objects
    gc.collect()
    torch.cuda.empty_cache()


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=10, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else None
    except Exception:
        return None


def cpu_model() -> str | None:
    """CPU model string. /proc/cpuinfo on Linux; falls back to platform.processor()
    elsewhere (typically empty on Windows without extra tooling, which is
    honest -- this suite has never been run for real outside WSL2/Linux)."""
    try:
        text = pathlib.Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def storage_device_info(path: pathlib.Path) -> dict:
    """Best-effort storage identity for the filesystem holding `path`.

    This is the single most important hardware fact for a disk-bound
    benchmark and was previously not logged at all. Under WSL2 the real
    NVMe device is virtualized (`lsblk` reports "Virtual Disk"), so the
    filesystem type/mount is what WSL2 itself can see; the underlying
    physical device should additionally be recorded once, by hand, in the
    run's accompanying notes -- this function cannot see through the
    virtualization layer from inside the guest, and pretending otherwise
    would be a fabricated hardware claim, not a measured one.
    """
    info: dict = {"path": str(path)}
    fs = command_output(["df", "-T", str(path)])
    if fs:
        lines = fs.strip().splitlines()
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 2:
                info["filesystem"] = parts[0]
                info["fstype"] = parts[1]
    block = command_output(["lsblk", "-d", "-o", "NAME,MODEL,ROTA", "-n"])
    if block:
        info["lsblk_devices"] = block.strip()
        info["note"] = (
            "device MODEL above is virtualized under WSL2 (typically "
            "'Virtual Disk'); the real physical NVMe backing the WSL2 vhdx "
            "is a Windows-host fact this guest cannot see and must be "
            "recorded separately, e.g. via Windows Get-PhysicalDisk")
    return info


def gpu_thermal_snapshot() -> dict:
    """Instantaneous clocks/temperature/power. Call once for the environment
    manifest and again per timed cell (see per-cell logging in run_afterimage
    / run_airllm) -- a laptop GPU throttles under sustained load, and a
    manifest-level snapshot alone cannot show drift across a multi-hour
    campaign."""
    raw = command_output([
        "nvidia-smi",
        "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,"
        "enforced.power.limit,clocks_throttle_reasons.active,"
        "clocks_event_reasons_counters.sw_thermal_slowdown,"
        "clocks_event_reasons_counters.sw_power_cap,"
        "clocks_event_reasons_counters.hw_thermal_slowdown,"
        "clocks_event_reasons_counters.hw_power_brake_slowdown",
        "--format=csv,noheader,nounits"])
    if not raw:
        return {}
    parts = [p.strip() for p in raw.strip().split(",")]
    if len(parts) < 6:
        return {"raw": raw.strip()}
    keys = ("sm_clock_mhz", "mem_clock_mhz", "temperature_c", "power_draw_w",
            "power_limit_w", "throttle_reasons_active",
            "sw_thermal_slowdown_counter_us", "sw_power_cap_counter_us",
            "hw_thermal_slowdown_counter_us", "hw_power_brake_counter_us")
    snapshot = dict(zip(keys, parts[:len(keys)]))
    snapshot["throttled"] = is_throttled(snapshot)
    snapshot["thermal_throttled"] = thermal_throttled(snapshot)
    snapshot["power_limited"] = power_limited(snapshot)
    return snapshot


# NVML clocks-event-reason bits that mean the GPU is being held below its
# requested clocks *while doing work*. GpuIdle (0x1) and the applications /
# display clock settings are deliberately excluded: they are not a
# performance confound, they are normal states.
#
# Split into thermal vs. power on purpose (not left as one merged
# "throttled" bit): a laptop's power limiter capping sustained draw is an
# ordinary, expected steady-state condition on this project's reference
# hardware, not a thermal fault -- conflating the two means a run that was
# simply power-capped the whole time (normal) looks identical to one that
# was genuinely overheating (a real confound worth flagging separately).
_THERMAL_MASK = (
    0x0000000000000020  # SwThermalSlowdown
    | 0x0000000000000040  # HwThermalSlowdown
)
_POWER_MASK = (
    0x0000000000000004  # SwPowerCap
    | 0x0000000000000080  # HwPowerBrakeSlowdown
)
# HwSlowdown (0x8) is a real hardware slowdown signal without NVML telling
# us *why* (thermal, power, or something else) -- kept in the combined
# mask for backward-compatible is_throttled(), but not attributed to
# either specific category since that would be a guess.
_THROTTLE_MASK = _THERMAL_MASK | _POWER_MASK | 0x0000000000000008


def _throttle_bits(snapshot: dict) -> int | None:
    raw = snapshot.get("throttle_reasons_active")
    if raw in (None, "", "[N/A]", "N/A"):
        return None
    try:
        return int(str(raw), 16)
    except ValueError:
        return None


def is_throttled(snapshot: dict) -> bool | None:
    """Whether ANY throttle reason (thermal, power, or unattributed
    hardware slowdown) was active in this snapshot -- kept for existing
    callers; thermal_throttled/power_limited (also set on the snapshot by
    gpu_thermal_snapshot) are the reclassified fields a comparison should
    actually condition on."""
    bits = _throttle_bits(snapshot)
    return None if bits is None else bool(bits & _THROTTLE_MASK)


def thermal_throttled(snapshot: dict) -> bool | None:
    bits = _throttle_bits(snapshot)
    return None if bits is None else bool(bits & _THERMAL_MASK)


def power_limited(snapshot: dict) -> bool | None:
    bits = _throttle_bits(snapshot)
    return None if bits is None else bool(bits & _POWER_MASK)


def cool_down(seconds: float, max_temperature_c: float | None) -> dict:
    """Idle between timed cells so thermal state does not accumulate.

    Without this the benchmark measures method *order* as much as method
    quality: the first system runs on a cool GPU and the last runs on a hot
    one.

    Gating on temperature alone is provably insufficient, not just
    theoretically: measured on this project's own reference machine mid-
    campaign, `sm_clock_mhz` collapsed from a 1890 MHz boost clock to a flat
    780 MHz while `temperature_c` simultaneously *dropped* to 59-63 C --
    below any reasonable cooldown target -- because a GPU clocked that low
    no longer generates enough heat to look hot. A temperature-only gate
    would have measured "cool enough" and immediately resumed the still-
    throttled run.

    Throttle-clear is therefore mandatory and unconditional: even the fully
    default call ``cool_down(0.0, None)`` -- what every timed cell got
    whenever a caller (canonical benchmark.sh included) passed neither
    --cooldown-seconds nor --cooldown-max-temp-c -- waits out
    ``is_throttled() is True`` up to the 600 s hard ceiling before
    proceeding. This used to be gated behind ``max_temperature_c`` being
    set, and separately the seconds-only floor path never consulted
    ``is_throttled()`` at all, so a genuinely throttled GPU was invisible to
    cool_down() unless a caller happened to also pass a temperature target.
    --cooldown-seconds and --cooldown-max-temp-c now only ADD an idle floor
    and/or a temperature ceiling on top of the mandatory throttle check;
    they do not gate whether that check runs. An unknown throttle reading
    (``is_throttled()`` returns None -- no nvidia-smi, non-NVIDIA host) does
    not block, matching every other place in this module that treats
    "unknown" as distinct from a confident "not throttled".
    """
    started = time.perf_counter()
    deadline = started + max(seconds, 0.0)
    reached = None
    while True:
        now = time.perf_counter()
        snapshot = gpu_thermal_snapshot()
        try:
            temperature = float(snapshot.get("temperature_c"))
        except (TypeError, ValueError):
            temperature = None
        throttled = snapshot.get("throttled")
        temperature_ok = max_temperature_c is None or (
            temperature is not None and temperature <= max_temperature_c)
        throttle_ok = throttled is not True  # None (unknown) does not block.
        reached = temperature_ok and throttle_ok
        # Honour the floor wait even once ready, so a fast-cooling run still
        # gets a consistent inter-cell gap.
        if reached and now >= deadline:
            break
        # Hard ceiling: never wait more than 10 minutes for a GPU that is
        # simply not cooling or not clearing its throttle, and say so in the
        # record rather than hanging.
        if now - started > 600:
            reached = False
            break
        time.sleep(2.0)
    final = gpu_thermal_snapshot()
    return {
        "cooldown_seconds": time.perf_counter() - started,
        "cooldown_target_c": max_temperature_c,
        "cooldown_reached_target": reached,
        "temperature_after_cooldown_c": final.get("temperature_c"),
        "throttled_after_cooldown": final.get("throttled"),
    }


def model_revision(model_id: str) -> str | None:
    """The exact commit SHA of the HF Hub model used, independent of the
    tokenizer's own (separately tracked) revision. A network lookup, not a
    download; failures (offline, rate-limited) degrade to None rather than
    failing the run, since this is provenance metadata, not a correctness
    requirement."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_id).sha
    except Exception:
        return None


def environment_manifest(repo_root: pathlib.Path, tokenizer,
                         store: pathlib.Path | None = None) -> dict:
    gpu = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu.name,
        "gpu_total_bytes": gpu.total_memory,
        "driver": command_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ]),
        "gpu_thermal_at_start": gpu_thermal_snapshot(),
        "host_memory": command_output(["free", "-b"]),
        "storage": storage_device_info(store or pathlib.Path.home()),
        "packages": {name: package_version(name) for name in (
            "airllm", "transformers", "accelerate", "dfloat11", "deepspeed",
            "safetensors", "numpy")},
        "git_commit": command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "-C", str(repo_root), "status", "--short"]),
        "tokenizer_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "model_revision": model_revision(MODEL),
    }


def render_cases(tokenizer, cases: tuple[PromptCase, ...]) -> list[dict]:
    rendered = []
    for case in cases:
        prompt = render_chat_prompt(tokenizer, case)
        input_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        rendered.append({"case": case, "prompt": prompt, "input_tokens": input_tokens})
    return rendered


def load_tokenizer(model_id: str):
    """Load one cross-family tokenizer with Mistral's regex fix enabled."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)


def result_row(case: PromptCase, method: Method, prompt: str, input_tokens: int,
               generated_ids: list[int], answer: str, wall_s: float,
               peak_vram_gb: float | None, cache_drop: tuple[bool, str | None],
               extra: dict | None = None) -> dict:
    row = {
        "case_id": case.id,
        "semantic_bucket": case.semantic_bucket,
        "method": method.id,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_tokens": input_tokens,
        "output_tokens": len(generated_ids),
        "output_token_ids": generated_ids,
        "answer": answer,
        "expected_any": list(case.expected_any),
        "expected_match": case.matches(answer),
        "wall_seconds": wall_s,
        "seconds_per_token": wall_s / max(len(generated_ids), 1),
        "committed_tokens_per_second": len(generated_ids) / max(wall_s, 1e-12),
        "peak_vram_gb": peak_vram_gb,
        "cache_drop_succeeded": cache_drop[0],
        "cache_drop_error": cache_drop[1],
    }
    row.update(extra or {})
    return row


def run_airllm(method: Method, rendered: list[dict], n_tokens: int,
               deadline: float,
               rows_checkpoint: Callable[[list[dict]], None] | None = None,
               repeats: int = 1, repeat_offset: int = 0,
               warmup_tokens: int = 0,
               ) -> tuple[list[dict], dict]:
    from airllm import AutoModel

    init_t0 = time.perf_counter()
    model = AutoModel.from_pretrained(MODEL)
    # AirLLM 3.1.0 constructs its own tokenizer without Transformers'
    # ``fix_mistral_regex`` compatibility flag.  Reuse the already-rendered
    # benchmark tokenizer so every backend consumes identical input IDs.
    model.tokenizer = rendered[0]["tokenizer"]
    init_s = time.perf_counter() - init_t0
    rows = []
    try:
        if warmup_tokens > 0 and rendered:
            # Untimed: absorbs first-call CUDA/Triton/allocator compilation
            # so it does not land inside repeat 0's measured seconds/token.
            # docs/RESULTS_LOG.md's own reproduction found 0.165/0.047/0.056
            # s/token across three back-to-back Qwen3-0.6B repeats -- most of
            # that spread is exactly this one-time warm-up cost.
            warm = rendered[0]
            enc = model.tokenizer(warm["prompt"], return_tensors="pt", truncation=True)
            ids = enc["input_ids"].cuda()
            warm_pad = model.tokenizer.pad_token_id
            if warm_pad is None:
                warm_pad = model.tokenizer.eos_token_id
            if isinstance(warm_pad, (list, tuple)):
                warm_pad = warm_pad[0] if warm_pad else None
            if warm_pad is None:
                warm_pad = 0
            model.generate(ids, max_new_tokens=warmup_tokens, eos_token_id=[],
                           pad_token_id=int(warm_pad), do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            del ids, enc
        # Repeats are the outer dimension so a truncated run (deadline hit)
        # still holds a complete sweep of every case for the repeats it did
        # finish, rather than many repeats of the first case and none of the
        # last. aggregate() reports dispersion across repeats.
        for repeat, item in [(r, it) for r in range(repeats) for it in rendered]:
            if time.perf_counter() >= deadline:
                break
            enc = model.tokenizer(item["prompt"], return_tensors="pt", truncation=True)
            ids = enc["input_ids"].cuda()
            kwargs = {}
            if enc.get("attention_mask") is not None:
                kwargs["attention_mask"] = enc["attention_mask"].cuda()
            cooldown = cool_down(COOLDOWN_SECONDS, COOLDOWN_MAX_TEMPERATURE_C)
            cache = drop_caches()
            reset_cuda_peak()
            read0 = process_read_bytes()
            gpu_thermal_before = gpu_thermal_snapshot()
            with MemoryProbe() as probe:
                t0 = time.perf_counter()
                # An empty EOS list forces a fixed token count. Transformers
                # 5.x still needs a concrete pad ID when EOS is empty,
                # otherwise it indexes eos_token_tensor[0] while preparing
                # special tokens.
                pad_token_id = model.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = model.tokenizer.eos_token_id
                if isinstance(pad_token_id, (list, tuple)):
                    pad_token_id = pad_token_id[0] if pad_token_id else None
                if pad_token_id is None:
                    pad_token_id = 0
                output = model.generate(
                    ids, max_new_tokens=n_tokens, eos_token_id=[],
                    pad_token_id=int(pad_token_id),
                    do_sample=False, use_cache=True, return_dict_in_generate=True, **kwargs)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
            gpu_thermal_after = gpu_thermal_snapshot()
            mem_report = probe.report()
            peak_vram_gb, peak_vram_source = canonical_peak_vram(mem_report)
            read_bytes = process_read_bytes() - read0
            sequence = output.sequences if hasattr(output, "sequences") else output
            generated = sequence[0, ids.shape[1]:].tolist()
            answer = model.tokenizer.decode(generated, skip_special_tokens=True)
            rows.append(result_row(
                item["case"], method, item["prompt"], item["input_tokens"],
                generated, answer, wall, peak_vram_gb,
                cache, {"generation_mode": "greedy", "repeat": repeat_offset + repeat,
                        "peak_vram_source": peak_vram_source,
                        **memory_probe_extra_fields(mem_report),
                        "gpu_thermal_before": gpu_thermal_before,
                        "gpu_thermal": gpu_thermal_after,
                        "process_read_bytes": read_bytes,
                        "process_read_bytes_per_token": read_bytes / max(len(generated), 1),
                        **cooldown}))
            if rows_checkpoint is not None:
                rows_checkpoint(rows)
            log("  %-18s %.2f s/token  %r%s" %
                (item["case"].id, rows[-1]["seconds_per_token"], answer,
                 "" if repeats == 1 else "  [repeat %d/%d]" % (repeat + 1, repeats)))
            del output, sequence, ids
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows, {"initialization_seconds": init_s}


def run_accelerate(method: Method, rendered: list[dict], n_tokens: int,
                   deadline: float,
                   rows_checkpoint: Callable[[list[dict]], None] | None = None,
                   repeats: int = 1, repeat_offset: int = 0,
                   warmup_tokens: int = 0,
                   ) -> tuple[list[dict], dict]:
    from afterimage.baselines.b0_hf_offload import load_hf_offload_baseline

    init_t0 = time.perf_counter()
    baseline = load_hf_offload_baseline(
        MODEL, os.environ.get("AFTERIMAGE_HF_OFFLOAD_DIR", "/root/afterimage/hf_offload_14b"),
        gpu_memory=method.overrides["gpu_memory"],
        cpu_memory=method.overrides["cpu_memory"])
    baseline.tokenizer = rendered[0]["tokenizer"]
    init_s = time.perf_counter() - init_t0
    rows = []
    try:
        if warmup_tokens > 0 and rendered:
            # Untimed, see run_airllm's identical rationale.
            baseline.generate(rendered[0]["prompt"], warmup_tokens)
        for repeat, item in [(r, it) for r in range(repeats) for it in rendered]:
            if time.perf_counter() >= deadline:
                break
            cooldown = cool_down(COOLDOWN_SECONDS, COOLDOWN_MAX_TEMPERATURE_C)
            cache = drop_caches()
            read0 = process_read_bytes()
            gpu_thermal_before = gpu_thermal_snapshot()
            result = baseline.generate(item["prompt"], n_tokens)
            gpu_thermal_after = gpu_thermal_snapshot()
            read_bytes = process_read_bytes() - read0
            generated = result["output_token_ids"]
            mem_report = result["memory_report"]
            peak_vram_gb, peak_vram_source = canonical_peak_vram(mem_report)
            rows.append(result_row(
                item["case"], method, item["prompt"], item["input_tokens"],
                generated, result["text"], result["wall_seconds"],
                peak_vram_gb, cache,
                {"generation_mode": "greedy", "repeat": repeat_offset + repeat,
                 "device_map": baseline.device_map,
                 "offload_dir": baseline.offload_dir,
                 "gpu_memory_limit": method.overrides["gpu_memory"],
                 "cpu_memory_limit": method.overrides["cpu_memory"],
                 "peak_vram_source": peak_vram_source,
                 **memory_probe_extra_fields(mem_report),
                 "gpu_thermal_before": gpu_thermal_before,
                 "gpu_thermal": gpu_thermal_after,
                 "process_read_bytes": read_bytes,
                 "process_read_bytes_per_token": read_bytes / max(len(generated), 1),
                 **cooldown}))
            if rows_checkpoint is not None:
                rows_checkpoint(rows)
            log("  %-18s %.2f s/token  %r%s" %
                (item["case"].id, rows[-1]["seconds_per_token"], rows[-1]["answer"],
                 "" if repeats == 1 else "  [repeat %d/%d]" % (repeat + 1, repeats)))
    finally:
        baseline.model = None
        del baseline
        gc.collect()
        torch.cuda.empty_cache()
    return rows, {"initialization_seconds": init_s,
                  "device_map": rows[0]["device_map"] if rows else {}}


def run_dfloat11(method: Method, rendered: list[dict], n_tokens: int,
                 deadline: float,
                 rows_checkpoint: Callable[[list[dict]], None] | None = None,
                 repeats: int = 1, repeat_offset: int = 0,
                 warmup_tokens: int = 0,
                 ) -> tuple[list[dict], dict]:
    """DFloat11 (arXiv:2504.11651, NeurIPS'25): the closest published
    convergent-evidence baseline to Afterimage's own compression mechanism
    (Huffman-coded bf16 exponents, GPU LUT decode -- see docs/LITERATURE.md
    and afterimage/probe/entropy.py). It decompresses into GPU/host memory
    at load time rather than streaming compressed weights per token, so
    this is a compression-ratio-class baseline, not a streaming-class one,
    which is exactly the comparison this project's own literature review
    calls for measuring head-to-head rather than trusting two papers'
    separately-measured numbers.

    Loads ``method.overrides["model_id"]`` (default DFLOAT11_MODEL, the
    real published DFloat11/Qwen3-14B-DF11 checkpoint -- parameter-for-
    parameter the same model as MODEL) rather than compressing MODEL itself;
    DFloat11 ships fixed pre-compressed repos per model, it does not
    compress an arbitrary checkpoint at runtime.
    """
    # dfloat11 0.5.0's own model-loading code does a lazy
    # `from transformers.modeling_utils import no_init_weights` deep
    # inside DFloat11Model.from_pretrained -- found by actually running
    # this against transformers 5.12.1 (the import itself succeeds; only
    # calling from_pretrained fails), where that symbol moved to
    # transformers.initialization. This is dfloat11's own compatibility
    # gap, not a project change to work around by pinning a different
    # transformers version project-wide (which every other method here
    # also depends on). Shimming the old location back is the minimal,
    # contained fix; a no-op once dfloat11 itself picks up the new import
    # path, since setattr on an already-present name is harmless.
    import transformers.modeling_utils as _modeling_utils
    if not hasattr(_modeling_utils, "no_init_weights"):
        from transformers.initialization import no_init_weights as _no_init_weights
        _modeling_utils.no_init_weights = _no_init_weights

    from dfloat11 import DFloat11Model

    model_id = method.overrides.get("model_id", DFLOAT11_MODEL)
    cpu_offload = method.overrides.get("cpu_offload", False)
    init_t0 = time.perf_counter()
    model = DFloat11Model.from_pretrained(
        model_id, device_map="auto", cpu_offload=cpu_offload)
    tokenizer = rendered[0]["tokenizer"]
    # device_map="auto" is standard HF Accelerate dispatch; `model.device`
    # is the conventional way to find where a dispatched model's inputs
    # belong. Fall back to cuda:0 for the single-consumer-GPU case this
    # project targets if that attribute is ever absent.
    input_device = getattr(model, "device", None) or torch.device("cuda")
    init_s = time.perf_counter() - init_t0
    rows = []

    def _pad_token_id() -> int:
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if isinstance(pad, (list, tuple)):
            pad = pad[0] if pad else None
        return int(pad) if pad is not None else 0

    try:
        if warmup_tokens > 0 and rendered:
            # Untimed, see run_airllm's identical rationale.
            enc = tokenizer(rendered[0]["prompt"], return_tensors="pt").to(input_device)
            with torch.no_grad():
                model.generate(**enc, max_new_tokens=warmup_tokens, eos_token_id=[],
                               pad_token_id=_pad_token_id(), do_sample=False,
                               use_cache=True)
            torch.cuda.synchronize()
            del enc
        for repeat, item in [(r, it) for r in range(repeats) for it in rendered]:
            if time.perf_counter() >= deadline:
                break
            enc = tokenizer(item["prompt"], return_tensors="pt").to(input_device)
            cooldown = cool_down(COOLDOWN_SECONDS, COOLDOWN_MAX_TEMPERATURE_C)
            cache = drop_caches()
            reset_cuda_peak()
            read0 = process_read_bytes()
            gpu_thermal_before = gpu_thermal_snapshot()
            with MemoryProbe() as probe:
                t0 = time.perf_counter()
                with torch.no_grad():
                    output = model.generate(
                        **enc, max_new_tokens=n_tokens, eos_token_id=[],
                        pad_token_id=_pad_token_id(), do_sample=False, use_cache=True,
                        return_dict_in_generate=True)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
            gpu_thermal_after = gpu_thermal_snapshot()
            mem_report = probe.report()
            peak_vram_gb, peak_vram_source = canonical_peak_vram(mem_report)
            read_bytes = process_read_bytes() - read0
            sequence = output.sequences if hasattr(output, "sequences") else output
            generated = sequence[0, enc["input_ids"].shape[1]:].tolist()
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            rows.append(result_row(
                item["case"], method, item["prompt"], item["input_tokens"],
                generated, answer, wall, peak_vram_gb,
                cache, {"generation_mode": "greedy",
                        "repeat": repeat_offset + repeat,
                        "dfloat11_model": model_id, "cpu_offload": cpu_offload,
                        "peak_vram_source": peak_vram_source,
                        **memory_probe_extra_fields(mem_report),
                        "gpu_thermal_before": gpu_thermal_before,
                        "gpu_thermal": gpu_thermal_after,
                        "process_read_bytes": read_bytes,
                        "process_read_bytes_per_token": read_bytes / max(len(generated), 1),
                        **cooldown}))
            if rows_checkpoint is not None:
                rows_checkpoint(rows)
            log("  %-18s %.2f s/token  %r%s" %
                (item["case"].id, rows[-1]["seconds_per_token"], answer,
                 "" if repeats == 1 else "  [repeat %d/%d]" % (repeat + 1, repeats)))
            del output, sequence, enc
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows, {"initialization_seconds": init_s, "dfloat11_model": model_id,
                  "cpu_offload": cpu_offload}


def run_deepspeed_zero_inference(method: Method, rendered: list[dict], n_tokens: int,
                 deadline: float,
                 rows_checkpoint: Callable[[list[dict]], None] | None = None,
                 repeats: int = 1, repeat_offset: int = 0,
                 warmup_tokens: int = 0,
                 ) -> tuple[list[dict], dict]:
    """DeepSpeed ZeRO-Inference (https://www.deepspeed.ai/2022/09/09/
    zero-inference.html): ZeRO-3 parameter partitioning applied at
    inference time, tiering live parameters across GPU VRAM, CPU RAM, and
    optionally NVMe -- the same GPU/RAM/(disk) memory-hierarchy category
    AirLLM and Accelerate occupy, confirmed against DeepSpeedExamples'
    own zero_inference reference implementation rather than assumed from
    its training-oriented ZeRO-Infinity sibling's marketing.

    deepspeed.initialize() conventionally runs under the `deepspeed` CLI
    launcher, which sets up RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/
    MASTER_PORT for a torch.distributed process group automatically. This
    project always launches a fresh plain-`python` subprocess per (block,
    method) cell regardless of method (see run_paper_comparison_worker.
    py's module docstring for why every method shares that isolation,
    not just this one), so those variables are set here directly for the
    single-process/single-GPU case the launcher would otherwise have
    configured (WORLD_SIZE=1, RANK=LOCAL_RANK=0) rather than adding a
    second, method-specific subprocess-launch path.
    """
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    import deepspeed
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.integrations.deepspeed import HfDeepSpeedConfig

    offload_device = method.overrides.get("offload_device", "cpu")
    if offload_device not in {"cpu", "nvme"}:
        raise ValueError("DeepSpeed offload_device must be 'cpu' or 'nvme'")
    pin_memory = method.overrides.get("pin_memory", True)
    offload_path = method.overrides.get("offload_path", DEEPSPEED_OFFLOAD_DIR)

    init_t0 = time.perf_counter()
    hf_config = AutoConfig.from_pretrained(MODEL)
    # stage3_*_bucket_size/threshold are sized off the model's own hidden
    # dimension, matching DeepSpeedExamples' own zero_inference reference
    # (2 * hidden_size**2 for the prefetch/live-parameter buckets,
    # hidden_size for the persistence threshold) rather than a value
    # copied from a different-sized model.
    hidden_size = getattr(hf_config, "hidden_size", 4096)
    offload_param = {"device": offload_device, "pin_memory": pin_memory}
    if offload_device == "nvme":
        pathlib.Path(offload_path).mkdir(parents=True, exist_ok=True)
        offload_param["nvme_path"] = str(offload_path)
        # AsyncPartitionedParameterSwapper's own buffer_size (default
        # 1e8 elements -- DeepSpeedZeroOffloadParamConfig) is a flat
        # constant, not sized to the model. At world_size=1 (this
        # project always runs a single process, see the RANK/
        # WORLD_SIZE note above) ZeRO-3 has no other rank to shard a
        # parameter across, so the embedding/lm_head weight (vocab_size
        # * hidden_size elements) swaps to NVMe as one whole chunk --
        # confirmed live: Qwen3-14B's 151936 * 5120 = 777,912,320
        # tripped the swapper's own "numel <= elements_per_buffer"
        # assertion against the 1e8 default. Size the buffer off the
        # model's own largest tensor instead, same principle as the
        # stage3_* buckets above, rather than DeepSpeedExamples'
        # reference fixed multi-GB buffers (tuned for bigger-memory
        # hosts than this project's 8 GB GPU / ~19 GiB WSL2 budget).
        # buffer_count is trimmed from DeepSpeed's own default of 5 to
        # 2: enough to double-buffer this single-stream generate loop
        # without paying for 5x this element count in pinned CPU RAM.
        vocab_size = getattr(hf_config, "vocab_size", 32000)
        offload_param["buffer_size"] = max(int(1e8), vocab_size * hidden_size)
        offload_param["buffer_count"] = 2
    ds_config = {
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 2 * hidden_size * hidden_size,
            "stage3_param_persistence_threshold": hidden_size,
            "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
            "offload_param": offload_param,
        },
    }
    # HfDeepSpeedConfig must be constructed BEFORE from_pretrained() is
    # called, not just before deepspeed.initialize() -- it patches
    # from_pretrained (via a weakref registry Transformers checks during
    # model construction) so the model is built already partitioned per
    # ds_config's ZeRO-3 settings. Skipping this and calling
    # deepspeed.initialize() only after a normal from_pretrained() is a
    # real bug this project hit by actually running it: from_pretrained
    # materializes the complete bf16 model on GPU first in that case (the
    # ~26 GB CUDA OOM this method raised on the reference 8 GB card came
    # from exactly that -- the model, not a ZeRO-3 shard of it, still sat
    # entirely in VRAM by the time deepspeed.initialize() ran).
    dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841 -- must stay alive (weakref-registered) through from_pretrained(), not merely constructed for a side effect at this line
    with torch.no_grad():
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module
    tokenizer = rendered[0]["tokenizer"]
    init_s = time.perf_counter() - init_t0
    rows = []

    def _pad_token_id() -> int:
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if isinstance(pad, (list, tuple)):
            pad = pad[0] if pad else None
        return int(pad) if pad is not None else 0

    try:
        if warmup_tokens > 0 and rendered:
            # Untimed, see run_airllm's identical rationale.
            enc = tokenizer(rendered[0]["prompt"], return_tensors="pt").to("cuda")
            with torch.no_grad():
                model.generate(**enc, max_new_tokens=warmup_tokens, eos_token_id=[],
                               pad_token_id=_pad_token_id(), do_sample=False,
                               use_cache=True)
            torch.cuda.synchronize()
            del enc
        for repeat, item in [(r, it) for r in range(repeats) for it in rendered]:
            if time.perf_counter() >= deadline:
                break
            enc = tokenizer(item["prompt"], return_tensors="pt").to("cuda")
            cooldown = cool_down(COOLDOWN_SECONDS, COOLDOWN_MAX_TEMPERATURE_C)
            cache = drop_caches()
            reset_cuda_peak()
            read0 = process_read_bytes()
            gpu_thermal_before = gpu_thermal_snapshot()
            with MemoryProbe() as probe:
                t0 = time.perf_counter()
                with torch.no_grad():
                    output = model.generate(
                        **enc, max_new_tokens=n_tokens, eos_token_id=[],
                        pad_token_id=_pad_token_id(), do_sample=False, use_cache=True,
                        return_dict_in_generate=True)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
            gpu_thermal_after = gpu_thermal_snapshot()
            mem_report = probe.report()
            peak_vram_gb, peak_vram_source = canonical_peak_vram(mem_report)
            read_bytes = process_read_bytes() - read0
            sequence = output.sequences if hasattr(output, "sequences") else output
            generated = sequence[0, enc["input_ids"].shape[1]:].tolist()
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            rows.append(result_row(
                item["case"], method, item["prompt"], item["input_tokens"],
                generated, answer, wall, peak_vram_gb,
                cache, {"generation_mode": "greedy",
                        "repeat": repeat_offset + repeat,
                        "offload_device": offload_device,
                        "peak_vram_source": peak_vram_source,
                        **memory_probe_extra_fields(mem_report),
                        "gpu_thermal_before": gpu_thermal_before,
                        "gpu_thermal": gpu_thermal_after,
                        "process_read_bytes": read_bytes,
                        "process_read_bytes_per_token": read_bytes / max(len(generated), 1),
                        **cooldown}))
            if rows_checkpoint is not None:
                rows_checkpoint(rows)
            log("  %-18s %.2f s/token  %r%s" %
                (item["case"].id, rows[-1]["seconds_per_token"], answer,
                 "" if repeats == 1 else "  [repeat %d/%d]" % (repeat + 1, repeats)))
            del output, sequence, enc
    finally:
        del model, ds_engine
        gc.collect()
        torch.cuda.empty_cache()
    return rows, {"initialization_seconds": init_s, "offload_device": offload_device,
                  "offload_path": str(offload_path) if offload_device == "nvme" else None,
                  "pin_memory": pin_memory}


def engine_for(method: Method, *, critical_profile: str | None = None,
               replay_plan: str | None = None, spec_state: str | None = None,
               learning: bool | None = None):
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    values = dict(method.overrides)
    if values.get("placement_policy") in {"profiled_knapsack", "critical_path"}:
        values["critical_path_profile"] = critical_profile
    if values.get("placement_policy") in {
            "replay_cem", "replay_qubo", "replay_extent_qubo"}:
        values["replay_plan_state"] = replay_plan
    if method.id in {"spec-hazard", "spec-neural"}:
        values["spec_policy_state"] = spec_state
        if learning is not None:
            values["spec_policy_learn"] = learning
    cfg = EngineConfig(**values)
    return StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg), cfg


def run_afterimage(method: Method, rendered: list[dict], n_tokens: int,
                   deadline: float, draft_model=None, critical_profile: str | None = None,
                   replay_plan: str | None = None,
                   spec_state: str | None = None,
                   burn_in_rendered: list[dict] | None = None,
                   burn_in_tokens: int = 0,
                   rows_checkpoint: Callable[[list[dict]], None] | None = None,
                   repeats: int = 1, repeat_offset: int = 0,
                   ) -> tuple[list[dict], dict]:
    init_t0 = time.perf_counter()
    engine, cfg = engine_for(method, critical_profile=critical_profile,
                             replay_plan=replay_plan, spec_state=spec_state)
    init_s = time.perf_counter() - init_t0
    tokenizer = rendered[0]["tokenizer"]
    rows = []
    burn_in = []
    try:
        for burn_index, item in enumerate(burn_in_rendered or []):
            if burn_in_tokens < 1 or time.perf_counter() >= deadline:
                break
            ids = tokenizer(item["prompt"], return_tensors="pt").input_ids.cuda()
            cache = drop_caches()
            engine.stats.reset()
            t0 = time.perf_counter()
            # Warm the *measured* path, not just any path: burn-in exists to
            # absorb first-call CUDA/Triton/allocator compilation before the
            # timed loop, and for draft_mode="model" (spec-fixed and its
            # variants) that compilation happens inside generate_adaptive's
            # draft/verify machinery, not inside plain greedy decode. A
            # burn-in that always called generate_greedy warmed the wrong
            # kernels for every speculative method, leaving that
            # compilation cost to land inside repeat/block 0's measured
            # seconds/token instead of being absorbed here.
            if cfg.draft_mode == "model":
                warm_generator = torch.Generator(device="cuda").manual_seed(
                    2_000_000 + burn_index)
                sequence, _ = engine.generate_adaptive(
                    ids, max_new_tokens=burn_in_tokens, draft_model=draft_model,
                    temperature=0.0, generator=warm_generator)
            else:
                sequence = engine.generate_greedy(
                    ids, max_new_tokens=burn_in_tokens, use_cache=True)
            torch.cuda.synchronize()
            burn_in.append({
                "case_id": item["case"].id,
                "wall_seconds": time.perf_counter() - t0,
                "output_tokens": sequence.shape[1] - ids.shape[1],
                "cache_drop_succeeded": cache[0],
                "prefetch_controller_state": (
                    engine._prefetch_controller.state_dict()
                    if hasattr(engine._prefetch_controller, "state_dict") else None),
            })
            del sequence, ids
        # Repeats are the outer dimension so a truncated run (deadline hit)
        # still holds a complete sweep of every case for the repeats it did
        # finish. The per-case generator seed deliberately does NOT vary with
        # repeat: at temperature 0 generation is greedy, so identical seeds
        # make token IDs comparable across repeats and any mismatch a real
        # exactness failure rather than sampling noise. Repeats therefore
        # measure timing variance only, which is what they are for.
        for repeat, (case_index, item) in [
                (r, ci) for r in range(repeats) for ci in enumerate(rendered)]:
            if time.perf_counter() >= deadline:
                break
            ids = tokenizer(item["prompt"], return_tensors="pt").input_ids.cuda()
            cooldown = cool_down(COOLDOWN_SECONDS, COOLDOWN_MAX_TEMPERATURE_C)
            cache = drop_caches()
            engine.stats.reset()
            reset_cuda_peak()
            read0 = process_read_bytes()
            gpu_thermal_before = gpu_thermal_snapshot()
            with MemoryProbe() as probe:
                t0 = time.perf_counter()
                policy = None
                if cfg.draft_mode == "model":
                    generator = torch.Generator(device="cuda").manual_seed(1000 + case_index)
                    sequence, policy = engine.generate_adaptive(
                        ids, max_new_tokens=n_tokens, draft_model=draft_model,
                        temperature=0.0, generator=generator)
                else:
                    sequence = engine.generate_greedy(ids, max_new_tokens=n_tokens, use_cache=True)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
            gpu_thermal_after = gpu_thermal_snapshot()
            mem_report = probe.report()
            peak_vram_gb, peak_vram_source = canonical_peak_vram(mem_report)
            read_bytes = process_read_bytes() - read0
            generated = sequence[0, ids.shape[1]:].tolist()
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            stats = engine.stats
            # A normal greedy decode performs one target sweep per committed
            # token. Speculative execution records its verification sweeps
            # directly. Keep the raw spec_sweeps counter for diagnostics, but
            # expose a common target-sweep metric so k=0 is not incorrectly
            # reported as zero sweeps (or N tokens per sweep via max(0, 1)).
            target_sweeps = (
                stats.spec_sweeps
                if cfg.draft_mode == "model" else len(generated)
            )
            output_tokens = max(len(generated), 1)
            safe_target_sweeps = max(target_sweeps, 1)
            extra = {
                "generation_mode": "speculative_greedy" if cfg.draft_mode != "none" else "greedy",
                "repeat": repeat_offset + repeat,
                "config": cfg.to_dict(),
                "config_fingerprint": cfg.fingerprint(),
                "exactness_contract": cfg.exactness_contract,
                # bytes_read/gb_read_per_token: Afterimage's own logical
                # count of compressed bytes it asked the storage layer for
                # (engine.stats). process_read_bytes: the OS's own /proc/
                # self/io count for the same cell -- comparable across
                # every method, Afterimage included, unlike the logical
                # count which only Afterimage's own engine can report.
                "bytes_read": stats.bytes_read,
                "gb_read_per_token": stats.bytes_read / 1e9 / output_tokens,
                # These three fields make the Paper 1 mechanism identity
                # directly checkable from every row:
                # bytes/output-token = bytes/target-sweep *
                # target-sweeps/output-token.
                "target_sweeps": target_sweeps,
                "target_sweeps_per_output_token": target_sweeps / output_tokens,
                "target_storage_bytes_per_sweep": (
                    stats.bytes_read / safe_target_sweeps),
                "target_storage_bytes_per_output_token": (
                    stats.bytes_read / output_tokens),
                "process_read_bytes": read_bytes,
                "process_read_bytes_per_token": read_bytes / max(len(generated), 1),
                "io_seconds": stats.io_seconds,
                "decode_seconds": stats.decode_seconds,
                "compute_seconds": stats.compute_seconds,
                "h2d_seconds": stats.h2d_seconds,
                "h2d_bytes": stats.h2d_bytes,
                "draft_seconds": stats.draft_seconds,
                "target_seconds": stats.target_seconds,
                "breakdown_timing_mode": (
                    "trace_synchronized" if cfg.trace_events
                    else "aggregate_unsynchronized"),
                "prefetch_hits": stats.prefetch_hits,
                "prefetch_misses": stats.prefetch_misses,
                "prefetch_wait_seconds": stats.prefetch_wait_seconds,
                "prefetch_peak_inflight_bytes": stats.prefetch_peak_inflight_bytes,
                "storage_read_calls": stats.storage_read_calls,
                "storage_extent_bytes": stats.storage_extent_bytes,
                "gpu_thermal_before": gpu_thermal_before,
                "gpu_thermal": gpu_thermal_after,
                **cooldown,
                "pageable_ram_fallback_keys": sorted(
                    engine._ram_cache_pageable_keys),
                "tier_assignment_fingerprint": sha256_json(engine._tier),
                "tier_counts": {
                    tier: sum(value == tier for value in engine._tier.values())
                    for tier in ("vram", "ram", "disk", "row_gather")
                },
                "representation_summary": engine.representation_summary(),
                "final_prefetch_depth": engine._prefetch_controller.choose_depth(),
                "prefetch_controller_state": (
                    engine._prefetch_controller.state_dict()
                    if hasattr(engine._prefetch_controller, "state_dict") else None),
                "spec_sweeps": stats.spec_sweeps,
                "spec_accepted_tokens": stats.spec_accepted_tokens,
                "spec_cache_crops": stats.spec_cache_crops,
                "spec_cached_prefix_tokens": stats.spec_cached_prefix_tokens,
                "tokens_per_target_sweep": len(generated) / safe_target_sweeps,
                "policy_state": policy.state_dict() if policy is not None else None,
                "mips_certified": stats.mips_certified,
                "mips_fallbacks": stats.mips_fallbacks,
                "mips_rows_evaluated": stats.mips_rows_evaluated,
                "mips_rows_pruned": stats.mips_rows_pruned,
                "peak_vram_source": peak_vram_source,
                **memory_probe_extra_fields(mem_report),
            }
            rows.append(result_row(
                item["case"], method, item["prompt"], item["input_tokens"],
                generated, answer, wall, peak_vram_gb,
                cache, extra))
            if rows_checkpoint is not None:
                rows_checkpoint(rows)
            log("  %-18s %.2f s/token  %r%s" %
                (item["case"].id, rows[-1]["seconds_per_token"], answer,
                 "" if repeats == 1 else "  [repeat %d/%d]" % (repeat + 1, repeats)))
            del sequence, ids
    finally:
        index_build_s = engine.stats.mips_index_build_seconds
        index_bytes = engine.mips_index_bytes
        engine.close()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    return rows, {"initialization_seconds": init_s,
                  "burn_in": burn_in,
                  "mips_index_build_seconds": index_build_s,
                  "mips_index_bytes": index_bytes,
                  "resolved_config": cfg.to_dict()}


def calibration_item(tokenizer, case: PromptCase) -> dict:
    prompt = render_chat_prompt(tokenizer, case)
    return {"case": case, "prompt": prompt,
            "input_tokens": tokenizer(prompt, return_tensors="pt").input_ids.shape[1],
            "tokenizer": tokenizer}


def prepare_critical_profile(tokenizer, temp_dir: pathlib.Path, deadline: float) -> dict:
    from afterimage.runtime.critical_path import CriticalPathProfile, TraceRecorder
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    traces = []
    calibration = []
    cases = prompt_cases("calibration")
    # "minimum-memory-tiering" needs enough VRAM headroom to materialize
    # this model's own largest tensor (the embedding/lm_head, vocab_size *
    # hidden_size) plus decode scratch -- not a flat constant tuned for one
    # model's vocabulary. 1.80 GB was fine for Qwen3-14B (vocab 151936,
    # hidden 5120 -> ~1.56 GB largest tensor) but genuinely infeasible for
    # Gemma-2-27B (vocab 256000, hidden 4608 -> ~2.36 GB): "VRAM budget is
    # infeasible: vram_budget 1.80 GB is below the 2.50 GB needed", found by
    # actually running this calibration step against Gemma, the same
    # category of gap as run_deepspeed_zero_inference's buffer_size fix
    # just above in this file, sized the same way -- off the model's own
    # config rather than a literal copied from a different-sized model.
    # +0.4 GB covers the decode-scratch term this control's own
    # decode_slice_elems=1<<20 adds (observed ~0.14 GB on Gemma; not
    # assumed universal, so kept as headroom rather than the exact figure).
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(MODEL)
    largest_tensor_gb = (getattr(hf_config, "vocab_size", 32000)
                        * getattr(hf_config, "hidden_size", 4096) * 2) / 1e9
    min_memory_vram_gb = max(1.80, largest_tensor_gb + 0.4)
    controls = [
        ("legacy-layer-streaming", EngineConfig(
            io_prefetch_depth=2, trace_events=True,
            trace_output=str(temp_dir / "critical_legacy_trace.json"))),
        ("minimum-memory-tiering", EngineConfig(
            vram_budget_gb=min_memory_vram_gb, decode_slice_elems=1 << 20,
            io_prefetch_depth=2, trace_events=True,
            trace_output=str(temp_dir / "critical_min_trace.json"))),
    ]
    for index, (label, cfg) in enumerate(controls):
        if time.perf_counter() >= deadline:
            raise TimeoutError("time budget expired during critical-path calibration")
        item = calibration_item(tokenizer, cases[index % len(cases)])
        engine = StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg)
        ids = tokenizer(item["prompt"], return_tensors="pt").input_ids.cuda()
        cache = drop_caches()
        engine.stats.reset()
        t0 = time.perf_counter()
        sequence = engine.generate_greedy(ids, max_new_tokens=1)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        engine.close()
        events = TraceRecorder.load(cfg.trace_output)
        traces.append(events)
        calibration.append({"control": label, "case_id": item["case"].id,
                            "wall_seconds": wall, "event_count": len(events),
                            "cache_drop_succeeded": cache[0]})
        del engine, sequence, ids
        gc.collect()
        torch.cuda.empty_cache()
    profile = CriticalPathProfile.from_traces(traces)
    profile_path = temp_dir / "critical_path_profile.json"
    profile.save(profile_path)
    payload = json.loads(profile_path.read_text())
    return {"path": str(profile_path), "trace_paths": [
                str(temp_dir / "critical_legacy_trace.json"),
                str(temp_dir / "critical_min_trace.json")],
            "profile": payload,
            "sha256": sha256_json(payload), "calibration_trials": calibration}


def prepare_spec_state(tokenizer, draft_model, temp_dir: pathlib.Path,
                       deadline: float, n_tokens: int, method_id: str) -> dict:
    state_path = temp_dir / (method_id + "_state.json")
    method = METHODS[method_id]
    engine, cfg = engine_for(method, spec_state=str(state_path), learning=True)
    calibration = []
    cases = prompt_cases("calibration")
    if method_id == "spec-neural" and len(cases) >= 4:
        training_cases, gate_cases = cases[:-2], cases[-2:]
        minimum_observations = 200
    else:
        training_cases, gate_cases = cases, ()
        minimum_observations = 0
    round_index = 0
    last_observations = 0
    try:
        while True:
            for case_index, case in enumerate(training_cases):
                if time.perf_counter() >= deadline:
                    break
                item = calibration_item(tokenizer, case)
                ids = tokenizer(item["prompt"], return_tensors="pt").input_ids.cuda()
                cache = drop_caches()
                engine.stats.reset()
                t0 = time.perf_counter()
                generator = torch.Generator(device="cuda").manual_seed(
                    2000 + round_index * len(training_cases) + case_index)
                sequence, policy = engine.generate_adaptive(
                    ids, max_new_tokens=n_tokens, draft_model=draft_model,
                    temperature=0.0, generator=generator)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                generated = sequence.shape[1] - ids.shape[1]
                state = policy.state_dict()
                last_observations = int(state.get("n_observations", 0))
                calibration.append({
                    "round": round_index, "case_id": case.id,
                    "wall_seconds": wall, "output_tokens": generated,
                    "seconds_per_token": wall / max(generated, 1),
                    "tokens_per_target_sweep": (
                        generated / max(engine.stats.spec_sweeps, 1)),
                    "policy_state": state,
                    "cache_drop_succeeded": cache[0],
                })
                del ids, sequence
                if minimum_observations and last_observations >= minimum_observations:
                    break
            round_index += 1
            if (not minimum_observations
                    or last_observations >= minimum_observations
                    or time.perf_counter() >= deadline):
                break
    finally:
        engine.close()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    payload = json.loads(state_path.read_text())

    gate_trials = []
    if gate_cases and time.perf_counter() < deadline:
        gate_engine, _ = engine_for(
            method, spec_state=str(state_path), learning=False)
        try:
            for case_index, case in enumerate(gate_cases):
                if time.perf_counter() >= deadline:
                    break
                item = calibration_item(tokenizer, case)
                ids = tokenizer(item["prompt"], return_tensors="pt").input_ids.cuda()
                cache = drop_caches()
                gate_engine.stats.reset()
                generator = torch.Generator(device="cuda").manual_seed(9000 + case_index)
                sequence, policy = gate_engine.generate_adaptive(
                    ids, max_new_tokens=n_tokens, draft_model=draft_model,
                    temperature=0.0, generator=generator)
                state = policy.state_dict()
                gate_trials.append({
                    "case_id": case.id,
                    "output_tokens": sequence.shape[1] - ids.shape[1],
                    "decision_stops": int(state.get("decision_stops", 0)),
                    "decision_continues": int(state.get("decision_continues", 0)),
                    "last_stop_position": state.get("last_stop_position"),
                    "cache_drop_succeeded": cache[0],
                })
                del ids, sequence
        finally:
            gate_engine.close()
            del gate_engine
            gc.collect()
            torch.cuda.empty_cache()

    stops = sum(row["decision_stops"] for row in gate_trials)
    continues = sum(row["decision_continues"] for row in gate_trials)
    opportunities = stops + continues
    action_rate = stops / opportunities if opportunities else 0.0
    mechanism_gate = {
        "minimum_observations": minimum_observations,
        "calibration_observations": int(
            payload.get("state", {}).get("n_observations", last_observations)),
        "gate_opportunities": opportunities,
        "decision_stops": stops,
        "decision_continues": continues,
        "action_divergence_rate": action_rate,
        "required_action_divergence_rate": 0.10,
        "passed": (not minimum_observations or (
            int(payload.get("state", {}).get("n_observations", 0))
            >= minimum_observations and action_rate >= 0.10)),
    }
    return {"path": str(state_path), "state": payload,
            "sha256": sha256_json(payload), "calibration_trials": calibration,
            "gate_trials": gate_trials, "mechanism_gate": mechanism_gate,
            "resolved_config": cfg.to_dict()}


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"completed_cases": 0}
    total_wall = sum(row["wall_seconds"] for row in rows)
    total_tokens = sum(row["output_tokens"] for row in rows)
    # A row's peak_vram_gb is None when neither nvidia-smi nor the torch
    # allocator produced a usable reading for that cell (see
    # canonical_peak_vram) -- max() over a mix of None and float raises,
    # and silently coercing to 0.0 is exactly the bug this replaces (a
    # failed reading is not "used no memory", it is "unmeasured"). If
    # every row in this method's cells is unmeasured, the summary's own
    # peak_vram_gb is honestly None too, not a fabricated number.
    valid_vram = [row["peak_vram_gb"] for row in rows if row["peak_vram_gb"] is not None]
    # expected_match_rate is only meaningful for cases that declare a real
    # expected_any -- most of this project's prompts (paper_generation
    # split, deliberately) declare none, since there is no single correct
    # continuation for an open-ended generation prompt. case.matches()
    # returns False when expected_any is empty (any() of nothing), which
    # is the right value for the per-row expected_match field, but
    # averaging that across a whole method silently reports "0% correct"
    # for a metric that was never applicable in the first place.
    applicable_rows = [row for row in rows if row.get("expected_any")]
    summary = {
        "completed_cases": len(rows),
        "total_output_tokens": total_tokens,
        "total_wall_seconds": total_wall,
        "seconds_per_token": total_wall / max(total_tokens, 1),
        "median_cell_seconds_per_token": statistics.median(
            row["seconds_per_token"] for row in rows),
        "peak_vram_gb": max(valid_vram) if valid_vram else None,
        "peak_vram_measured_cells": len(valid_vram),
        "peak_vram_unmeasured_cells": len(rows) - len(valid_vram),
        "expected_matches": (
            sum(bool(row["expected_match"]) for row in applicable_rows)
            if applicable_rows else None),
        "expected_match_rate": (
            statistics.mean(bool(row["expected_match"]) for row in applicable_rows)
            if applicable_rows else None),
        "all_cache_drops_succeeded": all(row["cache_drop_succeeded"] for row in rows),
    }
    summary.update(_repeat_dispersion(rows))
    summary.update(_thermal_integrity(rows))
    summary.update(_io_traffic(rows))
    summary.update(_target_traffic(rows))
    return summary


def _target_traffic(rows: list[dict]) -> dict:
    """Aggregate the two factors in Paper 1's target-traffic identity.

    External baselines do not expose Afterimage's logical target-engine byte
    counter, so their summaries omit these fields instead of fabricating zero.
    For Afterimage, totals are combined before division so rows with different
    output lengths are weighted by their actual tokens and sweeps.
    """
    observed = [row for row in rows if all(key in row for key in (
        "target_sweeps", "target_storage_bytes_per_output_token",
        "output_tokens"))]
    if not observed:
        return {}
    total_tokens = sum(int(row["output_tokens"]) for row in observed)
    total_sweeps = sum(int(row["target_sweeps"]) for row in observed)
    total_bytes = sum(
        float(row["target_storage_bytes_per_output_token"])
        * int(row["output_tokens"])
        for row in observed
    )
    safe_tokens = max(total_tokens, 1)
    safe_sweeps = max(total_sweeps, 1)
    return {
        "target_storage_bytes": total_bytes,
        "target_sweeps": total_sweeps,
        "target_storage_bytes_per_output_token": total_bytes / safe_tokens,
        "target_storage_bytes_per_sweep": total_bytes / safe_sweeps,
        "target_sweeps_per_output_token": total_sweeps / safe_tokens,
        "tokens_per_target_sweep": total_tokens / safe_sweeps,
    }


def _io_traffic(rows: list[dict]) -> dict:
    """Median OS-level process/storage read traffic per token
    (process_read_bytes_per_token, from /proc/self/io -- see
    process_read_bytes()'s own docstring for why this is not the same
    claim as confirmed physical NVMe bytes under WSL2). Legacy rows
    written before this field existed simply have nothing to summarize,
    which must read as "not measured", not as zero traffic.
    """
    values = [row["process_read_bytes_per_token"] for row in rows
             if "process_read_bytes_per_token" in row]
    if not values:
        return {"process_read_bytes_per_token_median": None}
    return {"process_read_bytes_per_token_median": statistics.median(values)}


def _thermal_integrity(rows: list[dict]) -> dict:
    """How many timed cells ran while the GPU was throttling.

    A throttled cell is not comparable to an unthrottled one, and on a
    laptop GPU a long campaign will throttle unless cooled between cells.
    Measured on this project's own reference machine, an uncooled Qwen3-14B
    run degraded 1.51x between its first and third repeat with SW Thermal
    Slowdown active -- large enough to invert a method ranking. Surfacing
    the count here means a contaminated run reports that fact next to its
    mean instead of looking clean.
    """
    seen = [row.get("gpu_thermal", {}).get("throttled") for row in rows]
    known = [value for value in seen if value is not None]
    if not known:
        return {"thermally_throttled_cells": None}
    throttled = sum(bool(value) for value in known)
    # Thermal and power are reported separately (see thermal_throttled/
    # power_limited in gpu_thermal_snapshot): a laptop hitting its steady-
    # state power cap during every cell is normal and not comparable to
    # genuine thermal throttling, and "thermally_clean" below should not
    # be tripped by ordinary power limiting.
    thermal_seen = [row.get("gpu_thermal", {}).get("thermal_throttled") for row in rows]
    thermal_known = [value for value in thermal_seen if value is not None]
    power_seen = [row.get("gpu_thermal", {}).get("power_limited") for row in rows]
    power_known = [value for value in power_seen if value is not None]
    temperatures = []
    for row in rows:
        try:
            temperatures.append(float(row.get("gpu_thermal", {}).get("temperature_c")))
        except (TypeError, ValueError):
            pass
    integrity = {
        "thermally_throttled_cells": throttled,
        "thermally_observed_cells": len(known),
        "thermally_clean": throttled == 0,
        "thermal_throttled_cells": (
            sum(bool(value) for value in thermal_known) if thermal_known else None),
        "power_limited_cells": (
            sum(bool(value) for value in power_known) if power_known else None),
    }
    if temperatures:
        integrity["gpu_temperature_c_min"] = min(temperatures)
        integrity["gpu_temperature_c_max"] = max(temperatures)
    return integrity


def _repeat_dispersion(rows: list[dict]) -> dict:
    """Across-repeat spread for the headline seconds/token, when a run used
    --repeats > 1.

    A single observation per cell cannot distinguish a real effect from
    run-to-run noise, and this suite's own noise table (docs/HOW_IT_WORKS.md)
    puts that noise near 4%. With repeats, each repeat contributes one
    complete sweep of every case, so the per-repeat seconds/token values are
    directly comparable and their spread is the quantity a reader needs to
    judge whether two methods actually differ.

    Reported as median plus min/max and, from three repeats up, the sample
    standard deviation and relative standard deviation. Deliberately not a
    confidence interval: three repeats is far too few for one to mean
    anything, and labelling it as such would invite exactly the overclaim
    this project's protocol exists to prevent.
    """
    by_repeat: dict[int, list[dict]] = {}
    for row in rows:
        by_repeat.setdefault(row.get("repeat", 0), []).append(row)
    if len(by_repeat) < 2:
        return {"repeats_completed": len(by_repeat)}

    per_repeat = {}
    for repeat, repeat_rows in sorted(by_repeat.items()):
        wall = sum(r["wall_seconds"] for r in repeat_rows)
        tokens = sum(r["output_tokens"] for r in repeat_rows)
        per_repeat[repeat] = {
            "cases": len(repeat_rows),
            "seconds_per_token": wall / max(tokens, 1),
        }
    values = [v["seconds_per_token"] for v in per_repeat.values()]
    dispersion = {
        "repeats_completed": len(by_repeat),
        "per_repeat_seconds_per_token": per_repeat,
        "repeat_median_seconds_per_token": statistics.median(values),
        "repeat_min_seconds_per_token": min(values),
        "repeat_max_seconds_per_token": max(values),
        # Complete sweeps only: a repeat cut short by the deadline has fewer
        # cases and is not comparable to a full one, so flag rather than
        # silently average an apples-to-oranges set.
        "all_repeats_complete": len({v["cases"] for v in per_repeat.values()}) == 1,
    }
    if len(values) >= 3:
        stdev = statistics.stdev(values)
        dispersion["repeat_stdev_seconds_per_token"] = stdev
        dispersion["repeat_relative_stdev"] = stdev / max(
            statistics.mean(values), 1e-12)
    return dispersion


def add_comparisons(result: dict) -> None:
    by_method = {entry["method_id"]: entry for entry in result["methods"]
                 if entry.get("rows")}
    air = by_method.get("airllm")
    exact = by_method.get("exact-min")
    for entry in by_method.values():
        summary = entry["summary"]
        if air:
            summary["speedup_vs_airllm"] = (
                air["summary"]["seconds_per_token"] / summary["seconds_per_token"])
            summary["vram_vs_airllm"] = (
                summary["peak_vram_gb"] / air["summary"]["peak_vram_gb"])
        if exact:
            reference = {row["case_id"]: row["output_token_ids"] for row in exact["rows"]}
            shared = [row for row in entry["rows"] if row["case_id"] in reference]
            summary["token_agreement_vs_exact_min"] = (
                statistics.mean(row["output_token_ids"] == reference[row["case_id"]]
                                for row in shared) if shared else None)


def checkpoint(path: pathlib.Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    global MODEL, DRAFT_MODEL, STORE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--draft-model", default=DRAFT_MODEL)
    parser.add_argument("--store", default=STORE)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                        help="comma-separated IDs; choices: %s" % ",".join(METHODS))
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--cooldown-seconds", type=float, default=0.0,
        help="minimum idle time before each timed cell (default 0). A "
             "sustained campaign heats a laptop GPU until it throttles, which "
             "makes later methods look slower than earlier ones purely from "
             "run order.")
    parser.add_argument(
        "--cooldown-max-temp-c", type=float, default=None,
        help="additionally wait before each timed cell until the GPU is at or "
             "below this temperature (capped at 10 minutes per cell). Use with "
             "--cooldown-seconds to set a floor as well as a ceiling.")
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="complete sweeps of every case per method (default 1). Each "
             "repeat re-drops the page cache per cell and contributes one "
             "seconds/token observation, so the summary can report spread "
             "across repeats instead of a single unreplicated number. "
             "Multiplies wall time; raise --time-budget-minutes to match.")
    parser.add_argument("--case-ids", default=None,
                        help="comma-separated evaluation case IDs; default is all")
    parser.add_argument("--time-budget-minutes", type=float, default=58.0)
    parser.add_argument(
        "--ram-overlay-vram-budget-gb", type=float, default=None,
        help="override the matched VRAM budget for exact-min and ram-overlay-head")
    parser.add_argument(
        "--ram-overlay-host-budget-gb", type=float, default=None,
        help="override the pinned-host budget and allocation gate for ram-overlay-head")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-dirty-tree", action="store_true",
        help="proceed even with uncommitted changes (git status --short is "
             "non-empty); the resulting result JSON is still recorded but is "
             "not reproducible from its git_commit alone -- do not treat it "
             "as evidence for a publishable claim")
    args = parser.parse_args()

    MODEL, DRAFT_MODEL, STORE = args.model, args.draft_model, args.store
    selected = [part.strip() for part in args.methods.split(",") if part.strip()]
    unknown = sorted(set(selected) - set(METHODS))
    if unknown:
        parser.error("unknown methods: %s" % ", ".join(unknown))
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must not be negative")
    global COOLDOWN_SECONDS, COOLDOWN_MAX_TEMPERATURE_C
    COOLDOWN_SECONDS = args.cooldown_seconds
    COOLDOWN_MAX_TEMPERATURE_C = args.cooldown_max_temp_c
    if args.ram_overlay_vram_budget_gb is not None:
        if args.ram_overlay_vram_budget_gb <= 0:
            parser.error("--ram-overlay-vram-budget-gb must be positive")
        for method_id in ("exact-min", "ram-overlay-head"):
            method = METHODS[method_id]
            METHODS[method_id] = dataclasses.replace(
                method,
                overrides={**method.overrides,
                           "vram_budget_gb": args.ram_overlay_vram_budget_gb})
    if args.ram_overlay_host_budget_gb is not None:
        if args.ram_overlay_host_budget_gb <= 0:
            parser.error("--ram-overlay-host-budget-gb must be positive")
        method = METHODS["ram-overlay-head"]
        METHODS["ram-overlay-head"] = dataclasses.replace(
            method,
            overrides={**method.overrides,
                       "ram_budget_gb": args.ram_overlay_host_budget_gb})

    out = pathlib.Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("refusing to overwrite immutable result: %s" % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    if partial.exists():
        raise FileExistsError("partial result already exists: %s" % partial)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the hardware comparison")

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    dirty = command_output(["git", "-C", str(repo_root), "status", "--short"])
    if dirty and not args.allow_dirty_tree:
        raise RuntimeError(
            "refusing to run with uncommitted changes (git status --short "
            "is non-empty): a result's git_commit only reproduces the code "
            "that produced it if the tree was clean. Commit or stash first, "
            "or pass --allow-dirty-tree for a deliberately non-reproducible "
            "local/debugging run.\n" + dirty)

    tokenizer = load_tokenizer(MODEL)
    evaluation_cases = prompt_cases("evaluation")
    if args.case_ids:
        requested_cases = [part.strip() for part in args.case_ids.split(",") if part.strip()]
        by_id = {case.id: case for case in evaluation_cases}
        unknown_cases = sorted(set(requested_cases) - set(by_id))
        if unknown_cases:
            parser.error("unknown evaluation cases: %s" % ", ".join(unknown_cases))
        evaluation_cases = tuple(by_id[case_id] for case_id in requested_cases)
    if not evaluation_cases:
        parser.error("at least one evaluation case is required")
    rendered = render_cases(tokenizer, evaluation_cases)
    for item in rendered:
        item["tokenizer"] = tokenizer

    started = time.perf_counter()
    deadline = started + args.time_budget_minutes * 60
    result = {
        "schema_version": 1,
        "status": "running",
        "exploratory": True,
        "evidence_level": "L1_mechanism_screen",
        "protocol_schema_version": 1,
        "confirmatory_protocol_satisfied": False,
        "prompt_suite_version": PROMPT_SUITE_VERSION,
        "prompt_suite": [dataclasses.asdict(case) for case in prompt_cases("all")],
        "evaluation_case_ids": [case.id for case in evaluation_cases],
        "calibration_case_ids": [case.id for case in prompt_cases("calibration")],
        "max_new_tokens": args.max_new_tokens,
        "repeats_requested": args.repeats,
        "cooldown_seconds": args.cooldown_seconds,
        "cooldown_max_temp_c": args.cooldown_max_temp_c,
        "time_budget_minutes": args.time_budget_minutes,
        "cache_regime": "cold page cache before every timed cell",
        "metric_definitions": {
            "expected_match_rate": (
                "fraction of cases whose bounded text completes the prompt's "
                "semantic expected prefix; this is a task score, not an "
                "execution-exactness score"),
            "token_agreement_vs_exact_min": (
                "fraction of shared cases with an identical complete output "
                "token-id sequence to the exact-min Afterimage control"),
            "target_sweeps": (
                "full target-model weight sweeps: one per committed token for "
                "greedy execution, or measured verification sweeps for "
                "speculative execution"),
            "target_storage_bytes_per_output_token": (
                "Afterimage logical target-engine storage bytes divided by "
                "committed output tokens"),
            "target_storage_bytes_per_sweep": (
                "Afterimage logical target-engine storage bytes divided by "
                "target sweeps"),
            "target_sweeps_per_output_token": (
                "target sweeps divided by committed output tokens; multiplying "
                "this by target_storage_bytes_per_sweep reconstructs "
                "target_storage_bytes_per_output_token"),
        },
        "model": MODEL,
        "draft_model": DRAFT_MODEL,
        "store": STORE,
        "selected_methods": selected,
        "ram_overlay_matched_vram_budget_gb": (
            METHODS["ram-overlay-head"].overrides["vram_budget_gb"]),
        "ram_overlay_host_budget_gb": (
            METHODS["ram-overlay-head"].overrides["ram_budget_gb"]),
        "environment": environment_manifest(repo_root, tokenizer, store=pathlib.Path(STORE)),
        "reproducible_from_commit": not bool(dirty),
        "calibration_artifacts": {},
        "methods": [],
        "failures": [],
    }
    checkpoint(partial, result)

    pin_preflight = None
    if "ram-overlay-head" in selected:
        from afterimage.runtime.memory_preflight import pinned_memory_preflight
        pin_preflight = pinned_memory_preflight(
            int(METHODS["ram-overlay-head"].overrides["ram_budget_gb"] * 1e9),
            attempt_allocation=True)
        result["calibration_artifacts"]["pinned_memory_preflight"] = (
            dataclasses.asdict(pin_preflight))
        checkpoint(partial, result)

    draft_model = None
    critical = None
    replay_plans = {}
    spec_states = {}
    with tempfile.TemporaryDirectory(prefix="afterimage-bounded-") as temp_name:
        temp_dir = pathlib.Path(temp_name)
        if any(METHODS[name].overrides.get("placement_policy") in {
                "critical_path", "profiled_knapsack", "replay_cem",
                "replay_qubo", "replay_extent_qubo"}
                for name in selected):
            log("\nCALIBRATION: critical-path profile")
            try:
                critical = prepare_critical_profile(tokenizer, temp_dir, deadline)
                result["calibration_artifacts"]["critical_path"] = {
                    key: value for key, value in critical.items()
                    if key not in ("path", "trace_paths")}
            except Exception as exc:
                result["failures"].append({"phase": "critical_path_calibration",
                                           "error": repr(exc),
                                           "traceback": traceback.format_exc()})
            checkpoint(partial, result)

        if critical is not None and any(
                method_id in selected for method_id in (
                    "replay-cem", "replay-qubo", "replay-extent-qubo")):
            from afterimage.runtime.critical_path import TraceRecorder
            from afterimage.runtime.replay_planner import (
                optimize_extent_qubo_residency, optimize_qubo_residency,
                optimize_replay_residency,
            )
            manifest = json.loads(
                (pathlib.Path(STORE) / "manifest.json").read_text(encoding="utf-8"))
            traces = [TraceRecorder.load(path) for path in critical["trace_paths"]]
            for method_id in ("replay-cem", "replay-qubo", "replay-extent-qubo"):
                if method_id not in selected:
                    continue
                log("\nCALIBRATION: %s whole-set plan" % method_id)
                try:
                    if method_id == "replay-qubo":
                        replay_plan = optimize_qubo_residency(
                            manifest, traces, vram_budget_gb=4.0,
                            decode_slice_elems=1 << 22,
                            pairwise_candidates=24, restarts=8,
                            sweeps=2000, seed=0)
                    elif method_id == "replay-extent-qubo":
                        replay_plan = optimize_extent_qubo_residency(
                            manifest, traces, vram_budget_gb=4.0,
                            decode_slice_elems=1 << 22,
                            max_extent_bytes=1 << 28, max_gap_bytes=0,
                            max_tensors_per_extent=8,
                            pairwise_candidates=24, restarts=8,
                            sweeps=2000, seed=0)
                    else:
                        replay_plan = optimize_replay_residency(
                            manifest, traces, vram_budget_gb=4.0,
                            decode_slice_elems=1 << 22, iterations=8,
                            population=40, seed=0)
                    replay_path = temp_dir / (method_id.replace("-", "_") + "_plan.json")
                    replay_plan.save(replay_path)
                    replay_plans[method_id] = {
                        "path": str(replay_path),
                        "plan": dataclasses.asdict(replay_plan),
                    }
                    result["calibration_artifacts"][method_id.replace("-", "_")] = (
                        replay_plans[method_id]["plan"])
                except Exception as exc:
                    result["failures"].append({
                        "phase": method_id.replace("-", "_") + "_calibration",
                        "error": repr(exc), "traceback": traceback.format_exc(),
                    })
                checkpoint(partial, result)

        for method_id in selected:
            method = METHODS[method_id]
            if time.perf_counter() >= deadline:
                result["failures"].append({"method": method_id,
                                           "error": "not started: time budget exhausted"})
                continue
            if (method_id == "ram-overlay-head"
                    and (pin_preflight is None or not pin_preflight.success)):
                result["failures"].append({
                    "method": method_id,
                    "error": "not started: pinned-memory mechanism gate failed",
                    "mechanism_gate": (dataclasses.asdict(pin_preflight)
                                       if pin_preflight is not None else None),
                })
                continue
            if (method.overrides.get("placement_policy") in {
                    "critical_path", "profiled_knapsack"}
                    and critical is None):
                result["failures"].append({"method": method_id,
                                           "error": "not started: calibration failed"})
                continue
            if method_id in {
                    "replay-cem", "replay-qubo", "replay-extent-qubo"
                    } and method_id not in replay_plans:
                result["failures"].append({"method": method_id,
                                           "error": "not started: replay plan failed"})
                continue
            if method_id in {"replay-qubo", "replay-extent-qubo"}:
                report = replay_plans[method_id]["plan"]["report"]
                if (not report["treatment_diverged"]
                        or report["optimized_over_control"] < 0.02):
                    result["failures"].append({
                        "method": method_id,
                        "error": "not started: QUBO plan mechanism gate failed",
                        "mechanism_gate": {
                            "treatment_diverged": report["treatment_diverged"],
                            "optimized_over_control": report["optimized_over_control"],
                            "required_replay_gain": 0.02,
                        },
                    })
                    continue
            # Route this from the method's actual execution contract rather
            # than a hand-maintained ID allowlist. The fixed-k ablation IDs
            # (spec-k2/spec-k4/spec-k16) also use draft_mode="model" and were
            # previously passed draft_model=None by this runner even though
            # the isolated paper worker handled them correctly.
            if (method.overrides.get("draft_mode") == "model"
                    and draft_model is None):
                from afterimage.runtime.streaming_engine import load_draft_model
                log("\nLoading resident draft model %s" % DRAFT_MODEL)
                draft_model = load_draft_model(DRAFT_MODEL, device="cuda")
            if method_id in {"spec-hazard", "spec-neural"} and method_id not in spec_states:
                log("\nCALIBRATION: %s state (disjoint prompts)" % method_id)
                try:
                    spec_states[method_id] = prepare_spec_state(
                        tokenizer, draft_model, temp_dir, deadline,
                        n_tokens=max(16 if method_id == "spec-neural" else 4,
                                     args.max_new_tokens),
                        method_id=method_id)
                    result["calibration_artifacts"][method_id] = {
                        key: value for key, value in spec_states[method_id].items()
                        if key != "path"}
                except Exception as exc:
                    result["failures"].append({"phase": "speculation_calibration",
                                               "error": repr(exc),
                                               "traceback": traceback.format_exc()})
                    checkpoint(partial, result)
                    continue
                if (method_id == "spec-neural"
                        and not spec_states[method_id]["mechanism_gate"]["passed"]):
                    result["failures"].append({
                        "method": method_id,
                        "error": "not started: neural action-divergence gate failed",
                        "mechanism_gate": spec_states[method_id]["mechanism_gate"],
                    })
                    checkpoint(partial, result)
                    continue

            log("\nMETHOD: %s" % method.title)
            entry = {"method_id": method.id, "title": method.title,
                     "declared_exactness": method.exactness, "rows": []}
            result["methods"].append(entry)

            def save_interim_rows(rows: list[dict]) -> None:
                entry["rows"] = list(rows)
                entry["summary"] = aggregate(rows)
                entry["interim"] = True
                result["elapsed_seconds"] = time.perf_counter() - started
                add_comparisons(result)
                checkpoint(partial, result)

            method_t0 = time.perf_counter()
            try:
                if method.kind == "airllm":
                    rows, metadata = run_airllm(method, rendered, args.max_new_tokens,
                                                deadline, save_interim_rows,
                                                repeats=args.repeats)
                elif method.kind == "accelerate":
                    rows, metadata = run_accelerate(
                        method, rendered, args.max_new_tokens, deadline,
                        save_interim_rows, repeats=args.repeats)
                elif method.kind == "dfloat11":
                    rows, metadata = run_dfloat11(
                        method, rendered, args.max_new_tokens, deadline,
                        save_interim_rows, repeats=args.repeats)
                else:
                    rows, metadata = run_afterimage(
                        method, rendered, args.max_new_tokens, deadline,
                        draft_model=draft_model,
                        critical_profile=critical["path"] if critical else None,
                        replay_plan=(replay_plans[method_id]["path"]
                                     if method_id in replay_plans else None),
                        spec_state=(spec_states[method_id]["path"]
                                    if method_id in spec_states else None),
                        rows_checkpoint=save_interim_rows,
                        repeats=args.repeats)
                entry["rows"] = rows
                entry["metadata"] = metadata
                entry["summary"] = aggregate(rows)
            except Exception as exc:
                entry["error"] = repr(exc)
                entry["traceback"] = traceback.format_exc()
                entry["summary"] = aggregate(entry["rows"])
                result["failures"].append({"method": method.id, "error": repr(exc)})
                log("  FAILED: %r" % exc)
            entry.pop("interim", None)
            entry["method_wall_seconds"] = time.perf_counter() - method_t0
            result["elapsed_seconds"] = time.perf_counter() - started
            add_comparisons(result)
            checkpoint(partial, result)

    if draft_model is not None:
        del draft_model
        gc.collect()
        torch.cuda.empty_cache()
    add_comparisons(result)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = ("time_capped" if result["elapsed_seconds"] >=
                        args.time_budget_minutes * 60 else "complete")
    result["completed_at_unix"] = time.time()
    checkpoint(partial, result)
    partial.replace(out)
    log("\nwrote immutable result %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
