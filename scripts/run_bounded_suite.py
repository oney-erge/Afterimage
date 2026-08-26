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


METHODS = {
    "airllm": Method("airllm", _installed_airllm_title(), "airllm", {},
                     "reference_greedy", 30.0),
    "accelerate": Method(
        "accelerate", _installed_accelerate_title(), "accelerate",
        {"gpu_memory": "1500MB", "cpu_memory": "8GB"},
        "reference_greedy", 30.0),
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


def reset_cuda_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


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
        "enforced.power.limit,clocks_throttle_reasons.active",
        "--format=csv,noheader,nounits"])
    if not raw:
        return {}
    parts = [p.strip() for p in raw.strip().split(",")]
    if len(parts) < 6:
        return {"raw": raw.strip()}
    keys = ("sm_clock_mhz", "mem_clock_mhz", "temperature_c", "power_draw_w",
            "power_limit_w", "throttle_reasons_active")
    snapshot = dict(zip(keys, parts, strict=True))
    snapshot["throttled"] = is_throttled(snapshot)
    return snapshot


# NVML clocks-event-reason bits that mean the GPU is being held below its
# requested clocks *while doing work*. GpuIdle (0x1) and the applications /
# display clock settings are deliberately excluded: they are not a
# performance confound, they are normal states.
_THROTTLE_MASK = (
    0x0000000000000004  # SwPowerCap
    | 0x0000000000000008  # HwSlowdown
    | 0x0000000000000020  # SwThermalSlowdown
    | 0x0000000000000040  # HwThermalSlowdown
    | 0x0000000000000080  # HwPowerBrakeSlowdown
)


def is_throttled(snapshot: dict) -> bool | None:
    """Whether a thermal/power throttle was active in this snapshot.

    A sustained campaign on a laptop GPU will throttle, and a throttled cell
    is not comparable to an unthrottled one. Recording the raw reason bits
    is not enough on its own: nobody reads hex when scanning a result file,
    so the decoded boolean is what makes the confound visible.
    """
    raw = snapshot.get("throttle_reasons_active")
    if raw in (None, "", "[N/A]", "N/A"):
        return None
    try:
        return bool(int(str(raw), 16) & _THROTTLE_MASK)
    except ValueError:
        return None


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
    throttled run. This additionally requires ``is_throttled()`` to read
    False before considering the GPU recovered, regardless of temperature.
    """
    if seconds <= 0 and max_temperature_c is None:
        return {}
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
        if max_temperature_c is not None and temperature is not None:
            reached = temperature <= max_temperature_c and throttled is not True
            # Honour the floor wait even once cool, so a fast-cooling run
            # still gets a consistent inter-cell gap.
            if reached and now >= deadline:
                break
        elif now >= deadline:
            break
        if now >= deadline and max_temperature_c is None:
            break
        # Hard ceiling: never wait more than 10 minutes for a GPU that is
        # simply not cooling, and say so in the record rather than hanging.
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
            "airllm", "transformers", "accelerate", "safetensors", "numpy")},
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
               peak_vram_gb: float, cache_drop: tuple[bool, str | None],
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
               repeats: int = 1,
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
            t0 = time.perf_counter()
            # An empty EOS list forces a fixed token count. Transformers 5.x
            # still needs a concrete pad ID when EOS is empty, otherwise it
            # indexes eos_token_tensor[0] while preparing special tokens.
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
            sequence = output.sequences if hasattr(output, "sequences") else output
            generated = sequence[0, ids.shape[1]:].tolist()
            answer = model.tokenizer.decode(generated, skip_special_tokens=True)
            rows.append(result_row(
                item["case"], method, item["prompt"], item["input_tokens"],
                generated, answer, wall, torch.cuda.max_memory_allocated() / 1e9,
                cache, {"generation_mode": "greedy", "repeat": repeat,
                        "gpu_thermal": gpu_thermal_snapshot(), **cooldown}))
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
                   repeats: int = 1,
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
        for repeat, item in [(r, it) for r in range(repeats) for it in rendered]:
            if time.perf_counter() >= deadline:
                break
            cooldown = cool_down(COOLDOWN_SECONDS, COOLDOWN_MAX_TEMPERATURE_C)
            cache = drop_caches()
            result = baseline.generate(item["prompt"], n_tokens)
            generated = result["output_token_ids"]
            rows.append(result_row(
                item["case"], method, item["prompt"], item["input_tokens"],
                generated, result["text"], result["wall_seconds"],
                result["peak_vram_gb"], cache,
                {"generation_mode": "greedy", "repeat": repeat,
                 "device_map": baseline.device_map,
                 "offload_dir": baseline.offload_dir,
                 "gpu_memory_limit": method.overrides["gpu_memory"],
                 "cpu_memory_limit": method.overrides["cpu_memory"],
                 "gpu_thermal": gpu_thermal_snapshot(), **cooldown}))
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
                   repeats: int = 1,
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
            generated = sequence[0, ids.shape[1]:].tolist()
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            stats = engine.stats
            extra = {
                "generation_mode": "speculative_greedy" if cfg.draft_mode != "none" else "greedy",
                "repeat": repeat,
                "config": cfg.to_dict(),
                "config_fingerprint": cfg.fingerprint(),
                "exactness_contract": cfg.exactness_contract,
                "bytes_read": stats.bytes_read,
                "gb_read_per_token": stats.bytes_read / 1e9 / max(len(generated), 1),
                "io_seconds": stats.io_seconds,
                "decode_seconds": stats.decode_seconds,
                "compute_seconds": stats.compute_seconds,
                "prefetch_hits": stats.prefetch_hits,
                "prefetch_misses": stats.prefetch_misses,
                "prefetch_wait_seconds": stats.prefetch_wait_seconds,
                "prefetch_peak_inflight_bytes": stats.prefetch_peak_inflight_bytes,
                "storage_read_calls": stats.storage_read_calls,
                "storage_extent_bytes": stats.storage_extent_bytes,
                "gpu_thermal": gpu_thermal_snapshot(),
                **cooldown,
                "pageable_ram_fallback_keys": sorted(
                    engine._ram_cache_pageable_keys),
                "tier_assignment_fingerprint": sha256_json(engine._tier),
                "tier_counts": {
                    tier: sum(value == tier for value in engine._tier.values())
                    for tier in ("vram", "ram", "disk", "row_gather")
                },
                "final_prefetch_depth": engine._prefetch_controller.choose_depth(),
                "prefetch_controller_state": (
                    engine._prefetch_controller.state_dict()
                    if hasattr(engine._prefetch_controller, "state_dict") else None),
                "spec_sweeps": stats.spec_sweeps,
                "spec_accepted_tokens": stats.spec_accepted_tokens,
                "spec_cache_crops": stats.spec_cache_crops,
                "spec_cached_prefix_tokens": stats.spec_cached_prefix_tokens,
                "tokens_per_target_sweep": len(generated) / max(stats.spec_sweeps, 1),
                "policy_state": policy.state_dict() if policy is not None else None,
                "mips_certified": stats.mips_certified,
                "mips_fallbacks": stats.mips_fallbacks,
                "mips_rows_evaluated": stats.mips_rows_evaluated,
                "mips_rows_pruned": stats.mips_rows_pruned,
            }
            rows.append(result_row(
                item["case"], method, item["prompt"], item["input_tokens"],
                generated, answer, wall, torch.cuda.max_memory_allocated() / 1e9,
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
    controls = [
        ("legacy-layer-streaming", EngineConfig(
            io_prefetch_depth=2, trace_events=True,
            trace_output=str(temp_dir / "critical_legacy_trace.json"))),
        ("minimum-memory-tiering", EngineConfig(
            vram_budget_gb=1.80, decode_slice_elems=1 << 20,
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
    summary = {
        "completed_cases": len(rows),
        "total_output_tokens": total_tokens,
        "total_wall_seconds": total_wall,
        "seconds_per_token": total_wall / max(total_tokens, 1),
        "median_cell_seconds_per_token": statistics.median(
            row["seconds_per_token"] for row in rows),
        "peak_vram_gb": max(row["peak_vram_gb"] for row in rows),
        "expected_matches": sum(bool(row["expected_match"]) for row in rows),
        "expected_match_rate": statistics.mean(
            bool(row["expected_match"]) for row in rows),
        "all_cache_drops_succeeded": all(row["cache_drop_succeeded"] for row in rows),
    }
    summary.update(_repeat_dispersion(rows))
    summary.update(_thermal_integrity(rows))
    return summary


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
        if any(name in selected for name in (
                "critical-path", "profiled-knapsack", "replay-cem",
                "replay-qubo", "replay-extent-qubo", "spec-critical")):
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
            if method_id in {
                    "critical-path", "profiled-knapsack", "spec-critical"
                    } and critical is None:
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
            if method_id in {
                    "spec-fixed", "spec-critical", "spec-cached", "spec-hazard",
                    "spec-neural", "chunked-spec"} and draft_model is None:
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
