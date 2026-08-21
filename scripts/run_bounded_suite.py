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


@dataclasses.dataclass(frozen=True)
class Method:
    id: str
    title: str
    kind: str
    overrides: dict
    exactness: str
    estimated_s_per_token: float


METHODS = {
    "airllm": Method("airllm", "AirLLM 3.1.0", "airllm", {},
                     "reference_greedy", 30.0),
    "exact-min": Method(
        "exact-min", "Afterimage exact streaming, minimum-memory control", "afterimage",
        {"vram_budget_gb": 1.80, "decode_slice_elems": 1 << 20,
         "io_prefetch_depth": 2}, "reference_execution_equivalent", 31.0),
    "exact-resident": Method(
        "exact-resident", "Afterimage exact streaming + 4 GB residency", "afterimage",
        {"vram_budget_gb": 4.00, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2}, "reference_execution_equivalent", 20.0),
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
    "certified-mips": Method(
        "certified-mips", "Afterimage certified greedy MIPS head", "afterimage",
        {"io_prefetch_depth": 2, "lm_head_policy": "certified_mips"},
        "greedy_token_exact", 30.0),
    "spec-fixed": Method(
        "spec-fixed", "Afterimage fixed-k speculative decoding", "afterimage",
        {"vram_budget_gb": 2.70, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "draft_mode": "model", "spec_k": 8,
         "spec_k_policy": "fixed"}, "greedy_token_exact_at_temperature_zero", 8.0),
    "spec-hazard": Method(
        "spec-hazard", "Afterimage frozen rejection-hazard stopping", "afterimage",
        {"vram_budget_gb": 2.70, "decode_slice_elems": 1 << 22,
         "io_prefetch_depth": 2, "draft_mode": "model", "spec_k": 8,
         "spec_k_policy": "hazard_cost", "spec_policy_learn": False},
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


def environment_manifest(repo_root: pathlib.Path, tokenizer) -> dict:
    gpu = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu.name,
        "gpu_total_bytes": gpu.total_memory,
        "driver": command_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ]),
        "host_memory": command_output(["free", "-b"]),
        "packages": {name: package_version(name) for name in (
            "airllm", "transformers", "accelerate", "safetensors", "numpy")},
        "git_commit": command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "-C", str(repo_root), "status", "--short"]),
        "tokenizer_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
    }


def render_cases(tokenizer, cases: tuple[PromptCase, ...]) -> list[dict]:
    rendered = []
    for case in cases:
        prompt = render_chat_prompt(tokenizer, case)
        input_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        rendered.append({"case": case, "prompt": prompt, "input_tokens": input_tokens})
    return rendered


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
               deadline: float) -> tuple[list[dict], dict]:
    from airllm import AutoModel

    init_t0 = time.perf_counter()
    model = AutoModel.from_pretrained(MODEL)
    init_s = time.perf_counter() - init_t0
    rows = []
    try:
        for item in rendered:
            if time.perf_counter() >= deadline:
                break
            enc = model.tokenizer(item["prompt"], return_tensors="pt", truncation=True)
            ids = enc["input_ids"].cuda()
            kwargs = {}
            if enc.get("attention_mask") is not None:
                kwargs["attention_mask"] = enc["attention_mask"].cuda()
            cache = drop_caches()
            reset_cuda_peak()
            t0 = time.perf_counter()
            output = model.generate(
                ids, max_new_tokens=n_tokens, eos_token_id=[],
                do_sample=False, use_cache=True, return_dict_in_generate=True, **kwargs)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            sequence = output.sequences if hasattr(output, "sequences") else output
            generated = sequence[0, ids.shape[1]:].tolist()
            answer = model.tokenizer.decode(generated, skip_special_tokens=True)
            rows.append(result_row(
                item["case"], method, item["prompt"], item["input_tokens"],
                generated, answer, wall, torch.cuda.max_memory_allocated() / 1e9,
                cache, {"generation_mode": "greedy"}))
            log("  %-18s %.2f s/token  %r" %
                (item["case"].id, rows[-1]["seconds_per_token"], answer))
            del output, sequence, ids
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows, {"initialization_seconds": init_s}


def engine_for(method: Method, *, critical_profile: str | None = None,
               hazard_state: str | None = None, learning: bool | None = None):
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    values = dict(method.overrides)
    if values.get("placement_policy") in {"profiled_knapsack", "critical_path"}:
        values["critical_path_profile"] = critical_profile
    if method.id == "spec-hazard":
        values["spec_policy_state"] = hazard_state
        if learning is not None:
            values["spec_policy_learn"] = learning
    cfg = EngineConfig(**values)
    return StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg), cfg


def run_afterimage(method: Method, rendered: list[dict], n_tokens: int,
                   deadline: float, draft_model=None, critical_profile: str | None = None,
                   hazard_state: str | None = None) -> tuple[list[dict], dict]:
    init_t0 = time.perf_counter()
    engine, cfg = engine_for(method, critical_profile=critical_profile,
                             hazard_state=hazard_state)
    init_s = time.perf_counter() - init_t0
    tokenizer = rendered[0]["tokenizer"]
    rows = []
    try:
        for case_index, item in enumerate(rendered):
            if time.perf_counter() >= deadline:
                break
            ids = tokenizer(item["prompt"], return_tensors="pt").input_ids.cuda()
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
                "final_prefetch_depth": engine._prefetch_controller.choose_depth(),
                "spec_sweeps": stats.spec_sweeps,
                "spec_accepted_tokens": stats.spec_accepted_tokens,
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
            log("  %-18s %.2f s/token  %r" %
                (item["case"].id, rows[-1]["seconds_per_token"], answer))
            del sequence, ids
    finally:
        index_build_s = engine.stats.mips_index_build_seconds
        index_bytes = engine.mips_index_bytes
        engine.close()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    return rows, {"initialization_seconds": init_s,
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
    return {"path": str(profile_path), "profile": payload,
            "sha256": sha256_json(payload), "calibration_trials": calibration}


def prepare_hazard_state(tokenizer, draft_model, temp_dir: pathlib.Path,
                         deadline: float, n_tokens: int) -> dict:
    state_path = temp_dir / "hazard_state.json"
    method = METHODS["spec-hazard"]
    engine, cfg = engine_for(method, hazard_state=str(state_path), learning=True)
    calibration = []
    try:
        for index, case in enumerate(prompt_cases("calibration")):
            if time.perf_counter() >= deadline:
                raise TimeoutError("time budget expired during hazard calibration")
            item = calibration_item(tokenizer, case)
            ids = tokenizer(item["prompt"], return_tensors="pt").input_ids.cuda()
            cache = drop_caches()
            engine.stats.reset()
            t0 = time.perf_counter()
            generator = torch.Generator(device="cuda").manual_seed(2000 + index)
            sequence, policy = engine.generate_adaptive(
                ids, max_new_tokens=n_tokens, draft_model=draft_model,
                temperature=0.0, generator=generator)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            generated = sequence.shape[1] - ids.shape[1]
            calibration.append({
                "case_id": case.id, "wall_seconds": wall,
                "output_tokens": generated,
                "seconds_per_token": wall / max(generated, 1),
                "tokens_per_target_sweep": generated / max(engine.stats.spec_sweeps, 1),
                "policy_state": policy.state_dict(),
                "cache_drop_succeeded": cache[0],
            })
            del ids, sequence
    finally:
        engine.close()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    payload = json.loads(state_path.read_text())
    return {"path": str(state_path), "state": payload,
            "sha256": sha256_json(payload), "calibration_trials": calibration,
            "resolved_config": cfg.to_dict()}


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"completed_cases": 0}
    total_wall = sum(row["wall_seconds"] for row in rows)
    total_tokens = sum(row["output_tokens"] for row in rows)
    return {
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
    parser.add_argument("--case-ids", default=None,
                        help="comma-separated evaluation case IDs; default is all")
    parser.add_argument("--time-budget-minutes", type=float, default=58.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    MODEL, DRAFT_MODEL, STORE = args.model, args.draft_model, args.store
    selected = [part.strip() for part in args.methods.split(",") if part.strip()]
    unknown = sorted(set(selected) - set(METHODS))
    if unknown:
        parser.error("unknown methods: %s" % ", ".join(unknown))
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    out = pathlib.Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("refusing to overwrite immutable result: %s" % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    if partial.exists():
        raise FileExistsError("partial result already exists: %s" % partial)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the hardware comparison")

    from transformers import AutoTokenizer
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
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
        "confirmatory_protocol_satisfied": False,
        "prompt_suite_version": PROMPT_SUITE_VERSION,
        "prompt_suite": [dataclasses.asdict(case) for case in prompt_cases("all")],
        "evaluation_case_ids": [case.id for case in evaluation_cases],
        "calibration_case_ids": [case.id for case in prompt_cases("calibration")],
        "max_new_tokens": args.max_new_tokens,
        "time_budget_minutes": args.time_budget_minutes,
        "cache_regime": "cold page cache before every timed cell",
        "model": MODEL,
        "draft_model": DRAFT_MODEL,
        "store": STORE,
        "selected_methods": selected,
        "environment": environment_manifest(repo_root, tokenizer),
        "calibration_artifacts": {},
        "methods": [],
        "failures": [],
    }
    checkpoint(partial, result)

    draft_model = None
    critical = None
    hazard = None
    with tempfile.TemporaryDirectory(prefix="afterimage-bounded-") as temp_name:
        temp_dir = pathlib.Path(temp_name)
        if any(name in selected for name in ("critical-path", "profiled-knapsack")):
            log("\nCALIBRATION: critical-path profile")
            try:
                critical = prepare_critical_profile(tokenizer, temp_dir, deadline)
                result["calibration_artifacts"]["critical_path"] = {
                    key: value for key, value in critical.items() if key != "path"}
            except Exception as exc:
                result["failures"].append({"phase": "critical_path_calibration",
                                           "error": repr(exc),
                                           "traceback": traceback.format_exc()})
            checkpoint(partial, result)

        for method_id in selected:
            method = METHODS[method_id]
            if time.perf_counter() >= deadline:
                result["failures"].append({"method": method_id,
                                           "error": "not started: time budget exhausted"})
                continue
            if method_id in {"critical-path", "profiled-knapsack"} and critical is None:
                result["failures"].append({"method": method_id,
                                           "error": "not started: calibration failed"})
                continue
            if method_id in {"spec-fixed", "spec-hazard", "chunked-spec"} and draft_model is None:
                from afterimage.runtime.streaming_engine import load_draft_model
                log("\nLoading resident draft model %s" % DRAFT_MODEL)
                draft_model = load_draft_model(DRAFT_MODEL, device="cuda")
            if method_id == "spec-hazard" and hazard is None:
                log("\nCALIBRATION: rejection-hazard state (disjoint prompts)")
                try:
                    hazard = prepare_hazard_state(
                        tokenizer, draft_model, temp_dir, deadline,
                        n_tokens=max(4, args.max_new_tokens))
                    result["calibration_artifacts"]["hazard_cost"] = {
                        key: value for key, value in hazard.items() if key != "path"}
                except Exception as exc:
                    result["failures"].append({"phase": "hazard_calibration",
                                               "error": repr(exc),
                                               "traceback": traceback.format_exc()})
                    checkpoint(partial, result)
                    continue

            log("\nMETHOD: %s" % method.title)
            entry = {"method_id": method.id, "title": method.title,
                     "declared_exactness": method.exactness, "rows": []}
            method_t0 = time.perf_counter()
            try:
                if method.kind == "airllm":
                    rows, metadata = run_airllm(method, rendered, args.max_new_tokens,
                                                deadline)
                else:
                    rows, metadata = run_afterimage(
                        method, rendered, args.max_new_tokens, deadline,
                        draft_model=draft_model,
                        critical_profile=critical["path"] if critical else None,
                        hazard_state=hazard["path"] if hazard else None)
                entry["rows"] = rows
                entry["metadata"] = metadata
                entry["summary"] = aggregate(rows)
            except Exception as exc:
                entry["error"] = repr(exc)
                entry["traceback"] = traceback.format_exc()
                entry["summary"] = aggregate(entry["rows"])
                result["failures"].append({"method": method.id, "error": repr(exc)})
                log("  FAILED: %r" % exc)
            entry["method_wall_seconds"] = time.perf_counter() - method_t0
            result["methods"].append(entry)
            result["elapsed_seconds"] = time.perf_counter() - started
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
