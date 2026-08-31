#!/usr/bin/env python3
"""Run a bounded, counterbalanced H6.5 causal paper matrix.

The experiment separates four questions at one matched memory budget:

* disk-only exact streaming (semantic reference);
* ordinary traffic-density placement;
* H6.5 tier placement with compressed RAM disabled; and
* full H6.5 joint representation/tier planning.

Three disjoint calibration prompts produce scheduler-aware all-disk traces.
Every live block evaluates all four held-out prompt families in a fresh process
per method.  Method position is balanced across blocks, cache and thermal gates
are retained per row, and the parent kills any cell that exceeds its hard time
limit.  The default four-block/one-token run is a regulated paper pilot, not a
claim of confirmatory statistical power.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback


REPO = pathlib.Path(__file__).resolve().parents[1]
WORKER = REPO / "scripts/run_h65_paper_worker.py"
sys.path.insert(0, str(REPO))

from afterimage.runtime.critical_path import TraceRecorder  # noqa: E402
from afterimage.runtime.h65_planner import (  # noqa: E402
    build_uniform_disk_plan,
    optimize_h65_plan,
)
from afterimage.runtime.replay_planner import manifest_fingerprint  # noqa: E402


CAMPAIGN_ID = "P1-H6.5-CAUSAL-MATRIX"
METHODS = (
    "disk-only",
    "traffic-placement",
    "h65-placement-only",
    "h65-full",
)
DEFAULT_CALIBRATION_CASES = (
    "summary-photosynthesis",
    "logic-glippets",
    "copy-nonce",
)
DEFAULT_EVALUATION_CASES = (
    "fact-gold",
    "arithmetic-17x6",
    "code-square",
    "retrieval-7319",
)
SOURCE_FILES = (
    pathlib.Path(__file__),
    WORKER,
    REPO / "scripts/run_bounded_suite.py",
    REPO / "afterimage/runtime/h65_planner.py",
    REPO / "afterimage/runtime/streaming_engine.py",
    REPO / "afterimage/runtime/representations.py",
)


def log(message: str) -> None:
    print(message, flush=True)


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command, cwd=REPO, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    return completed.stdout.strip()


def gpu_processes() -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def wait_for_gpu(minutes: float) -> None:
    deadline = time.monotonic() + minutes * 60.0
    while True:
        active = gpu_processes()
        if not active:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("GPU remained busy: %s" % "; ".join(active))
        log("GPU busy; waiting: %s" % "; ".join(active))
        time.sleep(15)


def cache_drop_preflight() -> tuple[bool, str | None]:
    try:
        subprocess.run(["sync"], check=True, timeout=60)
        pathlib.Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return True, None
    except Exception as exc:
        return False, repr(exc)


def terminate_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_cell(label: str, config: dict, *, root: pathlib.Path,
             timeout_minutes: float) -> dict:
    cells = root / "cells"
    config_path = cells / (label + ".config.json")
    result_path = cells / (label + ".result.json")
    log_path = cells / (label + ".log")
    checkpoint(config_path, config)
    log("START %s (hard timeout %.1f min)" % (label, timeout_minutes))
    started = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, "-u", str(WORKER), "--config", str(config_path),
             "--out", str(result_path)], cwd=REPO, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return_code = process.wait(timeout=timeout_minutes * 60.0)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_group(process)
            return_code = process.returncode
    elapsed = time.monotonic() - started
    if result_path.exists():
        cell = load(result_path)
    else:
        cell = {
            "rows": [], "summary": {},
            "error": (
                "worker exceeded %.1f-minute hard timeout" % timeout_minutes
                if timed_out else
                "worker produced no result (exit %s)" % return_code),
        }
    cell.update({
        "label": label,
        "worker_exit_code": return_code,
        "worker_timed_out": timed_out,
        "cell_wall_seconds": elapsed,
        "worker_log": str(log_path),
        "worker_result": str(result_path),
    })
    log("%s %s in %.1f min" % (
        "FAILED" if cell.get("error") else "OK", label, elapsed / 60.0))
    return cell


def parse_cases(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def common_overrides(args: argparse.Namespace) -> dict:
    return {
        "vram_budget_gb": args.vram_gb,
        "ram_budget_gb": args.ram_gb,
        "vram_cap_gb": args.vram_gb,
        "vram_safety_margin_gb": args.vram_safety_margin_gb,
        "decode_slice_elems": args.decode_slice_elems,
        "io_prefetch_depth": 2,
        "storage_read_policy": "per_blob",
        "progress": True,
    }


def worker_config(*, args: argparse.Namespace, method_id: str,
                  overrides: dict, block: int, split: str,
                  case_ids: tuple[str, ...], max_new_tokens: int) -> dict:
    return {
        "model": args.model,
        "store": str(pathlib.Path(args.store).resolve()),
        "block": block,
        "budget": {"vram_gb": args.vram_gb, "ram_gb": args.ram_gb},
        "method_id": method_id,
        "overrides": overrides,
        "case_split": split,
        "case_ids": list(case_ids),
        "max_new_tokens": max_new_tokens,
        "cooldown_seconds": args.cooldown_seconds,
        "cooldown_max_temp_c": args.cooldown_max_temp_c,
        "time_budget_minutes": args.cell_timeout_minutes,
    }


def balanced_order(seed: int, blocks: int) -> tuple[tuple[str, ...], ...]:
    base = list(METHODS)
    random.Random(seed).shuffle(base)
    orders = []
    for block in range(blocks):
        cycle = block // len(base)
        offset = block % len(base)
        current = base[offset:] + base[:offset]
        if cycle % 2:
            current = list(reversed(current))
        orders.append(tuple(current))
    return tuple(orders)


def archive_sources(root: pathlib.Path) -> dict[str, dict]:
    destination = root / "source_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    archived = {}
    for source in SOURCE_FILES:
        if not source.exists():
            raise FileNotFoundError("source file missing: %s" % source)
        relative = source.relative_to(REPO)
        target = destination / str(relative).replace("/", "__")
        shutil.copy2(source, target)
        archived[str(relative)] = {
            "source_sha256": sha256(source),
            "snapshot": str(target),
            "snapshot_sha256": sha256(target),
        }
    return archived


def plan_budget_ok(plan) -> bool:
    return bool(
        plan.vram_bytes + plan.vram_headroom_bytes <= plan.vram_budget_bytes
        and plan.ram_bytes <= plan.ram_budget_bytes)


def rows_by_case(cell: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in cell.get("rows", [])}


def cell_metric(cell: dict) -> float | None:
    seconds = [
        float(row["seconds_per_token"])
        for row in cell.get("rows", []) if row.get("seconds_per_token")
    ]
    return statistics.median(seconds) if seconds else None


def summarize_live(cells: list[dict], blocks: int) -> dict:
    indexed = {
        (int(cell["block"]), cell["method_id"]): cell
        for cell in cells if cell.get("case_split") == "evaluation"
    }
    method_summary = {}
    for method in METHODS:
        method_cells = [indexed.get((block, method)) for block in range(blocks)]
        metrics = [cell_metric(cell) for cell in method_cells if cell]
        metrics = [value for value in metrics if value is not None]
        rows = [
            row for cell in method_cells if cell
            for row in cell.get("rows", [])
        ]
        method_summary[method] = {
            "completed_cells": sum(bool(cell and not cell.get("error"))
                                   for cell in method_cells),
            "rows": len(rows),
            "block_median_seconds_per_token": metrics,
            "median_seconds_per_token": (
                statistics.median(metrics) if metrics else None),
            "median_whole_cell_peak_vram_gb": (
                statistics.median([
                    float(row["peak_vram_gb"]) for row in rows
                    if row.get("peak_vram_gb") is not None
                ]) if any(row.get("peak_vram_gb") is not None for row in rows)
                else None),
        }

    paired = {}
    traffic = method_summary["traffic-placement"]["block_median_seconds_per_token"]
    for method in ("disk-only", "h65-placement-only", "h65-full"):
        candidate = method_summary[method]["block_median_seconds_per_token"]
        reductions = [
            (control - treatment) / control
            for control, treatment in zip(traffic, candidate)
        ]
        paired["traffic-placement_vs_%s" % method] = {
            "control_seconds_per_token": traffic,
            "candidate_seconds_per_token": candidate,
            "paired_latency_reductions": reductions,
            "median_paired_latency_reduction": (
                statistics.median(reductions) if reductions else None),
            "conservative_paired_latency_reduction": (
                min(reductions) if reductions else None),
            "median_speedup": (
                statistics.median(traffic) / statistics.median(candidate)
                if traffic and candidate else None),
        }
    placement = method_summary[
        "h65-placement-only"]["block_median_seconds_per_token"]
    full = method_summary["h65-full"]["block_median_seconds_per_token"]
    reductions = [
        (control - treatment) / control
        for control, treatment in zip(placement, full)
    ]
    paired["h65-placement-only_vs_h65-full"] = {
        "control_seconds_per_token": placement,
        "candidate_seconds_per_token": full,
        "paired_latency_reductions": reductions,
        "median_paired_latency_reduction": (
            statistics.median(reductions) if reductions else None),
        "conservative_paired_latency_reduction": (
            min(reductions) if reductions else None),
        "median_speedup": (
            statistics.median(placement) / statistics.median(full)
            if placement and full else None),
    }
    return {"methods": method_summary, "paired_effects": paired}


def exactness_failures(cells: list[dict], blocks: int,
                       evaluation_cases: tuple[str, ...]) -> list[dict]:
    indexed = {
        (int(cell["block"]), cell["method_id"]): cell
        for cell in cells if cell.get("case_split") == "evaluation"
    }
    failures = []
    for block in range(blocks):
        reference = indexed.get((block, "disk-only"))
        reference_rows = rows_by_case(reference) if reference else {}
        for method in METHODS:
            cell = indexed.get((block, method))
            actual = rows_by_case(cell) if cell else {}
            for case_id in evaluation_cases:
                if case_id not in reference_rows:
                    failures.append({
                        "block": block, "method": method, "case_id": case_id,
                        "reason": "missing disk reference",
                    })
                elif case_id not in actual:
                    failures.append({
                        "block": block, "method": method, "case_id": case_id,
                        "reason": "missing treatment row",
                    })
                elif (actual[case_id].get("output_token_ids") !=
                      reference_rows[case_id].get("output_token_ids")):
                    failures.append({
                        "block": block, "method": method, "case_id": case_id,
                        "reason": "output token mismatch",
                        "reference": reference_rows[case_id].get("output_token_ids"),
                        "actual": actual[case_id].get("output_token_ids"),
                    })
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--store", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--h2d", required=True)
    parser.add_argument("--vram-gb", type=float, default=8.0)
    parser.add_argument("--ram-gb", type=float, default=16.0)
    parser.add_argument("--decode-slice-elems", type=int, default=1 << 22)
    parser.add_argument("--vram-safety-margin-gb", type=float, default=0.5)
    parser.add_argument("--search-iterations", type=int, default=256)
    parser.add_argument("--minimum-live-improvement", type=float, default=0.05)
    parser.add_argument("--calibration-cases", default=",".join(DEFAULT_CALIBRATION_CASES))
    parser.add_argument("--evaluation-cases", default=",".join(DEFAULT_EVALUATION_CASES))
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--cooldown-max-temp-c", type=float, default=75.0)
    parser.add_argument("--cell-timeout-minutes", type=float, default=20.0)
    parser.add_argument("--wait-for-gpu-minutes", type=float, default=5.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    calibration_cases = parse_cases(args.calibration_cases)
    evaluation_cases = parse_cases(args.evaluation_cases)
    if len(calibration_cases) < 3 or len(set(calibration_cases)) != len(calibration_cases):
        parser.error("at least three unique calibration cases are required")
    if len(evaluation_cases) < 4 or len(set(evaluation_cases)) != len(evaluation_cases):
        parser.error("at least four unique evaluation cases are required")
    if set(calibration_cases) & set(evaluation_cases):
        parser.error("calibration and evaluation cases must be disjoint")
    if args.blocks < len(METHODS) or args.blocks % len(METHODS):
        parser.error("--blocks must be a positive multiple of %d" % len(METHODS))
    if (args.max_new_tokens < 1 or args.vram_gb <= 0 or args.ram_gb < 0
            or args.decode_slice_elems < 1 or args.search_iterations < 0
            or args.cell_timeout_minutes <= 0):
        parser.error("token count, budgets, search, and timeout are invalid")

    store = pathlib.Path(args.store).resolve()
    manifest_path = (pathlib.Path(args.manifest).resolve() if args.manifest
                     else store / "manifest.json")
    h2d_path = pathlib.Path(args.h2d).resolve()
    out = pathlib.Path(args.out).resolve()
    partial = out.with_suffix(out.suffix + ".partial")
    root = out.parent / (out.stem + "-artifacts")
    for required in (store, manifest_path, h2d_path, WORKER, *SOURCE_FILES):
        if not required.exists():
            parser.error("required path does not exist: %s" % required)
    if out.exists():
        if args.resume:
            log("matrix is already complete: %s" % out)
            return 0
        raise FileExistsError("refusing to overwrite completed matrix: %s" % out)
    if args.resume:
        if not partial.exists() or not root.exists():
            raise FileNotFoundError("--resume requires partial result and artifact directory")
    elif partial.exists() or root.exists():
        raise FileExistsError("partial matrix exists; pass --resume: %s" % partial)

    manifest = load(manifest_path)
    h2d = load(h2d_path)
    h2d_gbps = float(h2d["median_stable_gbps"])
    session_started = time.time()
    orders = balanced_order(args.seed, args.blocks)
    settings = {
        "model": args.model,
        "store": str(store),
        "manifest": str(manifest_path),
        "h2d_artifact": str(h2d_path),
        "vram_budget_gb": args.vram_gb,
        "ram_budget_gb": args.ram_gb,
        "decode_slice_elems": args.decode_slice_elems,
        "vram_safety_margin_gb": args.vram_safety_margin_gb,
        "search_iterations": args.search_iterations,
        "minimum_live_improvement": args.minimum_live_improvement,
        "calibration_case_ids": list(calibration_cases),
        "evaluation_case_ids": list(evaluation_cases),
        "max_new_tokens": args.max_new_tokens,
        "blocks": args.blocks,
        "seed": args.seed,
        "cell_timeout_minutes": args.cell_timeout_minutes,
        "method_order_per_block": [list(order) for order in orders],
    }
    if args.resume:
        result = load(partial)
        mismatch = [
            (key, result.get(key), value) for key, value in settings.items()
            if result.get(key) != value
        ]
        if mismatch:
            raise ValueError("resume settings differ: %s" % mismatch)
        for relative, entry in result["source_snapshot"].items():
            current = REPO / relative
            if sha256(current) != entry["source_sha256"]:
                raise RuntimeError("source changed since partial run: %s" % relative)
        result.pop("error", None)
        result.pop("traceback", None)
        result["resume_count"] = int(result.get("resume_count", 0)) + 1
    else:
        root.mkdir(parents=True)
        source_snapshot = archive_sources(root)
        result = {
            "schema_version": 1,
            "kind": "h65_causal_paper_matrix",
            "campaign_id": CAMPAIGN_ID,
            "status": "initializing",
            "evidence_level": "L2_regulated_exploratory",
            "exploratory": True,
            "confirmatory_protocol_satisfied": False,
            "paper_claim_scope": (
                "bounded TTFT/short-decode causal pilot; use for effect-size and "
                "power planning, not a standalone confirmatory claim"),
            **settings,
            "manifest_sha256": manifest_fingerprint(manifest),
            "h2d_gbps": h2d_gbps,
            "h2d_memory_mode": h2d.get("memory_mode"),
            "started_at_unix": session_started,
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "status", "--short"]),
            "source_snapshot": source_snapshot,
            "cache_regime": "cold page cache before every timed prompt",
            "cell_isolation": "fresh process per block-method cell",
            "cells": [],
        }
    checkpoint(partial, result)

    try:
        wait_for_gpu(args.wait_for_gpu_minutes)
        cache_ok, cache_error = cache_drop_preflight()
        result["cache_drop_preflight"] = {
            "succeeded": cache_ok, "error": cache_error,
        }
        checkpoint(partial, result)
        if not cache_ok:
            raise RuntimeError(
                "cold-cache experiment requires /proc/sys/vm/drop_caches: %s" %
                cache_error)

        common = common_overrides(args)
        disk_plan = build_uniform_disk_plan(
            manifest, vram_budget_gb=args.vram_gb,
            ram_budget_gb=args.ram_gb,
            decode_slice_elems=args.decode_slice_elems,
            vram_safety_margin_gb=args.vram_safety_margin_gb)
        disk_plan_path = root / "disk-plan.json"
        disk_plan.save(disk_plan_path)

        result["status"] = "calibrating"
        checkpoint(partial, result)
        trace_paths = []
        for block, case_id in enumerate(calibration_cases):
            label = "calibration-disk-b%d-%s" % (block, case_id)
            trace_path = root / ("calibration-trace-b%d-%s.json" % (block, case_id))
            trace_paths.append(trace_path)
            completed = next((
                cell for cell in result["cells"]
                if cell.get("label") == label and not cell.get("error")
            ), None)
            if completed and trace_path.exists():
                log("SKIP %s (resumed)" % label)
                continue
            overrides = {
                **common,
                "representation_policy": "per_tensor",
                "representation_plan_state": str(disk_plan_path),
                "trace_events": True,
                "trace_output": str(trace_path),
            }
            cell = run_cell(
                label, worker_config(
                    args=args, method_id="h65-calibration-disk",
                    overrides=overrides, block=block, split="calibration",
                    case_ids=(case_id,), max_new_tokens=1),
                root=root, timeout_minutes=args.cell_timeout_minutes)
            result["cells"] = [
                existing for existing in result["cells"]
                if existing.get("label") != label
            ] + [cell]
            checkpoint(partial, result)
            if cell.get("error") or not trace_path.exists():
                raise RuntimeError("calibration failed: %s" % label)

        traces = [TraceRecorder.load(path) for path in trace_paths]
        result["status"] = "planning"
        checkpoint(partial, result)
        planned = {}
        for method, compressed in (
                ("h65-placement-only", False), ("h65-full", True)):
            log("PLANNING %s from %d disjoint traces" % (method, len(traces)))
            planning = optimize_h65_plan(
                manifest, traces,
                vram_budget_gb=args.vram_gb, ram_budget_gb=args.ram_gb,
                h2d_gbps=h2d_gbps,
                decode_slice_elems=args.decode_slice_elems,
                search_iterations=args.search_iterations, seed=args.seed,
                minimum_predicted_improvement=0.0,
                minimum_live_improvement=args.minimum_live_improvement,
                vram_safety_margin_gb=args.vram_safety_margin_gb,
                h2d_memory_mode=str(h2d.get("memory_mode") or "unknown"),
                enable_compressed_ram=compressed)
            candidate_path = root / (method + "-candidate.json")
            deployment_path = root / (method + "-deployment.json")
            planning.candidate_plan.save(candidate_path)
            planning.plan.save(deployment_path)
            planned[method] = {
                "planning": planning,
                "candidate_path": candidate_path,
                "deployment_path": deployment_path,
            }
            result.setdefault("planner", {})[method] = {
                **dataclasses.asdict(planning.report),
                "compressed_ram_enabled": compressed,
                "candidate_plan": str(candidate_path),
                "candidate_plan_sha256": sha256(candidate_path),
                "guarded_deployment_plan": str(deployment_path),
                "guarded_deployment_plan_sha256": sha256(deployment_path),
                "raw_traces": [str(path) for path in trace_paths],
                "raw_trace_sha256": [sha256(path) for path in trace_paths],
            }
            checkpoint(partial, result)

        method_overrides = {
            "disk-only": {
                **common,
                "representation_policy": "per_tensor",
                "representation_plan_state": str(disk_plan_path),
            },
            "traffic-placement": {
                **common,
                "placement_policy": "traffic_density",
                "ram_tier_format": "decoded",
            },
            "h65-placement-only": {
                **common,
                "representation_policy": "multi_state",
                "representation_plan_state": str(
                    planned["h65-placement-only"]["candidate_path"]),
            },
            "h65-full": {
                **common,
                "representation_policy": "multi_state",
                "representation_plan_state": str(planned["h65-full"]["candidate_path"]),
            },
        }

        result["status"] = "evaluating"
        checkpoint(partial, result)
        for block, order in enumerate(orders):
            for method in order:
                label = "%s-b%d" % (method, block)
                completed = next((
                    cell for cell in result["cells"]
                    if cell.get("label") == label and not cell.get("error")
                ), None)
                if completed:
                    log("SKIP %s (resumed)" % label)
                    continue
                wait_for_gpu(args.wait_for_gpu_minutes)
                cell = run_cell(
                    label, worker_config(
                        args=args, method_id=method,
                        overrides=method_overrides[method], block=block,
                        split="evaluation", case_ids=evaluation_cases,
                        max_new_tokens=args.max_new_tokens),
                    root=root, timeout_minutes=args.cell_timeout_minutes)
                result["cells"] = [
                    existing for existing in result["cells"]
                    if existing.get("label") != label
                ] + [cell]
                checkpoint(partial, result)

        evaluation_cells = [
            cell for cell in result["cells"]
            if cell.get("case_split") == "evaluation"
        ]
        live = summarize_live(evaluation_cells, args.blocks)
        result["live_comparison"] = live

        finalized_reports = {}
        traffic_seconds = tuple(
            live["methods"]["traffic-placement"][
                "block_median_seconds_per_token"])
        for method, compressed in (
                ("h65-placement-only", False), ("h65-full", True)):
            candidate_seconds = tuple(
                live["methods"][method]["block_median_seconds_per_token"])
            eligible = bool(
                len(traffic_seconds) == len(candidate_seconds) == args.blocks)
            final = optimize_h65_plan(
                manifest, traces,
                vram_budget_gb=args.vram_gb, ram_budget_gb=args.ram_gb,
                h2d_gbps=h2d_gbps,
                decode_slice_elems=args.decode_slice_elems,
                search_iterations=args.search_iterations, seed=args.seed,
                minimum_predicted_improvement=0.0,
                minimum_live_improvement=args.minimum_live_improvement,
                vram_safety_margin_gb=args.vram_safety_margin_gb,
                h2d_memory_mode=str(h2d.get("memory_mode") or "unknown"),
                enable_compressed_ram=compressed,
                live_control_seconds=traffic_seconds,
                live_candidate_seconds=candidate_seconds,
                live_validation_eligible=eligible,
                minimum_live_blocks=args.blocks)
            final.plan.save(planned[method]["deployment_path"])
            finalized_reports[method] = final
            result["planner"][method] = {
                **dataclasses.asdict(final.report),
                "compressed_ram_enabled": compressed,
                "candidate_plan": str(planned[method]["candidate_path"]),
                "candidate_plan_sha256": sha256(planned[method]["candidate_path"]),
                "guarded_deployment_plan": str(planned[method]["deployment_path"]),
                "guarded_deployment_plan_sha256": sha256(
                    planned[method]["deployment_path"]),
                "raw_traces": [str(path) for path in trace_paths],
                "raw_trace_sha256": [sha256(path) for path in trace_paths],
            }

        exact_failures = exactness_failures(
            evaluation_cells, args.blocks, evaluation_cases)
        expected_cells = len(calibration_cases) + args.blocks * len(METHODS)
        expected_rows = args.blocks * len(METHODS) * len(evaluation_cases)
        rows = [row for cell in evaluation_cells for row in cell.get("rows", [])]
        cells_clean = bool(
            len(result["cells"]) == expected_cells
            and all(not cell.get("error") for cell in result["cells"]))
        cache_clean = bool(rows and all(
            row.get("cache_drop_succeeded") for row in rows))
        thermal_clean = bool(rows and all(
            row.get("cooldown_reached_target")
            and not row.get("throttled_after_cooldown")
            and not (row.get("gpu_thermal") or {}).get("throttled")
            for row in rows))
        positional_balance = all(
            sum(order[position] == method for order in orders)
            == args.blocks // len(METHODS)
            for method in METHODS for position in range(len(METHODS)))
        placement_plan = finalized_reports["h65-placement-only"].candidate_plan
        full_plan = finalized_reports["h65-full"].candidate_plan
        gates = {
            "all_cells_completed": cells_clean,
            "expected_evaluation_rows": len(rows) == expected_rows,
            "raw_scheduler_traces_retained": all(path.exists() for path in trace_paths),
            "calibration_evaluation_disjoint": not bool(
                set(calibration_cases) & set(evaluation_cases)),
            "balanced_method_position": positional_balance,
            "exact_output_tokens": not exact_failures,
            "requested_token_count_completed": bool(rows and all(
                row.get("output_tokens") == args.max_new_tokens for row in rows)),
            "cold_cache_confirmed": cache_clean,
            "thermal_gate_passed": thermal_clean,
            "whole_cell_vram_measured": bool(rows and all(
                row.get("peak_vram_gb") is not None for row in rows)),
            "placement_candidate_within_budget": plan_budget_ok(placement_plan),
            "full_candidate_within_budget": plan_budget_ok(full_plan),
            "placement_candidate_diverged_from_traffic": bool(
                finalized_reports["h65-placement-only"].report.treatment_diverged),
            "full_candidate_diverged_from_traffic": bool(
                finalized_reports["h65-full"].report.treatment_diverged),
            "source_snapshot_retained": all(
                pathlib.Path(entry["snapshot"]).exists()
                and entry["source_sha256"] == entry["snapshot_sha256"]
                for entry in result["source_snapshot"].values()),
        }
        gates["paper_pilot_eligible"] = all(gates.values())
        result["exactness_failures"] = exact_failures
        result["gates"] = gates
        result["status"] = "complete"
        result["completed_at_unix"] = time.time()
        result["elapsed_seconds"] = result.get("elapsed_seconds", 0.0) + (
            time.time() - session_started)
        checkpoint(partial, result)
        partial.replace(out)
        log("DONE eligible=%s result=%s" % (gates["paper_pilot_eligible"], out))
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        result["completed_at_unix"] = time.time()
        result["elapsed_seconds"] = result.get("elapsed_seconds", 0.0) + (
            time.time() - session_started)
        checkpoint(partial, result)
        log("FAILED: %s" % exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
