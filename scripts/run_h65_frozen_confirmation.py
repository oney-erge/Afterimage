#!/usr/bin/env python3
"""Evaluate frozen pilot H6.5 plans on independent Llama blocks.

This is the correction to the first confirmation attempt: calibration and
planning are not repeated. The pilot's candidate plans are immutable inputs;
only fresh-process held-out evaluation cells are collected. Thus a fallback on
new calibration traces cannot silently turn the confirmation into a different
algorithm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import time
import traceback


REPO = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))

from scripts.run_h65_paper_matrix import (  # noqa: E402
    DEFAULT_EVALUATION_CASES,
    METHODS,
    balanced_order,
    cache_drop_preflight,
    cell_metric,
    common_overrides,
    exactness_failures,
    gpu_processes,
    load,
    run_cell,
    sha256,
    summarize_live,
    terminate_group,
    wait_for_gpu,
    worker_config,
    checkpoint,
    log,
)


CAMPAIGN_ID = "P1-H6.5-FROZEN-PLAN-CONFIRMATION"
WORKER = REPO / "scripts/run_h65_paper_worker.py"


def parse_cases(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def archive_sources(root: pathlib.Path, protocol: pathlib.Path) -> dict[str, dict]:
    destination = root / "source_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    sources = [
        pathlib.Path(__file__),
        REPO / "scripts/run_h65_paper_matrix.py",
        WORKER,
        REPO / "scripts/run_bounded_suite.py",
        REPO / "afterimage/runtime/h65_planner.py",
        REPO / "afterimage/runtime/streaming_engine.py",
        REPO / "afterimage/runtime/representations.py",
        protocol,
    ]
    snapshots = {}
    for source in sources:
        if not source.exists():
            raise FileNotFoundError("source file missing: %s" % source)
        relative = (
            str(source.relative_to(REPO))
            if source.is_relative_to(REPO) else source.name)
        target = destination / relative.replace("/", "__")
        shutil.copy2(source, target)
        snapshots[relative] = {
            "source_sha256": sha256(source),
            "snapshot": str(target),
            "snapshot_sha256": sha256(target),
        }
    return snapshots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--store", required=True)
    parser.add_argument("--h2d", required=True)
    parser.add_argument("--pilot-result", required=True)
    parser.add_argument("--confirmatory-protocol", required=True)
    parser.add_argument("--vram-gb", type=float, default=8.0)
    parser.add_argument("--ram-gb", type=float, default=16.0)
    parser.add_argument("--decode-slice-elems", type=int, default=1 << 22)
    parser.add_argument("--vram-safety-margin-gb", type=float, default=0.5)
    parser.add_argument("--evaluation-cases", default=",".join(DEFAULT_EVALUATION_CASES))
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--cooldown-max-temp-c", type=float, default=75.0)
    parser.add_argument("--cell-timeout-minutes", type=float, default=20.0)
    parser.add_argument("--wait-for-gpu-minutes", type=float, default=5.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    evaluation_cases = parse_cases(args.evaluation_cases)
    if len(evaluation_cases) < 4 or len(set(evaluation_cases)) != len(evaluation_cases):
        parser.error("at least four unique evaluation cases are required")
    if args.blocks < 8 or args.blocks % len(METHODS):
        parser.error("frozen confirmation requires at least eight balanced blocks")
    if args.max_new_tokens < 1 or args.vram_gb <= 0 or args.ram_gb < 0:
        parser.error("token count and budgets are invalid")

    store = pathlib.Path(args.store).resolve()
    h2d_path = pathlib.Path(args.h2d).resolve()
    pilot_path = pathlib.Path(args.pilot_result).resolve()
    protocol_path = pathlib.Path(args.confirmatory_protocol).resolve()
    out = pathlib.Path(args.out).resolve()
    partial = out.with_suffix(out.suffix + ".partial")
    root = out.parent / (out.stem + "-artifacts")
    for required in (store, h2d_path, pilot_path, protocol_path, WORKER):
        if not required.exists():
            parser.error("required path does not exist: %s" % required)
    if out.exists() or partial.exists() or root.exists():
        raise FileExistsError("refusing to overwrite existing frozen confirmation")

    pilot = load(pilot_path)
    if not (pilot.get("gates") or {}).get("paper_pilot_eligible"):
        raise RuntimeError("pilot result did not pass paper_pilot_eligible")
    pilot_artifacts = pilot_path.parent / (pilot_path.stem + "-artifacts")
    disk_plan = pilot_artifacts / "disk-plan.json"
    if not disk_plan.exists():
        raise FileNotFoundError("pilot disk plan missing: %s" % disk_plan)
    frozen = {}
    for method in ("h65-placement-only", "h65-full"):
        planner = (pilot.get("planner") or {}).get(method) or {}
        candidate = pathlib.Path(planner["candidate_plan"]).resolve()
        if not candidate.exists() or not planner.get("treatment_diverged"):
            raise RuntimeError(
                "pilot has no divergent frozen candidate for %s" % method)
        frozen[method] = {
            "path": candidate,
            "sha256": sha256(candidate),
            "pilot_report_sha256": planner.get("candidate_plan_sha256"),
        }

    h2d = load(h2d_path)
    orders = balanced_order(args.seed, args.blocks)
    session_started = time.time()
    root.mkdir(parents=True)
    source_snapshot = archive_sources(root, protocol_path)
    settings = {
        "model": args.model,
        "store": str(store),
        "h2d_artifact": str(h2d_path),
        "pilot_result": str(pilot_path),
        "pilot_result_sha256": sha256(pilot_path),
        "confirmatory_protocol": str(protocol_path),
        "confirmatory_protocol_sha256": sha256(protocol_path),
        "frozen_candidate_plan_sha256": {
            method: entry["sha256"] for method, entry in frozen.items()},
        "vram_budget_gb": args.vram_gb,
        "ram_budget_gb": args.ram_gb,
        "decode_slice_elems": args.decode_slice_elems,
        "vram_safety_margin_gb": args.vram_safety_margin_gb,
        "evaluation_case_ids": list(evaluation_cases),
        "max_new_tokens": args.max_new_tokens,
        "blocks": args.blocks,
        "seed": args.seed,
        "method_order_per_block": [list(order) for order in orders],
        "cell_timeout_minutes": args.cell_timeout_minutes,
    }
    result = {
        "schema_version": 1,
        "kind": "h65_frozen_plan_confirmatory_matrix",
        "campaign_id": CAMPAIGN_ID,
        "status": "evaluating",
        "evidence_level": "L3_frozen_confirmatory",
        "exploratory": False,
        "confirmatory_protocol_satisfied": False,
        "paper_claim_scope": "frozen-plan independent Llama causal confirmation",
        "started_at_unix": session_started,
        "git_commit": "unknown",
        "git_status": "unknown",
        "source_snapshot": source_snapshot,
        "cache_regime": "cold page cache before every timed prompt",
        "cell_isolation": "fresh process per block-method cell",
        "frozen_source": {
            "pilot_result": str(pilot_path),
            "pilot_result_sha256": sha256(pilot_path),
            "disk_plan": str(disk_plan),
            "disk_plan_sha256": sha256(disk_plan),
            "candidate_plans": {
                method: str(entry["path"]) for method, entry in frozen.items()},
        },
        **settings,
        "h2d_gbps": float(h2d["median_stable_gbps"]),
        "h2d_memory_mode": h2d.get("memory_mode"),
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
            raise RuntimeError("cold-cache preflight failed: %s" % cache_error)

        common = common_overrides(args)
        disk_plan = disk_plan.resolve()
        overrides = {
            "disk-only": {
                **common,
                "representation_policy": "per_tensor",
                "representation_plan_state": str(disk_plan),
            },
            "traffic-placement": {
                **common,
                "placement_policy": "traffic_density",
                "ram_tier_format": "decoded",
            },
            "h65-placement-only": {
                **common,
                "representation_policy": "multi_state",
                "representation_plan_state": str(frozen[
                    "h65-placement-only"]["path"]),
            },
            "h65-full": {
                **common,
                "representation_policy": "multi_state",
                "representation_plan_state": str(frozen["h65-full"]["path"]),
            },
        }
        for block, order in enumerate(orders):
            for method in order:
                label = "%s-b%d" % (method, block)
                cell = run_cell(
                    label, worker_config(
                        args=args, method_id=method, overrides=overrides[method],
                        block=block, split="evaluation", case_ids=evaluation_cases,
                        max_new_tokens=args.max_new_tokens),
                    root=root, timeout_minutes=args.cell_timeout_minutes)
                result["cells"].append(cell)
                checkpoint(partial, result)

        live_cells = [cell for cell in result["cells"] if not cell.get("error")]
        result["live_comparison"] = summarize_live(live_cells, args.blocks)
        failures = exactness_failures(live_cells, args.blocks, evaluation_cases)
        rows = [row for cell in live_cells for row in cell.get("rows", [])]
        expected_rows = args.blocks * len(METHODS) * len(evaluation_cases)
        all_cells = bool(
            len(result["cells"]) == args.blocks * len(METHODS)
            and all(not cell.get("error") for cell in result["cells"]))
        cache_clean = bool(rows and all(row.get("cache_drop_succeeded") for row in rows))
        thermal_clean = bool(rows and all(
            row.get("cooldown_reached_target")
            and not row.get("throttled_after_cooldown")
            and not (row.get("gpu_thermal") or {}).get("throttled")
            for row in rows))
        gates = {
            "all_cells_completed": all_cells,
            "expected_evaluation_rows": len(rows) == expected_rows,
            "balanced_method_position": all(
                sum(order[position] == method for order in orders)
                == args.blocks // len(METHODS)
                for method in METHODS for position in range(len(METHODS))),
            "exact_output_tokens": not failures,
            "requested_token_count_completed": bool(rows and all(
                row.get("output_tokens") == args.max_new_tokens for row in rows)),
            "cold_cache_confirmed": cache_clean,
            "thermal_gate_passed": thermal_clean,
            "whole_cell_vram_measured": bool(rows and all(
                row.get("peak_vram_gb") is not None for row in rows)),
            "frozen_pilot_plans_unchanged": all(
                entry["sha256"] == entry["pilot_report_sha256"]
                for entry in frozen.values()),
            "frozen_candidates_diverged_from_traffic": all(
                (pilot.get("planner") or {}).get(method, {}).get(
                    "treatment_diverged")
                for method in ("h65-placement-only", "h65-full")),
            "source_snapshot_retained": all(
                pathlib.Path(entry["snapshot"]).exists()
                and entry["source_sha256"] == entry["snapshot_sha256"]
                for entry in source_snapshot.values()),
        }
        gates["paper_pilot_eligible"] = all(gates.values())
        gates["confirmatory_execution_eligible"] = bool(
            gates["paper_pilot_eligible"])
        result["exactness_failures"] = failures
        result["gates"] = gates
        result["confirmatory_protocol_satisfied"] = gates[
            "confirmatory_execution_eligible"]
        result["status"] = "complete"
        result["completed_at_unix"] = time.time()
        result["elapsed_seconds"] = time.time() - session_started
        checkpoint(partial, result)
        partial.replace(out)
        log("DONE confirmatory=%s result=%s" % (
            result["confirmatory_protocol_satisfied"], out))
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        result["completed_at_unix"] = time.time()
        result["elapsed_seconds"] = time.time() - session_started
        checkpoint(partial, result)
        log("FAILED: %s" % exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
