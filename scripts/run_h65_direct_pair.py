#!/usr/bin/env python3
"""Run a pre-registered adjacent-pair H6.5 placement confirmation.

The runner evaluates traffic-density placement and one immutable H6.5
placement-only plan in alternating AB/BA pairs. No calibration or plan search is
performed here: the pilot artifact and candidate hash are treated as inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback


REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_h65_paper_matrix import (  # noqa: E402
    DEFAULT_EVALUATION_CASES,
    cache_drop_preflight,
    cell_metric,
    checkpoint,
    common_overrides,
    load,
    log,
    run_cell,
    sha256,
    wait_for_gpu,
    worker_config,
)
from afterimage.runtime.replay_planner import manifest_fingerprint  # noqa: E402


CAMPAIGN_ID = "P1-H6.5-DIRECT-PAIR-CONFIRMATION"
METHODS = ("traffic-placement", "h65-placement-only")
WORKER = REPO / "scripts/run_h65_paper_worker.py"
T_CRITICAL_95_DF11 = 2.200985160082949
EXPECTED_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
EXPECTED_CANDIDATE_SHA256 = (
    "28bcc25502025e0720912a6a1393eb9d9ff52fd51121a54c95d64e9f6d523838")


def parse_cases(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command, cwd=REPO, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    return completed.stdout.strip()


def git_output(*arguments: str) -> str:
    return command_output([
        "git", "-c", "safe.directory=%s" % REPO, *arguments])


def pair_orders(seed: int, blocks: int) -> tuple[tuple[str, str], ...]:
    base = list(METHODS)
    if random.Random(seed).randrange(2):
        base.reverse()
    return tuple(
        tuple(base if block % 2 == 0 else reversed(base))
        for block in range(blocks)
    )


def archive_sources(root: pathlib.Path, protocol: pathlib.Path) -> dict[str, dict]:
    destination = root / "source_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    sources = (
        pathlib.Path(__file__),
        REPO / "scripts/run_h65_paper_matrix.py",
        WORKER,
        REPO / "scripts/run_bounded_suite.py",
        REPO / "afterimage/runtime/h65_planner.py",
        REPO / "afterimage/runtime/streaming_engine.py",
        REPO / "afterimage/runtime/representations.py",
        protocol,
    )
    snapshots = {}
    for source in sources:
        if not source.exists():
            raise FileNotFoundError("source file missing: %s" % source)
        relative = (
            str(source.relative_to(REPO))
            if source.is_relative_to(REPO) else "confirmatory_protocol")
        target = destination / relative.replace("/", "__")
        shutil.copy2(source, target)
        snapshots[relative] = {
            "source_sha256": sha256(source),
            "snapshot": str(target),
            "snapshot_sha256": sha256(target),
        }
    return snapshots


def rows_by_case(cell: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in cell.get("rows", [])}


def exactness_failures(cells: list[dict], blocks: int,
                       evaluation_cases: tuple[str, ...]) -> list[dict]:
    indexed = {
        (int(cell["block"]), cell["method_id"]): cell for cell in cells
        if cell.get("case_split") == "evaluation" and not cell.get("error")
    }
    failures = []
    for block in range(blocks):
        traffic = rows_by_case(indexed.get((block, METHODS[0]), {}))
        h65 = rows_by_case(indexed.get((block, METHODS[1]), {}))
        for case_id in evaluation_cases:
            if case_id not in traffic or case_id not in h65:
                failures.append({
                    "block": block, "case_id": case_id,
                    "reason": "missing paired row",
                })
            elif (traffic[case_id].get("output_token_ids") !=
                  h65[case_id].get("output_token_ids")):
                failures.append({
                    "block": block, "case_id": case_id,
                    "reason": "paired output token mismatch",
                    "traffic": traffic[case_id].get("output_token_ids"),
                    "h65": h65[case_id].get("output_token_ids"),
                })
    return failures


def summarize(cells: list[dict], blocks: int) -> dict:
    indexed = {
        (int(cell["block"]), cell["method_id"]): cell for cell in cells
        if cell.get("case_split") == "evaluation" and not cell.get("error")
    }
    method_summary = {}
    block_metrics: dict[str, list[float]] = {}
    for method in METHODS:
        method_cells = [indexed.get((block, method)) for block in range(blocks)]
        metrics = [cell_metric(cell) for cell in method_cells if cell]
        metrics = [float(value) for value in metrics if value is not None]
        block_metrics[method] = metrics
        rows = [
            row for cell in method_cells if cell for row in cell.get("rows", [])]
        vram = [
            float(row["peak_vram_gb"]) for row in rows
            if row.get("peak_vram_gb") is not None]
        method_summary[method] = {
            "completed_cells": sum(
                bool(cell and not cell.get("error")) for cell in method_cells),
            "rows": len(rows),
            "block_median_seconds_per_token": metrics,
            "median_seconds_per_token": (
                statistics.median(metrics) if metrics else None),
            "median_whole_cell_peak_vram_gb": (
                statistics.median(vram) if vram else None),
        }

    traffic = block_metrics[METHODS[0]]
    h65 = block_metrics[METHODS[1]]
    paired = []
    if len(traffic) == blocks and len(h65) == blocks:
        paired = [
            {
                "block": block,
                "traffic_seconds_per_token": control,
                "h65_seconds_per_token": treatment,
                "latency_reduction": (control - treatment) / control,
                "log_speedup": math.log(control / treatment),
            }
            for block, (control, treatment) in enumerate(zip(traffic, h65))
        ]
    log_speedups = [entry["log_speedup"] for entry in paired]
    mean_log = statistics.mean(log_speedups) if log_speedups else None
    ci = None
    if len(log_speedups) == 12:
        standard_error = statistics.stdev(log_speedups) / math.sqrt(12)
        half_width = T_CRITICAL_95_DF11 * standard_error
        ci = [math.exp(mean_log - half_width), math.exp(mean_log + half_width)]
    return {
        "methods": method_summary,
        "paired_effect": {
            "block_pairs": paired,
            "mean_log_speedup": mean_log,
            "geometric_speedup": math.exp(mean_log) if mean_log is not None else None,
            "geometric_speedup_95pct_paired_t_interval": ci,
            "median_paired_latency_reduction": (
                statistics.median([entry["latency_reduction"] for entry in paired])
                if paired else None),
            "student_t_critical": (
                T_CRITICAL_95_DF11 if len(log_speedups) == 12 else None),
            "degrees_of_freedom": len(log_speedups) - 1,
        },
    }


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
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--cooldown-max-temp-c", type=float, default=75.0)
    parser.add_argument("--cell-timeout-minutes", type=float, default=20.0)
    parser.add_argument("--wait-for-gpu-minutes", type=float, default=5.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    evaluation_cases = parse_cases(args.evaluation_cases)
    if evaluation_cases != DEFAULT_EVALUATION_CASES:
        parser.error("the frozen protocol requires the four fixed evaluation cases")
    if args.model != EXPECTED_MODEL:
        parser.error("the frozen protocol requires %s" % EXPECTED_MODEL)
    if args.blocks != 12:
        parser.error("the frozen protocol requires exactly 12 blocks")
    if args.seed != 20260902:
        parser.error("the frozen protocol requires seed 20260902")
    if args.max_new_tokens != 1:
        parser.error("the frozen protocol requires exactly one generated token")
    if (args.vram_gb != 8.0 or args.ram_gb != 16.0
            or args.vram_safety_margin_gb != 0.5):
        parser.error("the frozen protocol requires 8/16 GiB and 0.5 GiB margin")
    if args.cell_timeout_minutes != 20.0:
        parser.error("the frozen protocol requires a 20-minute cell timeout")

    store = pathlib.Path(args.store).resolve()
    h2d_path = pathlib.Path(args.h2d).resolve()
    pilot_path = pathlib.Path(args.pilot_result).resolve()
    protocol_path = pathlib.Path(args.confirmatory_protocol).resolve()
    out = pathlib.Path(args.out).resolve()
    partial = out.with_suffix(out.suffix + ".partial")
    root = out.parent / (out.stem + "-artifacts")
    manifest_path = store / "manifest.json"
    for required in (
            store, manifest_path, h2d_path, pilot_path, protocol_path, WORKER):
        if not required.exists():
            parser.error("required path does not exist: %s" % required)
    if out.exists():
        if args.resume:
            log("direct-pair campaign is already complete: %s" % out)
            return 0
        raise FileExistsError("refusing to overwrite completed result: %s" % out)
    if args.resume:
        if not partial.exists() or not root.exists():
            raise FileNotFoundError(
                "--resume requires partial result and artifact directory")
    elif partial.exists() or root.exists():
        raise FileExistsError("partial result exists; pass --resume: %s" % partial)

    pilot = load(pilot_path)
    planner = (pilot.get("planner") or {}).get("h65-placement-only") or {}
    if not (pilot.get("gates") or {}).get("paper_pilot_eligible"):
        raise RuntimeError("pilot did not pass paper_pilot_eligible")
    if pathlib.Path(pilot.get("store", "")).resolve() != store:
        raise RuntimeError("store differs from the frozen pilot")
    manifest_sha = manifest_fingerprint(load(manifest_path))
    if manifest_sha != pilot.get("manifest_sha256"):
        raise RuntimeError("store manifest differs from the frozen pilot")
    if not planner.get("treatment_diverged"):
        raise RuntimeError("pilot H6.5 placement candidate did not diverge")
    candidate = pathlib.Path(planner.get("candidate_plan", "")).resolve()
    if not candidate.exists():
        raise FileNotFoundError("frozen candidate is missing: %s" % candidate)
    candidate_sha = sha256(candidate)
    if not (candidate_sha == planner.get("candidate_plan_sha256")
            == EXPECTED_CANDIDATE_SHA256):
        raise RuntimeError("frozen candidate hash differs from the protocol")

    h2d = load(h2d_path)
    orders = pair_orders(args.seed, args.blocks)
    settings = {
        "model": args.model,
        "store": str(store),
        "h2d_artifact": str(h2d_path),
        "h2d_artifact_sha256": sha256(h2d_path),
        "pilot_result": str(pilot_path),
        "pilot_result_sha256": sha256(pilot_path),
        "confirmatory_protocol": str(protocol_path),
        "confirmatory_protocol_sha256": sha256(protocol_path),
        "frozen_candidate_plan": str(candidate),
        "frozen_candidate_plan_sha256": candidate_sha,
        "manifest_sha256": manifest_sha,
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

    if args.resume:
        result = load(partial)
        mismatch = [
            (key, result.get(key), value) for key, value in settings.items()
            if result.get(key) != value]
        if mismatch:
            raise ValueError("resume settings differ: %s" % mismatch)
        for relative, entry in result["source_snapshot"].items():
            source = (
                protocol_path if relative == "confirmatory_protocol"
                else REPO / relative)
            if sha256(source) != entry["source_sha256"]:
                raise RuntimeError("source changed since partial run: %s" % relative)
        result.pop("error", None)
        result.pop("traceback", None)
        result["resume_count"] = int(result.get("resume_count", 0)) + 1
        session_started = float(result["started_at_unix"])
    else:
        session_started = time.time()
        root.mkdir(parents=True)
        result = {
            "schema_version": 1,
            "kind": "h65_direct_pair_confirmatory",
            "campaign_id": CAMPAIGN_ID,
            "status": "initializing",
            "evidence_level": "L3_frozen_confirmatory",
            "exploratory": False,
            "confirmatory_protocol_satisfied": False,
            "paper_claim_scope": (
                "frozen-plan adjacent-pair Llama one-token causal confirmation"),
            "started_at_unix": session_started,
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status": git_output("status", "--short"),
            "source_snapshot": archive_sources(root, protocol_path),
            "cache_regime": "cold page cache before every timed prompt",
            "cell_isolation": "fresh process per block-method cell",
            "pairing": "adjacent alternating AB/BA",
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
            "succeeded": cache_ok, "error": cache_error}
        checkpoint(partial, result)
        if not cache_ok:
            raise RuntimeError("cold-cache preflight failed: %s" % cache_error)

        common = common_overrides(args)
        overrides = {
            "traffic-placement": {
                **common,
                "placement_policy": "traffic_density",
                "ram_tier_format": "decoded",
            },
            "h65-placement-only": {
                **common,
                "representation_policy": "multi_state",
                "representation_plan_state": str(candidate),
            },
        }
        result["status"] = "evaluating"
        checkpoint(partial, result)
        execution_index = 0
        for block, order in enumerate(orders):
            for position, method in enumerate(order):
                label = "b%02d-p%d-%s" % (block, position, method)
                completed = next((
                    cell for cell in result["cells"]
                    if cell.get("label") == label and not cell.get("error")
                ), None)
                if completed:
                    log("SKIP %s (resumed)" % label)
                    execution_index += 1
                    continue
                cell = run_cell(
                    label, worker_config(
                        args=args, method_id=method, overrides=overrides[method],
                        block=block, split="evaluation",
                        case_ids=evaluation_cases,
                        max_new_tokens=args.max_new_tokens),
                    root=root, timeout_minutes=args.cell_timeout_minutes)
                cell.update({
                    "execution_index": execution_index,
                    "pair_position": position,
                    "planned_method_order": list(order),
                })
                result["cells"] = [
                    existing for existing in result["cells"]
                    if existing.get("label") != label] + [cell]
                checkpoint(partial, result)
                execution_index += 1

        live_cells = [cell for cell in result["cells"] if not cell.get("error")]
        result["live_comparison"] = summarize(live_cells, args.blocks)
        failures = exactness_failures(
            live_cells, args.blocks, evaluation_cases)
        rows = [row for cell in live_cells for row in cell.get("rows", [])]
        expected_cells = args.blocks * len(METHODS)
        expected_rows = expected_cells * len(evaluation_cases)
        adjacent_pairs_ok = True
        for block in range(args.blocks):
            members = [
                cell for cell in result["cells"]
                if int(cell.get("block", -1)) == block]
            adjacent_pairs_ok = bool(
                adjacent_pairs_ok
                and sorted(cell.get("pair_position") for cell in members) == [0, 1]
                and sorted(cell.get("execution_index") for cell in members)
                == [2 * block, 2 * block + 1])
        positions = {
            method: [
                order[position] for order in orders
                for position in range(len(METHODS))].count(method)
            for method in METHODS}
        source_snapshot = result["source_snapshot"]
        gates = {
            "all_cells_completed": bool(
                len(result["cells"]) == expected_cells
                and all(not cell.get("error") for cell in result["cells"])),
            "expected_evaluation_rows": len(rows) == expected_rows,
            "balanced_pair_order": bool(
                sum(order[0] == METHODS[0] for order in orders) == args.blocks // 2
                and sum(order[0] == METHODS[1] for order in orders) == args.blocks // 2
                and positions == {method: args.blocks for method in METHODS}),
            "adjacent_pairs": adjacent_pairs_ok,
            "exact_output_tokens": not failures,
            "requested_token_count_completed": bool(rows and all(
                row.get("output_tokens") == args.max_new_tokens for row in rows)),
            "cold_cache_confirmed": bool(rows and all(
                row.get("cache_drop_succeeded") for row in rows)),
            "thermal_gate_passed": bool(rows and all(
                row.get("cooldown_reached_target")
                and not row.get("throttled_after_cooldown")
                and not (row.get("gpu_thermal") or {}).get("throttled")
                for row in rows)),
            "whole_cell_vram_measured": bool(rows and all(
                row.get("peak_vram_gb") is not None for row in rows)),
            "matched_logical_budget": bool(
                live_cells and all(
                    cell.get("budget") == {
                        "vram_gb": args.vram_gb, "ram_gb": args.ram_gb}
                    for cell in live_cells)),
            "frozen_pilot_plan_unchanged": bool(
                sha256(candidate) == candidate_sha
                == planner.get("candidate_plan_sha256")),
            "frozen_candidate_diverged_from_traffic": bool(
                planner.get("treatment_diverged")),
            "source_snapshot_retained": all(
                pathlib.Path(entry["snapshot"]).exists()
                and entry["source_sha256"] == entry["snapshot_sha256"]
                for entry in source_snapshot.values()),
        }
        gates["confirmatory_execution_eligible"] = all(gates.values())
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
