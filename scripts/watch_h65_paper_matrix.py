#!/usr/bin/env python3
"""Low-frequency health/ETA watcher for a resumable H6.5 paper matrix."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import statistics
import subprocess
import time


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_processes() -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def latest_log(artifacts: pathlib.Path) -> pathlib.Path | None:
    logs = list((artifacts / "cells").glob("*.log"))
    return max(logs, key=lambda path: path.stat().st_mtime) if logs else None


def status(result_path: pathlib.Path, total_cells: int,
           stale_minutes: float) -> dict:
    final_exists = result_path.exists()
    partial = result_path.with_suffix(result_path.suffix + ".partial")
    source = result_path if final_exists else partial
    payload = load(source) if source.exists() else {}
    cells = payload.get("cells") or []
    completed = [cell for cell in cells if not cell.get("error")]
    failed = [cell for cell in cells if cell.get("error")]
    durations = [
        float(cell["cell_wall_seconds"]) for cell in completed
        if cell.get("cell_wall_seconds") is not None
    ]
    typical = statistics.median(durations) if durations else None
    remaining = max(0, total_cells - len(completed))
    eta_seconds = typical * remaining if typical is not None else None

    artifacts = result_path.parent / (result_path.stem + "-artifacts")
    active_log = latest_log(artifacts)
    log_age_seconds = (
        time.time() - active_log.stat().st_mtime if active_log else None)
    gpu = gpu_processes()
    running = payload.get("status") not in {"complete", "failed"} and not final_exists
    warnings = []
    if running and log_age_seconds is not None and log_age_seconds > stale_minutes * 60:
        warnings.append("active log stale for %.1f minutes" % (log_age_seconds / 60))
    if running and not gpu:
        warnings.append("no GPU compute process visible while campaign says running")
    if failed:
        warnings.append("%d failed cell(s) recorded" % len(failed))
    return {
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "campaign_status": payload.get("status", "not_started"),
        "completed_cells": len(completed),
        "failed_cells": len(failed),
        "total_cells": total_cells,
        "remaining_cells": remaining,
        "median_completed_cell_minutes": (
            typical / 60 if typical is not None else None),
        "estimated_remaining_minutes": (
            eta_seconds / 60 if eta_seconds is not None else None),
        "active_log": str(active_log) if active_log else None,
        "active_log_age_seconds": log_age_seconds,
        "gpu_compute_processes": gpu,
        "warnings": warnings,
        "final_result_exists": final_exists,
    }


def write_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def render(item: dict) -> str:
    eta = item["estimated_remaining_minutes"]
    eta_text = "unknown" if eta is None else "%.0f min" % eta
    warning = (
        " WARNING: " + "; ".join(item["warnings"])
        if item["warnings"] else " healthy")
    return ("WATCH {completed_cells}/{total_cells} cells, status={campaign_status}, "
            "ETA={eta}{warning}").format(eta=eta_text, warning=warning, **item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--status-out", required=True)
    parser.add_argument("--total-cells", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--stale-minutes", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.total_cells < 1 or args.interval_seconds <= 0 or args.stale_minutes <= 0:
        parser.error("counts and intervals must be positive")

    result_path = pathlib.Path(args.result).resolve()
    status_out = pathlib.Path(args.status_out).resolve()
    while True:
        item = status(result_path, args.total_cells, args.stale_minutes)
        write_atomic(status_out, item)
        print(render(item), flush=True)
        if (args.once or item["final_result_exists"]
                or item["campaign_status"] == "failed"):
            return 1 if item["warnings"] else 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
