#!/usr/bin/env python3
"""Run one isolated H6.5 paper-matrix cell.

The parent process owns calibration, frozen plans, counterbalancing, and hard
timeouts.  This worker initializes the model once, runs every prompt assigned
to one block/method cell, and records whole-process memory and thermal state.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
import time
import traceback


REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from afterimage.bench.memory import MemoryProbe  # noqa: E402
from afterimage.bench.prompt_suite import prompt_cases  # noqa: E402
import scripts.run_bounded_suite as rbs  # noqa: E402
from scripts.run_paper_comparison_worker import ThermalSampler  # noqa: E402


def _write_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _whole_cell_peak(report) -> tuple[float | None, str | None]:
    if report.smi_delta_gb is not None:
        return report.smi_delta_gb, "whole_cell_nvidia_smi_delta"
    if report.torch_peak_vram_gb is not None:
        return report.torch_peak_vram_gb, "whole_cell_torch_allocator"
    return None, None


def run(config: dict) -> dict:
    rbs.MODEL = config["model"]
    rbs.STORE = config["store"]
    rbs.COOLDOWN_SECONDS = float(config["cooldown_seconds"])
    rbs.COOLDOWN_MAX_TEMPERATURE_C = float(config["cooldown_max_temp_c"])

    started = time.perf_counter()
    deadline = started + float(config["time_budget_minutes"]) * 60.0
    rows: list[dict] = []
    metadata: dict = {}
    error = None
    error_traceback = None

    try:
        with ThermalSampler() as thermal_sampler:
            # The outer probe starts before tokenizer/model/engine creation so
            # persistent representations are part of the reported VRAM peak.
            with MemoryProbe() as whole_probe:
                tokenizer = rbs.load_tokenizer(config["model"])
                available = {
                    case.id: case
                    for case in prompt_cases(config.get("case_split", "evaluation"))
                }
                missing = sorted(set(config["case_ids"]) - set(available))
                if missing:
                    raise ValueError("unknown prompt case(s): %s" % ", ".join(missing))
                cases = tuple(available[case_id] for case_id in config["case_ids"])
                rendered = rbs.render_cases(tokenizer, cases)
                for item in rendered:
                    item["tokenizer"] = tokenizer

                method = rbs.Method(
                    config["method_id"], config["method_id"], "afterimage",
                    config["overrides"], "reference_execution_equivalent", 60.0)
                rows, metadata = rbs.run_afterimage(
                    method, rendered, int(config["max_new_tokens"]), deadline,
                    critical_profile=None, repeats=1,
                    repeat_offset=int(config["block"]), rows_checkpoint=None)
            whole_report = whole_probe.report()
        thermal = thermal_sampler.summary()

        peak_vram_gb, peak_source = _whole_cell_peak(whole_report)
        for row in rows:
            row["generation_only_peak_vram_gb"] = row.get("peak_vram_gb")
            row["generation_only_peak_vram_source"] = row.get("peak_vram_source")
            row["peak_vram_gb"] = peak_vram_gb
            row["peak_vram_source"] = peak_source
            row["whole_cell_smi_baseline_vram_gb"] = (
                whole_report.smi_baseline_used_mb / 1000.0
                if whole_report.smi_baseline_used_mb is not None else None)
            row["whole_cell_smi_peak_vram_gb"] = (
                whole_report.smi_peak_used_mb / 1000.0
                if whole_report.smi_peak_used_mb is not None else None)
            row["whole_cell_torch_peak_vram_gb"] = whole_report.torch_peak_vram_gb
    except Exception as exc:
        error = repr(exc)
        error_traceback = traceback.format_exc()
        thermal = thermal_sampler.summary() if "thermal_sampler" in locals() else {}

    return {
        "schema_version": 1,
        "block": int(config["block"]),
        "method_id": config["method_id"],
        "budget": config["budget"],
        "case_split": config.get("case_split", "evaluation"),
        "case_ids": list(config["case_ids"]),
        "max_new_tokens": int(config["max_new_tokens"]),
        "rows": rows,
        "summary": rbs.aggregate(rows),
        "run_meta": metadata,
        "thermal_monitoring": thermal,
        "peak_host_rss_bytes": _peak_rss_bytes(),
        "elapsed_seconds": time.perf_counter() - started,
        "error": error,
        "traceback": error_traceback,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    result = run(config)
    _write_atomic(pathlib.Path(args.out), result)
    return 1 if result["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
