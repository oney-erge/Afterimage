#!/usr/bin/env python3
"""Score the 32-token H6.5 follow-up against its disk control.

Reads one ``run_paper_comparison.py`` result and applies the analysis frozen in
docs/h65/protocols/PROTOCOL-h65-32token-calibration-followup-v2-20261003.md. Stdlib
only, so it runs anywhere the result file does.

For each arm the primary estimate is the geometric mean over complete blocks of
the per-block geometric mean of (control wall seconds / arm wall seconds) over
the four prompts, with a two-sided 90% t interval on the log of the block
ratios (the paper's estimator, at the protocol's level). A ratio above 1 means
the arm is faster than the control.

    python scripts/analyze_h65_32token_followup.py RESULT.json \\
        --control disk-frozen --arms h65-cal32-ram8,h65-cal32-ram0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys

MIN_BLOCKS = 3
# The control runs in every block; if its own speed moves more than this
# between blocks, the environment changed during the run and every verdict is
# reported with that caveat.
CONTROL_DRIFT_LIMIT = 0.05
# t_{0.95, df}: the multiplier for a two-sided 90% interval, df = blocks - 1.
T_90 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
        8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782}


def geomean(values) -> float:
    values = list(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


def interval(block_ratios: list[float]) -> tuple[float, float] | None:
    if len(block_ratios) < 2:
        return None
    if len(block_ratios) - 1 not in T_90:
        raise ValueError("more than %d blocks: extend T_90" % (max(T_90) + 1))
    logs = [math.log(value) for value in block_ratios]
    half = (T_90[len(logs) - 1] * statistics.stdev(logs) / math.sqrt(len(logs)))
    centre = statistics.mean(logs)
    return math.exp(centre - half), math.exp(centre + half)


def rows_by_key(method: dict) -> dict[tuple[int, str], dict]:
    return {(int(row["repeat"]), row["case_id"]): row for row in method["rows"]}


def complete_blocks(rows: dict[str, dict], cases: set[str]) -> list[int]:
    """Blocks in which every compared method has every prompt. A block that
    the run did not finish is not evidence for or against any arm."""
    blocks = sorted({block for table in rows.values() for block, _ in table})
    return [block for block in blocks
            if all((block, case) in table for table in rows.values() for case in cases)]


def verdict(estimate: float, bounds, blocks: int, exact: bool) -> str:
    if not exact:
        return "invalid: output tokens differ from the control"
    if blocks < MIN_BLOCKS or bounds is None:
        return "insufficient blocks (%d of %d)" % (blocks, MIN_BLOCKS)
    if bounds[0] > 1.0:
        return "faster than the control"
    if bounds[1] < 1.0:
        return "slower than the control"
    return "inconclusive"


def thermal_exposure(result: dict, method: str, blocks: list[int],
                     measured_seconds: float) -> dict:
    """Seconds each analysed cell spent thermally throttled or power limited.
    On this laptop every cell, the control included, reports some of both
    (the GPU mostly waits on reads), so this is a balance check across arms,
    not an exclusion rule."""
    thermal = power = 0.0
    cells = 0
    for cell in result.get("cells", []):
        if cell.get("method") != method or cell.get("block") not in blocks:
            continue
        monitoring = cell.get("thermal_measurement_monitoring") or {}
        thermal += float(monitoring.get("thermal_throttle_counter_delta_seconds") or 0.0)
        power += float(monitoring.get("power_limit_counter_delta_seconds") or 0.0)
        cells += 1
    return {"cells": cells, "thermal_throttle_seconds": thermal,
            "power_limit_seconds": power,
            "power_limited_share": power / measured_seconds if measured_seconds else None}


def control_stability(tables: dict, control: str, blocks: list[int],
                      cases: set[str], expected: float | None) -> dict:
    per_block = [statistics.mean(tables[control][(block, case)]["seconds_per_token"]
                                 for case in cases) for block in blocks]
    spread = ((max(per_block) - min(per_block)) / statistics.mean(per_block)
              if per_block else None)
    mean = statistics.mean(per_block) if per_block else None
    return {
        "per_block_seconds_per_token": per_block,
        "relative_spread": spread,
        "drifted": bool(spread is not None and spread > CONTROL_DRIFT_LIMIT),
        "expected_seconds_per_token": expected,
        "relative_difference_from_expected": (
            (mean - expected) / expected if expected and mean else None),
    }


def analyze(result: dict, control: str, arms: list[str],
            expected_control_seconds_per_token: float | None = None) -> dict:
    methods = {method["method_id"]: method for method in result["methods"]}
    missing = [name for name in [control, *arms] if name not in methods]
    if missing:
        raise KeyError("result has no method(s): %s (has %s)" % (
            ", ".join(missing), ", ".join(sorted(methods))))
    tables = {name: rows_by_key(methods[name]) for name in [control, *arms]}
    cases = {case for table in tables.values() for _, case in table}
    blocks = complete_blocks(tables, cases)
    out = {"control": control, "complete_blocks": blocks,
           "requested_blocks": result.get("blocks_requested"),
           "control_stability": control_stability(
               tables, control, blocks, cases, expected_control_seconds_per_token),
           "control_thermal_exposure": thermal_exposure(
               result, control, blocks,
               sum(row["wall_seconds"] for (block, _), row in tables[control].items()
                   if block in blocks)),
           "arms": {}}
    for arm in arms:
        block_ratios, request_ratios = [], []
        exact = True
        for block in blocks:
            ratios = []
            for case in sorted(cases):
                base, row = tables[control][(block, case)], tables[arm][(block, case)]
                exact &= row["output_token_ids"] == base["output_token_ids"]
                ratios.append(base["wall_seconds"] / row["wall_seconds"])
            request_ratios.extend(ratios)
            block_ratios.append(geomean(ratios))
        used = [tables[arm][(block, case)] for block in blocks for case in cases]
        phys = [row["process_read_bytes_per_token"] / 1e9 for row in used]
        logical = [row["gb_read_per_token"] for row in used]
        bounds = interval(block_ratios) if block_ratios else None
        estimate = geomean(block_ratios) if block_ratios else float("nan")
        out["arms"][arm] = {
            "speed_ratio_vs_control": estimate,
            "ci90": list(bounds) if bounds else None,
            "block_ratios": block_ratios,
            "request_wins": sum(ratio > 1 for ratio in request_ratios),
            "requests": len(request_ratios),
            "seconds_per_token": statistics.mean(
                row["seconds_per_token"] for row in used) if used else None,
            "ssd_gb_per_token": statistics.mean(phys) if phys else None,
            "requested_gb_per_token": statistics.mean(logical) if logical else None,
            # None when the arm requested nothing from storage (a model that
            # fits its VRAM budget): there is no cache share to report.
            "cache_served_share": (1 - sum(phys) / sum(logical))
                                  if logical and sum(logical) > 0 else None,
            "peak_vram_gb": max((row["peak_vram_gb"] for row in used), default=None),
            "peak_host_rss_gb": max((row["host_rss_peak_gb"] for row in used), default=None),
            "throttled_requests": sum(
                bool(row.get("throttled_after_cooldown")
                     or row.get("power_limited_after_cooldown")) for row in used),
            "tokens_identical_to_control": exact,
            "thermal_exposure": thermal_exposure(
                result, arm, blocks, sum(row["wall_seconds"] for row in used)),
            "verdict": verdict(estimate, bounds, len(blocks), exact),
        }
    return out


def render(summary: dict) -> str:
    lines = ["control %s | complete blocks %s of %s requested" % (
        summary["control"], summary["complete_blocks"], summary["requested_blocks"]), ""]
    header = ("%-18s %8s %-17s %6s %7s %7s %6s %6s  %s" % (
        "arm", "ratio", "90% interval", "wins", "s/tok", "SSD GB", "cache", "VRAM", "verdict"))
    lines.append(header)
    for arm, row in summary["arms"].items():
        bounds = row["ci90"]
        share = row["cache_served_share"]
        lines.append("%-18s %8.3f %-17s %3d/%-2d %7.2f %7.2f %6s %6.2f  %s" % (
            arm, row["speed_ratio_vs_control"],
            "[%.3f, %.3f]" % tuple(bounds) if bounds else "n/a",
            row["request_wins"], row["requests"], row["seconds_per_token"] or 0,
            row["ssd_gb_per_token"] or 0,
            "n/a" if share is None else "%.0f%%" % (100 * share),
            row["peak_vram_gb"] or 0, row["verdict"]))
    stability = summary["control_stability"]
    if stability["per_block_seconds_per_token"]:
        lines += ["", "control per block (s/token): %s, spread %.1f%%%s" % (
            ", ".join("%.2f" % value for value in stability["per_block_seconds_per_token"]),
            100 * stability["relative_spread"],
            "  ** DRIFTED: environment changed during the run **"
            if stability["drifted"] else "")]
        if stability["relative_difference_from_expected"] is not None:
            lines.append("control vs expected %.2f s/token: %+.1f%%" % (
                stability["expected_seconds_per_token"],
                100 * stability["relative_difference_from_expected"]))
    exposure = [("control", summary["control_thermal_exposure"])] + [
        (arm, row["thermal_exposure"]) for arm, row in summary["arms"].items()]
    lines.append("share of measured time power limited: " + "; ".join(
        "%s %.0f%%" % (name, 100 * (item["power_limited_share"] or 0))
        for name, item in exposure))
    lines += ["", "ratio > 1: faster than the control. SSD GB: bytes actually read from "
              "storage per token; cache: share of requested bytes served from the OS "
              "page cache; VRAM: peak whole-process GB."]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", type=pathlib.Path)
    parser.add_argument("--control", default="disk-frozen")
    parser.add_argument("--arms", required=True, help="comma-separated method ids")
    parser.add_argument("--protocol", type=pathlib.Path,
                        help="the frozen protocol; its SHA-256 is printed and stored")
    parser.add_argument("--json-out", type=pathlib.Path)
    parser.add_argument(
        "--expect-control-seconds-per-token", type=float,
        help="the control's speed in an earlier run (D8's disk control: 17.93), "
             "to show how far this run's environment moved")
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    summary = analyze(result, args.control,
                      [a.strip() for a in args.arms.split(",") if a.strip()],
                      args.expect_control_seconds_per_token)
    summary["result_sha256"] = hashlib.sha256(args.result.read_bytes()).hexdigest()
    if args.protocol:
        summary["protocol_sha256"] = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    print(render(summary))
    if args.protocol:
        print("protocol sha256 %s" % summary["protocol_sha256"])
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
