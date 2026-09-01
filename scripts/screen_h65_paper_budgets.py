#!/usr/bin/env python3
"""Screen H6.5 paper budgets offline before scheduling live GPU cells.

The screen reuses scheduler-aware calibration traces and builds both the
placement-only and full joint-representation candidates at every requested
RAM/VRAM point.  A live representation-ablation cell is useful only when the
two candidates select different physical states.  This script records that
gate without presenting replay estimates as measured inference results.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from afterimage.runtime.critical_path import TraceRecorder  # noqa: E402
from afterimage.runtime.h65_planner import optimize_h65_plan  # noqa: E402
from afterimage.runtime.replay_planner import manifest_fingerprint  # noqa: E402


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_budgets(value: str) -> tuple[tuple[float, float], ...]:
    budgets = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            vram, ram = (float(part) for part in item.split(":", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("budgets must use VRAM:RAM pairs") from exc
        if vram <= 0 or ram < 0:
            raise ValueError("VRAM must be positive and RAM non-negative")
        budgets.append((vram, ram))
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("at least one unique budget is required")
    return tuple(budgets)


def plan_fingerprint(plan) -> str:
    payload = {
        key: {
            "name": option.name,
            "vram_bytes": option.vram_bytes,
            "ram_bytes": option.ram_bytes,
            "storage_bytes": option.storage_bytes,
            "artifact": option.artifact,
        }
        for key, option in sorted(plan.choices.items())
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def summarize_plan(plan) -> dict:
    counts = collections.Counter(
        option.name for option in plan.choices.values())
    logical_bytes = collections.Counter()
    for option in plan.choices.values():
        logical_bytes[option.name] += max(
            option.vram_bytes, option.ram_bytes, option.storage_bytes)
    return {
        "plan_fingerprint": plan_fingerprint(plan),
        "representation_counts": dict(sorted(counts.items())),
        "representation_proxy_bytes": dict(sorted(logical_bytes.items())),
        "vram_bytes": plan.vram_bytes,
        "ram_bytes": plan.ram_bytes,
        "storage_bytes": plan.storage_bytes,
        "predicted_prepare_seconds": plan.predicted_prepare_s,
        "within_declared_budget": bool(
            plan.vram_bytes + plan.vram_headroom_bytes
            <= (plan.vram_budget_bytes or 0)
            and plan.ram_bytes <= (plan.ram_budget_bytes or 0)),
    }


def compare_plans(control, candidate) -> dict:
    transitions = collections.Counter()
    changed_proxy_bytes = 0
    keys = sorted(set(control.choices) | set(candidate.choices))
    for key in keys:
        before = control.choices.get(key)
        after = candidate.choices.get(key)
        before_name = before.name if before is not None else "missing"
        after_name = after.name if after is not None else "missing"
        if before_name == after_name:
            continue
        transitions["%s->%s" % (before_name, after_name)] += 1
        sizes = []
        for option in (before, after):
            if option is not None:
                sizes.extend((option.vram_bytes, option.ram_bytes,
                              option.storage_bytes))
        changed_proxy_bytes += max(sizes, default=0)
    control_s = float(control.predicted_prepare_s)
    candidate_s = float(candidate.predicted_prepare_s)
    return {
        "changed_tensors": sum(transitions.values()),
        "changed_proxy_bytes": changed_proxy_bytes,
        "transitions": dict(sorted(transitions.items())),
        "predicted_latency_reduction": (
            (control_s - candidate_s) / control_s if control_s > 0 else None),
    }


def checkpoint(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--h2d", required=True)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--budgets", default="3:6,3:8,4:6,4:8")
    parser.add_argument("--decode-slice-elems", type=int, default=1 << 22)
    parser.add_argument("--vram-safety-margin-gb", type=float, default=0.5)
    parser.add_argument("--search-iterations", type=int, default=128)
    parser.add_argument(
        "--minimum-predicted-representation-improvement", type=float,
        default=0.01,
        help=("minimum replay-predicted full-vs-placement improvement that "
              "makes a live representation ablation worth scheduling"))
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        budgets = parse_budgets(args.budgets)
    except ValueError as exc:
        parser.error(str(exc))
    if (len(args.traces) < 3 or args.decode_slice_elems < 1
            or args.search_iterations < 0
            or args.minimum_predicted_representation_improvement < 0
            or not 0 <= args.vram_safety_margin_gb):
        parser.error("three traces and non-negative planning settings are required")

    manifest_path = pathlib.Path(args.manifest).resolve()
    h2d_path = pathlib.Path(args.h2d).resolve()
    trace_paths = tuple(pathlib.Path(path).resolve() for path in args.traces)
    out = pathlib.Path(args.out).resolve()
    for required in (manifest_path, h2d_path, *trace_paths):
        if not required.exists():
            parser.error("required path does not exist: %s" % required)
    if out.exists():
        raise FileExistsError("refusing to overwrite budget screen: %s" % out)

    manifest = load(manifest_path)
    h2d = load(h2d_path)
    traces = [TraceRecorder.load(path) for path in trace_paths]
    h2d_gbps = float(h2d["median_stable_gbps"])
    rows = []
    for vram_gb, ram_gb in budgets:
        candidates = {}
        candidate_plans = {}
        reports = {}
        errors = {}
        for method, enable_compressed_ram in (
                ("h65-placement-only", False), ("h65-full", True)):
            try:
                planning = optimize_h65_plan(
                    manifest, traces,
                    vram_budget_gb=vram_gb, ram_budget_gb=ram_gb,
                    h2d_gbps=h2d_gbps,
                    decode_slice_elems=args.decode_slice_elems,
                    search_iterations=args.search_iterations, seed=args.seed,
                    minimum_predicted_improvement=0.0,
                    minimum_live_improvement=0.0,
                    vram_safety_margin_gb=args.vram_safety_margin_gb,
                    h2d_memory_mode=str(h2d.get("memory_mode") or "unknown"),
                    enable_compressed_ram=enable_compressed_ram)
            except ValueError as exc:
                errors[method] = str(exc)
                continue
            candidate_plans[method] = planning.candidate_plan
            candidates[method] = summarize_plan(planning.candidate_plan)
            reports[method] = dataclasses.asdict(planning.report)

        if errors:
            rows.append({
                "budget": {"vram_gb": vram_gb, "ram_gb": ram_gb},
                "representation_contrast": None,
                "material_representation_contrast": None,
                "full_vs_placement_delta": None,
                "live_run_priority": "skip_infeasible",
                "candidates": candidates,
                "planner_reports": reports,
                "errors": errors,
            })
            continue

        placement = candidates["h65-placement-only"]
        full = candidates["h65-full"]
        representation_contrast = (
            placement["plan_fingerprint"] != full["plan_fingerprint"])
        plan_delta = compare_plans(
            candidate_plans["h65-placement-only"],
            candidate_plans["h65-full"])
        predicted_reduction = plan_delta["predicted_latency_reduction"]
        material_representation_contrast = bool(
            representation_contrast
            and predicted_reduction is not None
            and predicted_reduction
            >= args.minimum_predicted_representation_improvement)
        if material_representation_contrast:
            priority = "run_full_representation_ablation"
        elif (reports["h65-placement-only"]["treatment_diverged"]
              or reports["h65-full"]["treatment_diverged"]):
            priority = "run_one_h65_candidate_not_both"
        else:
            priority = "skip_no_treatment_contrast"
        rows.append({
            "budget": {"vram_gb": vram_gb, "ram_gb": ram_gb},
            "representation_contrast": representation_contrast,
            "material_representation_contrast": material_representation_contrast,
            "full_vs_placement_delta": plan_delta,
            "live_run_priority": priority,
            "candidates": candidates,
            "planner_reports": reports,
            "errors": {},
        })

    result = {
        "schema_version": 1,
        "kind": "h65_offline_budget_divergence_screen",
        "evidence_scope": (
            "offline live-cell gate only; replay predictions are not measured "
            "inference outcomes"),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_fingerprint(manifest),
        "h2d_artifact": str(h2d_path),
        "h2d_gbps": h2d_gbps,
        "h2d_memory_mode": h2d.get("memory_mode"),
        "trace_paths": [str(path) for path in trace_paths],
        "search_iterations": args.search_iterations,
        "minimum_predicted_representation_improvement": (
            args.minimum_predicted_representation_improvement),
        "seed": args.seed,
        "rows": rows,
    }
    checkpoint(out, result)
    for row in rows:
        budget = row["budget"]
        print("v%.1f:r%.1f contrast=%s material=%s priority=%s" % (
            budget["vram_gb"], budget["ram_gb"],
            row.get("representation_contrast"),
            row.get("material_representation_contrast"),
            row["live_run_priority"]))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
