#!/usr/bin/env python3
"""Randomized paired L2 screens for online Afterimage mechanisms.

This runner deliberately supports only hypotheses whose treatment can be
tested in a short live session. Offline planners and learned policies must
first clear their separate mechanism gates in ``run_bounded_suite.py``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import tempfile
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts import run_bounded_suite as bounded

from afterimage.bench.prompt_suite import PROMPT_SUITE_VERSION, prompt_cases
from afterimage.protocols import assess_paired_effect


PROTOCOLS = {
    "H12": {
        "hypothesis_id": "h12-bayesian-prefetch",
        "control": "exact-resident",
        "candidate": "bayes-prefetch",
        "minimum_effect": 0.05,
        "burn_in_tokens": 2,
    },
    "H14": {
        "hypothesis_id": "h14-coalesced-storage",
        "control": "exact-resident",
        "candidate": "coalesced-storage",
        "minimum_effect": 0.05,
        "burn_in_tokens": 1,
    },
    "H16": {
        "hypothesis_id": "h16-spec-critical-path",
        "control": "spec-fixed",
        "candidate": "spec-critical",
        "minimum_effect": 0.05,
        "burn_in_tokens": 1,
    },
    "H17": {
        "hypothesis_id": "h17-tensor-extents",
        "control": "exact-resident",
        "candidate": "tensor-extents",
        "minimum_effect": 0.05,
        "burn_in_tokens": 1,
    },
    "H18": {
        "hypothesis_id": "h18-rollback-cached-spec",
        "control": "spec-fixed",
        "candidate": "spec-cached",
        "minimum_effect": 0.05,
        "burn_in_tokens": 1,
    },
}


def _aggregate_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"rows": 0}
    tokens = sum(row["output_tokens"] for row in rows)
    wall = sum(row["wall_seconds"] for row in rows)
    return {
        "rows": len(rows), "tokens": tokens, "wall_seconds": wall,
        "seconds_per_token": wall / max(tokens, 1),
        "peak_vram_gb": max(row["peak_vram_gb"] for row in rows),
        "storage_read_calls": sum(row.get("storage_read_calls", 0) for row in rows),
        "storage_extent_bytes": sum(row.get("storage_extent_bytes", 0) for row in rows),
        "logical_bytes_read": sum(row.get("bytes_read", 0) for row in rows),
        "prefetch_wait_seconds": sum(
            row.get("prefetch_wait_seconds", 0.0) for row in rows),
        "prefetch_peak_inflight_bytes": max(
            (row.get("prefetch_peak_inflight_bytes", 0) for row in rows),
            default=0),
        "spec_cache_crops": sum(row.get("spec_cache_crops", 0) for row in rows),
        "spec_cached_prefix_tokens": sum(
            row.get("spec_cached_prefix_tokens", 0) for row in rows),
    }


def _analyse(result: dict, protocol: dict, *, level: str = "L2") -> dict:
    trials = result["trials"]
    control_rows = [row for trial in trials if trial["arm"] == "control"
                    for row in trial["rows"]]
    candidate_rows = [row for trial in trials if trial["arm"] == "candidate"
                      for row in trial["rows"]]
    control_by_pair = {(row["block"], row["case_id"]): row
                       for row in control_rows}
    candidate_by_pair = {(row["block"], row["case_id"]): row
                         for row in candidate_rows}
    shared = sorted(set(control_by_pair) & set(candidate_by_pair))
    paired = assess_paired_effect(
        [control_by_pair[key]["wall_seconds"] for key in shared],
        [candidate_by_pair[key]["wall_seconds"] for key in shared],
        minimum_effect=protocol["minimum_effect"], level=level, seed=0)
    exact = bool(shared) and all(
        control_by_pair[key]["output_token_ids"]
        == candidate_by_pair[key]["output_token_ids"] for key in shared)
    control = _aggregate_rows(control_rows)
    candidate = _aggregate_rows(candidate_rows)
    analysis = {
        "paired_effect": paired,
        "paired_token_exact": exact,
        "completed_pairs": len(shared),
        "control": control,
        "candidate": candidate,
    }

    if protocol["hypothesis_id"] == "h12-bayesian-prefetch":
        control_wait = control.get("prefetch_wait_seconds", 0.0)
        candidate_wait = candidate.get("prefetch_wait_seconds", 0.0)
        wait_reduction = (1.0 - candidate_wait / control_wait
                          if control_wait > 0 else None)
        states = [row.get("prefetch_controller_state") or {}
                  for row in candidate_rows]
        posterior_counts = [
            min(state.get("read_posterior", {}).get("count", 0),
                state.get("lead_window_posterior", {}).get("count", 0))
            for state in states]
        briers = [state.get("brier_score") for state in states
                  if state.get("brier_score") is not None]
        mechanism = {
            "posterior_observations": max(posterior_counts, default=0),
            "minimum_posterior_observations": 160,
            "posterior_brier_score": briers[-1] if briers else None,
            "exposed_wait_reduction": wait_reduction,
            "required_wait_reduction": 0.10,
            "peak_inflight_bytes": candidate.get(
                "prefetch_peak_inflight_bytes", 0),
        }
        mechanism["passed"] = bool(
            exact and mechanism["posterior_observations"] >= 160
            and wait_reduction is not None and wait_reduction >= 0.10)
    elif protocol["hypothesis_id"] in (
            "h14-coalesced-storage", "h17-tensor-extents"):
        control_calls = control.get("storage_read_calls", 0)
        candidate_calls = candidate.get("storage_read_calls", 0)
        call_reduction = (1.0 - candidate_calls / control_calls
                          if control_calls else None)
        control_bytes = control.get("storage_extent_bytes", 0)
        candidate_bytes = candidate.get("storage_extent_bytes", 0)
        byte_amplification = (candidate_bytes / control_bytes - 1.0
                              if control_bytes else None)
        required_reduction = (0.20 if protocol["hypothesis_id"] ==
                              "h17-tensor-extents" else 0.50)
        mechanism = {
            "read_call_reduction": call_reduction,
            "required_read_call_reduction": required_reduction,
            "byte_amplification": byte_amplification,
            "maximum_byte_amplification": 0.05,
        }
        mechanism["passed"] = bool(
            exact and call_reduction is not None
            and call_reduction >= required_reduction
            and byte_amplification is not None and byte_amplification <= 0.05)
    elif protocol["hypothesis_id"] == "h16-spec-critical-path":
        candidate_fingerprints = {
            row.get("tier_assignment_fingerprint") for row in candidate_rows}
        control_fingerprints = {
            row.get("tier_assignment_fingerprint") for row in control_rows}
        peak_ratio = (candidate["peak_vram_gb"] / control["peak_vram_gb"]
                      if control.get("peak_vram_gb", 0) else None)
        mechanism = {
            "candidate_tier_fingerprints": sorted(candidate_fingerprints - {None}),
            "control_tier_fingerprints": sorted(control_fingerprints - {None}),
            "treatment_diverged": bool(
                candidate_fingerprints - {None}
                and candidate_fingerprints != control_fingerprints),
            "peak_vram_ratio": peak_ratio,
            "maximum_peak_vram_ratio": 1.05,
        }
        mechanism["passed"] = bool(
            exact and mechanism["treatment_diverged"]
            and peak_ratio is not None and peak_ratio <= 1.05)
    else:
        peak_ratio = (candidate["peak_vram_gb"] / control["peak_vram_gb"]
                      if control.get("peak_vram_gb", 0) else None)
        mechanism = {
            "cache_crops": candidate["spec_cache_crops"],
            "cached_prefix_tokens": candidate["spec_cached_prefix_tokens"],
            "control_cache_crops": control["spec_cache_crops"],
            "peak_vram_ratio": peak_ratio,
            "maximum_peak_vram_ratio": 1.05,
        }
        mechanism["passed"] = bool(
            exact and mechanism["cache_crops"] > 0
            and mechanism["cached_prefix_tokens"] > 0
            and mechanism["control_cache_crops"] == 0
            and peak_ratio is not None and peak_ratio <= 1.05)
    analysis["mechanism_gate"] = mechanism
    analysis["advance_to_l3"] = bool(
        mechanism["passed"]
        and paired["decision"] == "advance_to_confirmation")
    return analysis


def _checkpoint(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--time-budget-minutes", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-airllm", action="store_true")
    parser.add_argument("--model", default=bounded.MODEL)
    parser.add_argument("--draft-model", default=bounded.DRAFT_MODEL)
    parser.add_argument("--store", default=bounded.STORE)
    parser.add_argument(
        "--case-ids", default=None,
        help="comma-separated evaluation cases; default is all four families")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-dirty-tree", action="store_true",
        help="proceed even with uncommitted changes (git status --short is "
             "non-empty); the resulting result JSON is still recorded but is "
             "not reproducible from its git_commit alone -- do not treat it "
             "as evidence for a publishable claim")
    args = parser.parse_args()
    if args.blocks < 1 or args.max_new_tokens < 1:
        parser.error("blocks and max-new-tokens must be positive")

    protocol = PROTOCOLS[args.hypothesis]
    evidence_level = ("L2" if args.blocks >= 2 and args.max_new_tokens >= 4
                      else "L1")
    out = pathlib.Path(args.out).resolve()
    partial = out.with_suffix(out.suffix + ".partial")
    if out.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite immutable result")
    out.parent.mkdir(parents=True, exist_ok=True)

    bounded.MODEL = args.model
    bounded.DRAFT_MODEL = args.draft_model
    bounded.STORE = args.store

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    dirty = bounded.command_output(["git", "-C", str(repo_root), "status", "--short"])
    if dirty and not args.allow_dirty_tree:
        raise RuntimeError(
            "refusing to run with uncommitted changes (git status --short "
            "is non-empty): a result's git_commit only reproduces the code "
            "that produced it if the tree was clean. Commit or stash first, "
            "or pass --allow-dirty-tree for a deliberately non-reproducible "
            "local/debugging run.\n" + dirty)
    tokenizer = bounded.load_tokenizer(bounded.MODEL)
    evaluation_cases = prompt_cases("evaluation")
    if args.case_ids:
        by_id = {case.id: case for case in evaluation_cases}
        requested = [part.strip() for part in args.case_ids.split(",")
                     if part.strip()]
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            parser.error("unknown evaluation cases: %s" % ", ".join(unknown))
        evaluation_cases = tuple(by_id[case_id] for case_id in requested)
    evaluation = bounded.render_cases(tokenizer, evaluation_cases)
    calibration = bounded.render_cases(tokenizer, prompt_cases("calibration")[:2])
    for item in evaluation + calibration:
        item["tokenizer"] = tokenizer

    started = time.perf_counter()
    deadline = started + args.time_budget_minutes * 60
    method_ids = {protocol["control"], protocol["candidate"]}
    calibration_temp = tempfile.TemporaryDirectory(prefix="afterimage-regulated-")
    critical = None
    if "spec-critical" in method_ids:
        critical = bounded.prepare_critical_profile(
            tokenizer, pathlib.Path(calibration_temp.name), deadline)
    draft_model = None
    if any(bounded.METHODS[method_id].overrides.get("draft_mode") == "model"
           for method_id in method_ids):
        from afterimage.runtime.streaming_engine import load_draft_model
        draft_model = load_draft_model(bounded.DRAFT_MODEL, device="cuda")
    result = {
        "schema_version": 1,
        "status": "running",
        "evidence_level": (
            "L2_regulated_exploratory" if evidence_level == "L2"
            else "L1_mechanism_screen"),
        "confirmatory_protocol_satisfied": False,
        "hypothesis_id": protocol["hypothesis_id"],
        "protocol": protocol,
        "prompt_suite_version": PROMPT_SUITE_VERSION,
        "evaluation_case_ids": [item["case"].id for item in evaluation],
        "calibration_case_ids": [item["case"].id for item in calibration],
        "blocks": args.blocks,
        "max_new_tokens": args.max_new_tokens,
        "time_budget_minutes": args.time_budget_minutes,
        "randomization_seed": args.seed,
        "cache_regime": "cold page cache before every timed cell",
        "model": bounded.MODEL,
        "draft_model": bounded.DRAFT_MODEL,
        "store": bounded.STORE,
        "environment": bounded.environment_manifest(repo_root, tokenizer),
        "reproducible_from_commit": not bool(dirty),
        "trials": [], "airllm_anchor": None, "failures": [],
    }
    _checkpoint(partial, result)

    if not args.skip_airllm and time.perf_counter() < deadline:
        try:
            rows, metadata = bounded.run_airllm(
                bounded.METHODS["airllm"], evaluation,
                args.max_new_tokens, deadline)
            result["airllm_anchor"] = {
                "rows": rows, "metadata": metadata,
                "summary": _aggregate_rows(rows),
            }
        except Exception as exc:
            result["failures"].append({
                "phase": "airllm_anchor", "error": repr(exc),
                "traceback": traceback.format_exc()})
        _checkpoint(partial, result)

    rng = random.Random(args.seed)
    for block in range(args.blocks):
        order = ["control", "candidate"]
        rng.shuffle(order)
        for order_index, arm in enumerate(order):
            if time.perf_counter() >= deadline:
                result["failures"].append({
                    "block": block, "arm": arm,
                    "error": "not started: time budget exhausted"})
                continue
            method_id = protocol[arm]
            method = bounded.METHODS[method_id]
            try:
                rows, metadata = bounded.run_afterimage(
                    method, evaluation, args.max_new_tokens, deadline,
                    draft_model=draft_model,
                    critical_profile=critical["path"] if critical else None,
                    burn_in_rendered=calibration,
                    burn_in_tokens=protocol["burn_in_tokens"])
                for row in rows:
                    row["block"] = block
                    row["arm"] = arm
                    row["randomized_order"] = order_index
                result["trials"].append({
                    "block": block, "arm": arm,
                    "randomized_order": order_index,
                    "method_id": method_id, "rows": rows,
                    "metadata": metadata,
                })
            except Exception as exc:
                result["failures"].append({
                    "block": block, "arm": arm, "method": method_id,
                    "error": repr(exc), "traceback": traceback.format_exc()})
            result["elapsed_seconds"] = time.perf_counter() - started
            _checkpoint(partial, result)

    try:
        result["analysis"] = _analyse(result, protocol, level=evidence_level)
    except Exception as exc:
        result["failures"].append({
            "phase": "analysis", "error": repr(exc),
            "traceback": traceback.format_exc()})
    result["elapsed_seconds"] = time.perf_counter() - started
    expected_trials = args.blocks * 2
    result["status"] = (
        "complete" if len(result["trials"]) == expected_trials
        and all(len(trial["rows"]) == len(evaluation)
                for trial in result["trials"])
        else "incomplete")
    result["completed_at_unix"] = time.time()
    _checkpoint(partial, result)
    partial.replace(out)
    calibration_temp.cleanup()
    print("wrote immutable result %s" % out)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
