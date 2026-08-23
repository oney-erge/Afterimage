#!/usr/bin/env python3
"""Run a durable, staged cross-family benchmark campaign."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from afterimage.bench.campaign import (
    atomic_json,
    load_campaign_config,
    model_preflight,
    render_campaign_markdown,
    summarize_result,
)


def checkpoint(path: pathlib.Path, campaign: dict) -> None:
    atomic_json(path, campaign)
    report = path.parent / "INTERIM_RESULTS.md"
    temporary = report.with_suffix(".md.tmp")
    temporary.write_text(render_campaign_markdown(campaign), encoding="utf-8")
    temporary.replace(report)


def run_logged(command: list[str], log_path: pathlib.Path,
               repo_root: pathlib.Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ %s\n" % " ".join(command))
        log.flush()
        process = subprocess.Popen(
            command, cwd=repo_root, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait(), time.perf_counter() - started


def stage_record(model_state: dict, stage_id: str) -> dict:
    for stage in model_state["stages"]:
        if stage["id"] == stage_id:
            return stage
    stage = {"id": stage_id, "status": "pending"}
    model_state["stages"].append(stage)
    return stage


def preserve_interrupted_partial(artifact: pathlib.Path, *, stamp: int | None = None
                                 ) -> pathlib.Path | None:
    """Move a runner checkpoint aside before restarting that stage.

    Runner checkpoints are evidence, but the bounded runners intentionally
    refuse to overwrite them.  A campaign resume therefore archives the
    checkpoint instead of deleting it and starts a fresh immutable result.
    """
    partial = pathlib.Path(str(artifact) + ".partial")
    if not partial.exists():
        return None
    stamp = int(time.time()) if stamp is None else stamp
    archived = pathlib.Path(str(artifact) + ".interrupted-%d.partial" % stamp)
    suffix = 1
    while archived.exists():
        archived = pathlib.Path(
            str(artifact) + ".interrupted-%d-%d.partial" % (stamp, suffix))
        suffix += 1
    partial.replace(archived)
    return archived


def compression_command(model: dict, partial_store: pathlib.Path,
                        workers: int) -> list[str]:
    return [
        sys.executable, "-u", "-m", "afterimage.cli", "compress",
        model["model_id"], "--out", str(partial_store), "--yes",
        "--progress-every", "10", "--workers", str(workers),
    ]


def benchmark_command(model: dict, stage: dict,
                      artifact: pathlib.Path, config: dict) -> list[str]:
    runner = stage["runner"]
    cases = ",".join(stage.get("case_ids", []))
    if runner == "bounded":
        command = [
            sys.executable, "-u", "scripts/run_bounded_suite.py",
            "--model", model["model_id"], "--store", model["store"],
            "--methods", ",".join(stage["methods"]),
            "--max-new-tokens", str(stage["max_new_tokens"]),
            "--time-budget-minutes", str(stage["time_budget_minutes"]),
            "--out", str(artifact),
        ]
        if cases:
            command.extend(["--case-ids", cases])
        return command
    if runner == "regulated":
        command = [
            sys.executable, "-u", "scripts/run_regulated_pair.py",
            "--hypothesis", stage["hypothesis"],
            "--model", model["model_id"], "--store", model["store"],
            "--blocks", str(stage["blocks"]),
            "--max-new-tokens", str(stage["max_new_tokens"]),
            "--time-budget-minutes", str(stage["time_budget_minutes"]),
            "--skip-airllm", "--out", str(artifact),
        ]
        if cases:
            command.extend(["--case-ids", cases])
        return command
    if runner == "hf_accelerate":
        common = config["common"]
        command = [
            sys.executable, "-u", "scripts/run_hf_offload_baseline.py",
            "--model", model["model_id"],
            "--offload-dir", model["hf_offload_dir"],
            "--gpu-memory", common["gpu_memory"],
            "--cpu-memory", common["cpu_memory"],
            "--max-new-tokens", str(stage["max_new_tokens"]),
            "--out", str(artifact),
        ]
        if cases:
            command.extend(["--case-ids", cases])
        return command
    raise ValueError("unknown campaign runner: %s" % runner)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/cross_model_benchmark_v1.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--roles", default="small,large")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-compression", action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="resume campaign.json.partial; interrupted runner checkpoints "
             "are preserved with an .interrupted-* suffix")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    config_path = pathlib.Path(args.config).resolve()
    config = load_campaign_config(config_path)
    selected_roles = {part.strip() for part in args.roles.split(",") if part.strip()}
    output = pathlib.Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "campaign.json"
    partial_path = output / "campaign.json.partial"
    if final_path.exists():
        raise FileExistsError(
            "refusing to overwrite campaign evidence: %s" % output)
    if partial_path.exists():
        if not args.resume:
            raise FileExistsError(
                "campaign checkpoint exists; pass --resume or choose a new "
                "output directory: %s" % output)
        campaign = json.loads(partial_path.read_text(encoding="utf-8"))
        if campaign.get("campaign_id") != config["campaign_id"]:
            raise RuntimeError("campaign checkpoint/config id mismatch")
        campaign["status"] = "running"
        campaign["resumed_at_unix"] = time.time()
        campaign["failures"] = []
    else:
        if args.resume:
            raise FileNotFoundError(
                "--resume requires campaign.json.partial in %s" % output)
        campaign = {
            "schema_version": 1,
            "campaign_id": config["campaign_id"],
            "status": "running",
            "config": str(config_path),
            "started_at_unix": time.time(),
            "models": [],
            "deferred_models": config.get("deferred_models", []),
            "excluded_hypothesis_families": config.get(
                "excluded_hypothesis_families", []),
            "failures": [],
        }
    checkpoint(partial_path, campaign)

    for model in config["models"]:
        if not model.get("enabled") or model["role"] not in selected_roles:
            continue
        model_state = next(
            (item for item in campaign["models"] if item.get("id") == model["id"]),
            None)
        if model_state is None:
            model_state = {
                "id": model["id"], "role": model["role"],
                "model_id": model["model_id"], "stages": [],
            }
            campaign["models"].append(model_state)
        preflight = stage_record(model_state, "preflight")
        if not (args.resume and preflight.get("status") == "passed"):
            preflight["status"] = "running"
            checkpoint(partial_path, campaign)
            try:
                preflight["summary"] = model_preflight(model)
                preflight["status"] = preflight["summary"]["status"]
                if preflight["status"] != "passed":
                    raise RuntimeError("Afterimage model-layout preflight failed")
            except Exception as exc:
                preflight.update(status="failed", error=repr(exc),
                                 traceback=traceback.format_exc())
                campaign["failures"].append({
                    "model": model["id"], "stage": "preflight",
                    "error": repr(exc)})
                checkpoint(partial_path, campaign)
                continue
        checkpoint(partial_path, campaign)

        if args.preflight_only:
            continue

        store = pathlib.Path(model["store"])
        compress = stage_record(model_state, "compress")
        if (store / "manifest.json").exists():
            manifest = json.loads(
                (store / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("model_id") != model["model_id"]:
                raise RuntimeError("store model_id does not match campaign model")
            compress.update(status="reused", artifact=str(store), summary={
                key: manifest.get(key) for key in (
                    "total_orig_bytes", "total_comp_bytes", "ratio")})
        elif args.skip_compression:
            compress.update(status="skipped", error="--skip-compression")
        else:
            partial_store = pathlib.Path(str(store) + ".partial")
            if partial_store.exists():
                compress.update(
                    status="failed",
                    error="incomplete store already exists: %s" % partial_store)
            else:
                compress.update(status="running", artifact=str(partial_store))
                log_path = output / model["id"] / "compress.log"
                compress["log"] = str(log_path)
                checkpoint(partial_path, campaign)
                code, elapsed = run_logged(
                    compression_command(model, partial_store, args.workers),
                    log_path, repo_root)
                compress.update(return_code=code, elapsed_seconds=elapsed)
                if code == 0 and (partial_store / "manifest.json").exists():
                    store.parent.mkdir(parents=True, exist_ok=True)
                    partial_store.replace(store)
                    manifest = json.loads(
                        (store / "manifest.json").read_text(encoding="utf-8"))
                    compress.update(status="passed", artifact=str(store), summary={
                        key: manifest.get(key) for key in (
                            "total_orig_bytes", "total_comp_bytes", "ratio")})
                else:
                    compress.update(status="failed", error="compression command failed")
        if compress["status"] == "failed":
            campaign["failures"].append({
                "model": model["id"], "stage": "compress",
                "error": compress.get("error")})
            checkpoint(partial_path, campaign)
            continue
        checkpoint(partial_path, campaign)

        if args.skip_benchmarks:
            continue
        for definition in model.get("benchmarks", []):
            stage = stage_record(model_state, definition["id"])
            artifact = output / model["id"] / (definition["id"] + ".json")
            log_path = output / model["id"] / (definition["id"] + ".log")
            if args.resume and artifact.exists():
                stage.update(
                    status="passed", artifact=str(artifact), log=str(log_path),
                    summary=summarize_result(artifact),
                    recovered_from_completed_artifact=True)
                checkpoint(partial_path, campaign)
                continue
            if args.resume:
                archived = preserve_interrupted_partial(artifact)
                if archived is not None:
                    stage["interrupted_artifact"] = str(archived)
            stage.update(status="running", artifact=str(artifact), log=str(log_path))
            checkpoint(partial_path, campaign)
            command = benchmark_command(model, definition, artifact, config)
            try:
                code, elapsed = run_logged(command, log_path, repo_root)
                stage.update(return_code=code, elapsed_seconds=elapsed)
                if artifact.exists():
                    stage["summary"] = summarize_result(artifact)
                stage["status"] = "passed" if code == 0 else "failed"
                if code != 0:
                    stage["error"] = "benchmark command returned %d" % code
                if definition["id"] == "smoke" and stage["status"] == "passed":
                    methods = stage.get("summary", {}).get("methods", [])
                    exact = next((row for row in methods
                                  if row.get("method") == "exact-min"), None)
                    if (exact is None or exact.get("completed_cases") != 1
                            or exact.get("expected_match_rate") != 1.0):
                        stage["status"] = "failed"
                        stage["error"] = (
                            "exact smoke gate failed; later timing stages are invalid")
            except Exception as exc:
                stage.update(status="failed", error=repr(exc),
                             traceback=traceback.format_exc())
            if stage["status"] == "failed":
                campaign["failures"].append({
                    "model": model["id"], "stage": definition["id"],
                    "error": stage.get("error")})
            checkpoint(partial_path, campaign)
            if definition["id"] == "smoke" and stage["status"] == "failed":
                break

    campaign["failures"] = [
        {"model": model_state.get("id"), "stage": stage.get("id"),
         "error": stage.get("error")}
        for model_state in campaign["models"]
        for stage in model_state.get("stages", [])
        if stage.get("status") == "failed"
    ]
    campaign["completed_at_unix"] = time.time()
    campaign["status"] = (
        "complete" if not campaign["failures"] else "complete_with_failures")
    checkpoint(partial_path, campaign)
    partial_path.replace(final_path)
    print("wrote campaign evidence %s" % final_path)
    return 0 if not campaign["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
