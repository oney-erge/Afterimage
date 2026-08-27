#!/usr/bin/env python3
"""Run a bounded, cold-cache Hugging Face Accelerate disk-offload baseline."""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from afterimage.baselines.b0_hf_offload import load_hf_offload_baseline
from afterimage.bench.cachectl import drop_caches
from afterimage.bench.prompt_suite import prompt_cases, render_chat_prompt
from scripts.run_bounded_suite import canonical_peak_vram, memory_probe_extra_fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--offload-dir", default="/root/afterimage/hf_offload_14b")
    parser.add_argument("--gpu-memory", default="1500MB")
    parser.add_argument("--cpu-memory", default="8GB")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--case-ids", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    out = pathlib.Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("refusing to overwrite immutable result: %s" % out)
    tmp = out.with_suffix(out.suffix + ".partial")
    if tmp.exists():
        raise FileExistsError("partial result already exists: %s" % tmp)
    out.parent.mkdir(parents=True, exist_ok=True)

    cases = list(prompt_cases("evaluation"))
    if args.case_ids:
        wanted = {value.strip() for value in args.case_ids.split(",") if value.strip()}
        cases = [case for case in cases if case.id in wanted]
        if {case.id for case in cases} != wanted:
            parser.error("unknown case ID")

    started = time.time()
    baseline = load_hf_offload_baseline(
        args.model, args.offload_dir, gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory)
    rows = []
    payload = {
        "schema_version": 1,
        "status": "running",
        "model": args.model,
        "method": "huggingface-accelerate-disk-offload",
        "precision": "bfloat16",
        "gpu_memory_limit": args.gpu_memory,
        "cpu_memory_limit": args.cpu_memory,
        "offload_dir": args.offload_dir,
        "initialization_seconds": baseline.initialization_seconds,
        "device_map": baseline.device_map,
        "max_new_tokens": args.max_new_tokens,
        "prompt_suite": [dataclasses.asdict(case) for case in cases],
        "cache_regime": "cold page cache before every timed cell",
        "rows": rows,
        "started_at_unix": started,
    }
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for case in cases:
        prompt = render_chat_prompt(baseline.tokenizer, case)
        cache_succeeded = drop_caches()
        row = baseline.generate(prompt, args.max_new_tokens)
        mem_report = row.pop("memory_report")
        peak_vram_gb, peak_vram_source = canonical_peak_vram(mem_report)
        row.update(case_id=case.id, semantic_bucket=case.semantic_bucket,
                   expected_match=case.matches(row["text"]),
                   cache_drop_succeeded=cache_succeeded,
                   cache_drop_error=(None if cache_succeeded else
                                     "Linux page-cache drop failed"),
                   peak_vram_gb=peak_vram_gb, peak_vram_source=peak_vram_source,
                   **memory_probe_extra_fields(mem_report))
        rows.append(row)
        payload["rows"] = rows
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print("%-20s %.3f s/token" % (case.id, row["seconds_per_token"]))

    payload.update({
        "status": "complete",
        "summary": {
            "seconds_per_token": sum(row["wall_seconds"] for row in rows) /
                                 max(sum(row["tokens_generated"] for row in rows), 1),
            "median_cell_seconds_per_token": statistics.median(
                row["seconds_per_token"] for row in rows),
            "peak_vram_gb": (
                max(vram for row in rows if (vram := row["peak_vram_gb"]) is not None)
                if any(row["peak_vram_gb"] is not None for row in rows) else None),
            "expected_match_rate": statistics.mean(
                row["expected_match"] for row in rows),
            "all_cache_drops_succeeded": all(
                row["cache_drop_succeeded"] for row in rows),
        },
        "completed_at_unix": time.time(),
    })
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out)
    print("wrote immutable result %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
