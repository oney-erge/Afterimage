#!/usr/bin/env python3
"""Measure pinned-host-to-CUDA bandwidth for representation/H9 cost models."""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", default="32,64,128,256,512")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sizes = [int(value.strip()) for value in args.sizes_mib.split(",")
             if value.strip()]
    if not sizes or min(sizes) < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("sizes/repeats must be positive and warmups non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    out = pathlib.Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("refusing to overwrite immutable result: %s" % out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for size_mib in sizes:
        nbytes = size_mib << 20
        source = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        destination = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        for _ in range(args.warmups):
            destination.copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            destination.copy_(source, non_blocking=True)
            end.record()
            end.synchronize()
            seconds = start.elapsed_time(end) / 1000.0
            samples.append(nbytes / max(seconds, 1e-12) / 1e9)
        rows.append({
            "size_mib": size_mib,
            "median_gbps": statistics.median(samples),
            "min_gbps": min(samples),
            "max_gbps": max(samples),
            "samples_gbps": samples,
        })
        del source, destination
        torch.cuda.empty_cache()

    stable = [row["median_gbps"] for row in rows if row["size_mib"] >= 64]
    payload = {
        "schema_version": 1,
        "completed_at_unix": time.time(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "rows": rows,
        "median_stable_gbps": statistics.median(stable),
    }
    tmp = out.with_suffix(out.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out)
    print("wrote immutable result %s (median %.3f GB/s)" %
          (out, payload["median_stable_gbps"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
