#!/usr/bin/env python3
"""Runs exactly one (block, method) cell of the paper comparison, in its own
fresh process, and exits.

Why this is a separate process rather than a function call
------------------------------------------------------------
run_paper_comparison.py used to call run_airllm/run_accelerate/run_dfloat11/
run_afterimage directly, in-process, one after another within a block. That
under-isolates three ways that matter for a paper's numbers:

  1. spec-fixed loads a resident draft model (afterimage/runtime/
     streaming_engine.load_draft_model) that stayed allocated on the GPU for
     the rest of the campaign once first loaded -- every method that ran
     afterward in the same process reported peak_vram_gb inflated by that
     draft model's footprint, and had that much less real VRAM headroom
     than its own configuration claimed.
  2. DFloat11's custom CUDA decompression kernels and AirLLM's own internal
     state are not guaranteed to fully release GPU/allocator state on
     `del model; gc.collect(); torch.cuda.empty_cache()` -- a library-level
     leak or a cached allocator pool surviving into the next method's
     measurement is invisible from inside the same process.
  3. Peak host RAM is meaningless measured mid-process: Python's allocator
     does not reliably return freed pages to the OS, so RSS climbs across
     methods regardless of how much any individual method actually needed.

A fresh OS process per cell is the only isolation guarantee for all three:
process exit unconditionally releases the CUDA context, and this worker
reports its own peak RSS as an OS-level high-water mark over its own
lifetime (resource.getrusage(RUSAGE_SELF).ru_maxrss), not a same-process
running total polluted by whatever ran before it.

Contract
--------
    python run_paper_comparison_worker.py --config <cell.json> --out <result.json>

<cell.json> (written by run_paper_comparison.py, one per cell) holds:
    method_id, model, dfloat11_model, draft_model, store, n_tokens, block,
    warmup_tokens, cooldown_seconds, cooldown_max_temp_c, seconds_remaining,
    case_ids (or null for every evaluation case)

<result.json> (written by this process before it exits) holds:
    {"rows": [...], "metadata": {...}, "peak_host_rss_bytes": int|None,
     "error": str|None, "traceback": str|None}

seconds_remaining is a duration, not an absolute deadline: time.perf_counter
has no defined cross-process epoch (CPython documents it as "the reference
point of the returned value is undefined"), so a perf_counter timestamp
computed in the orchestrator's process is not safely comparable against
time.perf_counter() read inside this one, even though in practice both
happen to use CLOCK_MONOTONIC on Linux. This worker computes its own
deadline locally instead: `time.perf_counter() + seconds_remaining`.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import pathlib
import sys
import time
import traceback

try:
    import resource  # POSIX only. This suite is WSL2/Linux-only by design
                     # (see cpu_model()'s docstring in run_bounded_suite.py
                     # and drop_caches()'s /proc/sys/vm/drop_caches use), so
                     # the guard exists to keep this module importable for
                     # unit testing on any platform, not to support running
                     # real cells on Windows.
except ImportError:
    resource = None

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from afterimage.bench.prompt_suite import prompt_cases
from scripts.run_bounded_suite import (
    METHODS,
    load_tokenizer,
    log,
    render_cases,
    run_accelerate,
    run_afterimage,
    run_airllm,
    run_dfloat11,
)
from scripts import run_bounded_suite as bounded


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    # ru_maxrss is kibibytes on Linux, bytes on macOS/BSD; this project's
    # documented, tested environment is WSL2/Linux (see cpu_model()'s own
    # docstring in run_bounded_suite.py), so the KiB convention is assumed
    # deliberately rather than guessed at cross-platform.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def run_cell(config: dict) -> dict:
    bounded.MODEL = config["model"]
    bounded.STORE = config["store"]
    bounded.COOLDOWN_SECONDS = config["cooldown_seconds"]
    bounded.COOLDOWN_MAX_TEMPERATURE_C = config["cooldown_max_temp_c"]

    # METHODS["dfloat11"/"dfloat11-gpu-resident"].overrides["model_id"] is
    # baked in at module-import time from DFLOAT11_MODEL's value *then* --
    # run_dfloat11's own `method.overrides.get("model_id", DFLOAT11_MODEL)`
    # fallback is therefore dead code, since the key is always already
    # present. A --dfloat11-model override has to rewrite the overrides
    # dict directly, not the module-level default, or it is silently
    # ineffective.
    if config.get("dfloat11_model"):
        for method_id in ("dfloat11", "dfloat11-gpu-resident"):
            if method_id in METHODS:
                METHODS[method_id] = dataclasses.replace(
                    METHODS[method_id],
                    overrides={**METHODS[method_id].overrides,
                              "model_id": config["dfloat11_model"]})

    method_id = config["method_id"]
    method = METHODS[method_id]
    n_tokens = config["n_tokens"]
    block = config["block"]
    warmup_tokens = config["warmup_tokens"]
    deadline = time.perf_counter() + config["seconds_remaining"]

    tokenizer = load_tokenizer(config["model"])
    all_cases = prompt_cases("evaluation")
    if config.get("case_ids"):
        by_id = {case.id: case for case in all_cases}
        cases = tuple(by_id[case_id] for case_id in config["case_ids"])
    else:
        cases = all_cases
    rendered = render_cases(tokenizer, cases)
    for item in rendered:
        item["tokenizer"] = tokenizer

    draft_model = None
    try:
        if method.kind == "airllm":
            rows, metadata = run_airllm(
                method, rendered, n_tokens, deadline, None,
                repeats=1, repeat_offset=block, warmup_tokens=warmup_tokens)
        elif method.kind == "accelerate":
            rows, metadata = run_accelerate(
                method, rendered, n_tokens, deadline, None,
                repeats=1, repeat_offset=block, warmup_tokens=warmup_tokens)
        elif method.kind == "dfloat11":
            rows, metadata = run_dfloat11(
                method, rendered, n_tokens, deadline, None,
                repeats=1, repeat_offset=block, warmup_tokens=warmup_tokens)
        else:
            if method_id == "spec-fixed":
                from afterimage.runtime.streaming_engine import load_draft_model
                log("  loading resident draft model %s" % config["draft_model"])
                draft_model = load_draft_model(config["draft_model"], device="cuda")
            rows, metadata = run_afterimage(
                method, rendered, n_tokens, deadline,
                draft_model=draft_model,
                burn_in_rendered=rendered[:1] if warmup_tokens > 0 else None,
                burn_in_tokens=warmup_tokens,
                rows_checkpoint=None, repeats=1, repeat_offset=block)
        return {"rows": rows, "metadata": metadata,
               "peak_host_rss_bytes": _peak_rss_bytes(), "error": None,
               "traceback": None}
    except Exception as exc:
        return {"rows": [], "metadata": {}, "peak_host_rss_bytes": _peak_rss_bytes(),
               "error": repr(exc), "traceback": traceback.format_exc()}
    finally:
        if draft_model is not None:
            del draft_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    output = run_cell(config)
    pathlib.Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    return 0 if output["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
