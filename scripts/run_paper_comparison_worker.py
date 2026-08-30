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
import statistics
import sys
import threading
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
    run_deepspeed_zero_inference,
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


def is_capacity_failure(exc: BaseException) -> bool:
    """Whether this exception is a predeclared VRAM/host-memory capacity
    failure -- the method could not fit in the available hardware, which
    is a real, reportable OUTCOME (see run_paper_comparison.py's
    capacity_failed_cells / paper_eligibility), not a bug or a missing
    measurement that --resume should keep retrying forever.

    Checks the message text, not only the exception type: the type most
    callers would expect here, torch.cuda.OutOfMemoryError, is raised by
    torch's OWN caching allocator when it cannot satisfy a request --  but
    a real OOM this project has actually hit (DFloat11Model.from_pretrained
    calling tensor.pin_memory() during CUDA-side loading) surfaced as a
    plain RuntimeError with "CUDA error: out of memory" in its text
    instead, from a CUDA allocation that bypassed torch's allocator
    entirely. Matching on type alone would silently miss that real case.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def thermal_monitor_summary(samples: list[dict], interval_s: float = 1.0) -> dict:
    """Aggregates ~1 Hz gpu_thermal_snapshot() samples taken across a
    cell's entire timed portion, not just at cell start/end -- this
    project has directly measured an RTX 3080 clock collapse from ~1890
    MHz to ~780 MHz *during* a run (see cool_down()'s docstring in
    run_bounded_suite.py). A cell running a 100-128-token decode workload
    on this reference laptop GPU can take minutes: long enough to start
    throttling mid-measurement even though the mandatory pre-cell
    cool_down() confirmed a clean start, and long enough for a throttle to
    both begin and clear again before a single end-of-cell snapshot would
    have caught it.
    """
    clocks: list[float] = []
    temps: list[float] = []
    power_samples: list[tuple[float | None, float]] = []
    any_throttle = False
    any_known = False
    for sample in samples:
        try:
            clocks.append(float(sample.get("sm_clock_mhz")))
        except (TypeError, ValueError):
            pass
        try:
            temps.append(float(sample.get("temperature_c")))
        except (TypeError, ValueError):
            pass
        try:
            power = float(sample.get("power_draw_w"))
            try:
                sampled_at = float(sample.get("sampled_at_monotonic_s"))
            except (TypeError, ValueError):
                sampled_at = None
            power_samples.append((sampled_at, power))
        except (TypeError, ValueError):
            pass
        throttled = sample.get("throttled")
        if throttled is not None:
            any_known = True
        if throttled is True:
            any_throttle = True

    def flag_summary(key: str) -> dict:
        known = [sample.get(key) for sample in samples if sample.get(key) is not None]
        active = sum(value is True for value in known)
        duration_s = 0.0
        duration_observed = False
        for current, following in zip(samples, samples[1:]):
            if current.get(key) is not True:
                continue
            try:
                started_at = float(current["sampled_at_monotonic_s"])
                ended_at = float(following["sampled_at_monotonic_s"])
            except (KeyError, TypeError, ValueError):
                continue
            if ended_at > started_at:
                duration_s += ended_at - started_at
                duration_observed = True
        return {
            "any": (active > 0) if known else None,
            "active_samples": active,
            "observed_samples": len(known),
            "sample_fraction": (active / len(known)) if known else None,
            "duration_seconds_estimate": duration_s if duration_observed else None,
        }

    thermal = flag_summary("thermal_throttled")
    power = flag_summary("power_limited")

    def counter_delta_seconds(key: str) -> float | None:
        values = []
        for sample in samples:
            try:
                values.append(float(sample[key]))
            except (KeyError, TypeError, ValueError):
                continue
        if len(values) < 2 or values[-1] < values[0]:
            return None
        return (values[-1] - values[0]) / 1_000_000.0

    thermal_counter_s = counter_delta_seconds("sw_thermal_slowdown_counter_us")
    power_counter_s = counter_delta_seconds("sw_power_cap_counter_us")
    if thermal_counter_s is not None and thermal_counter_s > 0:
        thermal["any"] = True
    if power_counter_s is not None and power_counter_s > 0:
        power["any"] = True
    reason_masks = sorted({
        str(sample["throttle_reasons_active"])
        for sample in samples
        if sample.get("throttle_reasons_active") not in (None, "", "N/A", "[N/A]")
    })
    # Integrate the actual monotonic sample spacing. nvidia-smi itself takes
    # nonzero and variable time, so assuming exactly interval_s between calls
    # biases long runs. Legacy/synthetic samples without timestamps retain the
    # previous nominal-spacing estimate for backwards compatibility.
    energy_j = None
    energy_seconds = None
    energy_method = None
    if len(power_samples) >= 2:
        timestamps = [sampled_at for sampled_at, _power in power_samples]
        if all(value is not None for value in timestamps) and all(
                timestamps[index] > timestamps[index - 1]
                for index in range(1, len(timestamps))):
            energy_j = sum(
                0.5 * (power_samples[index - 1][1] + power_samples[index][1])
                * (timestamps[index] - timestamps[index - 1])
                for index in range(1, len(power_samples)))
            energy_seconds = timestamps[-1] - timestamps[0]
            energy_method = "trapezoidal_monotonic_samples"
        else:
            powers = [power for _sampled_at, power in power_samples]
            energy_seconds = (len(powers) - 1) * interval_s
            energy_j = statistics.mean(powers) * energy_seconds
            energy_method = "nominal_interval_fallback"
    return {
        "samples_collected": len(samples),
        "sm_clock_mhz_min": min(clocks) if clocks else None,
        "sm_clock_mhz_median": statistics.median(clocks) if clocks else None,
        "temperature_c_max": max(temps) if temps else None,
        "mean_power_draw_w": (statistics.mean(
            power for _sampled_at, power in power_samples) if power_samples else None),
        "energy_joules_estimate": energy_j,
        "energy_sampling_seconds": energy_seconds,
        "energy_estimation_method": energy_method,
        # None (not False) when no sample ever carried a known throttle
        # reading at all -- e.g. no nvidia-smi -- so "no throttle detected"
        # is never confused with "throttle status unknown for this cell".
        "any_throttle_during_measurement": any_throttle if any_known else None,
        "any_thermal_throttle_during_measurement": thermal["any"],
        "thermal_throttle_samples": thermal["active_samples"],
        "thermal_status_samples": thermal["observed_samples"],
        "thermal_throttle_sample_fraction": thermal["sample_fraction"],
        "thermal_throttle_duration_seconds_estimate": thermal["duration_seconds_estimate"],
        "thermal_throttle_counter_delta_seconds": thermal_counter_s,
        "any_power_limit_during_measurement": power["any"],
        "power_limit_samples": power["active_samples"],
        "power_status_samples": power["observed_samples"],
        "power_limit_sample_fraction": power["sample_fraction"],
        "power_limit_duration_seconds_estimate": power["duration_seconds_estimate"],
        "power_limit_counter_delta_seconds": power_counter_s,
        "clock_event_reason_masks_seen": reason_masks,
    }


class ThermalSampler:
    """Samples gpu_thermal_snapshot() on a background thread at roughly
    1 Hz for the lifetime of a ``with`` block, wrapping one cell's entire
    timed portion. A start/end snapshot pair (what cool_down() and each
    row's own gpu_thermal field already capture) cannot see a throttle
    that both starts and clears somewhere in between; this can.
    """

    def __init__(self, interval_s: float = 1.0):
        self._interval_s = interval_s
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sample = bounded.gpu_thermal_snapshot()
                sample["sampled_at_monotonic_s"] = time.perf_counter()
                self._samples.append(sample)
            except Exception:  # a monitoring thread must never take the cell down
                pass
            self._stop.wait(self._interval_s)

    def __enter__(self) -> "ThermalSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def summary(self) -> dict:
        return thermal_monitor_summary(self._samples, interval_s=self._interval_s)


def run_cell(config: dict) -> dict:
    bounded.MODEL = config["model"]
    bounded.STORE = config["store"]
    bounded.COOLDOWN_SECONDS = config["cooldown_seconds"]
    bounded.COOLDOWN_MAX_TEMPERATURE_C = config["cooldown_max_temp_c"]

    method_id = config["method_id"]
    if config.get("method_overrides") is not None:
        # This worker runs in a fresh subprocess and does not inherit the
        # orchestrator's own METHODS dict, so a method registered there
        # only at runtime (run_paper_comparison.budget_method_variants()'s
        # exact-<N>gb/accelerate-<N>gb entries, for instance) would not
        # exist here at all under a plain METHODS[method_id] lookup. The
        # orchestrator sends the resolved spec directly instead, for every
        # method, so this worker never has to assume its own METHODS
        # matches the process that dispatched it.
        overrides = dict(config["method_overrides"])
        if config.get("dfloat11_model") and config.get("method_kind") == "dfloat11":
            overrides["model_id"] = config["dfloat11_model"]
        method = bounded.Method(
            id=method_id, title=config.get("method_title", method_id),
            kind=config["method_kind"], overrides=overrides,
            exactness=config.get("method_exactness", "unknown"),
            estimated_s_per_token=0.0)
    else:
        # Fallback for a caller that only sends method_id (this module's
        # own unit tests do): resolve locally against this process's own
        # static METHODS, same as before method_overrides existed.
        # METHODS["dfloat11"/"dfloat11-gpu-resident"].overrides["model_id"]
        # is baked in at module-import time from DFLOAT11_MODEL's value
        # *then* -- run_dfloat11's own
        # `method.overrides.get("model_id", DFLOAT11_MODEL)` fallback is
        # therefore dead code, since the key is always already present. A
        # --dfloat11-model override has to rewrite the overrides dict
        # directly, not the module-level default, or it is silently
        # ineffective.
        if config.get("dfloat11_model"):
            for candidate_id in ("dfloat11", "dfloat11-gpu-resident"):
                if candidate_id in METHODS:
                    METHODS[candidate_id] = dataclasses.replace(
                        METHODS[candidate_id],
                        overrides={**METHODS[candidate_id].overrides,
                                  "model_id": config["dfloat11_model"]})
        method = METHODS[method_id]

    n_tokens = config["n_tokens"]
    block = config["block"]
    warmup_tokens = config["warmup_tokens"]
    deadline = time.perf_counter() + config["seconds_remaining"]

    tokenizer = load_tokenizer(config["model"])
    all_cases = prompt_cases(config.get("prompt_suite") or "evaluation")
    if config.get("case_ids"):
        by_id = {case.id: case for case in all_cases}
        cases = tuple(by_id[case_id] for case_id in config["case_ids"])
    else:
        cases = all_cases
    rendered = render_cases(tokenizer, cases)
    for item in rendered:
        item["tokenizer"] = tokenizer

    draft_model = None
    result: dict
    with ThermalSampler() as sampler:
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
            elif method.kind == "deepspeed":
                rows, metadata = run_deepspeed_zero_inference(
                    method, rendered, n_tokens, deadline, None,
                    repeats=1, repeat_offset=block, warmup_tokens=warmup_tokens)
            else:
                # Keyed on the method's own draft_mode, not a literal
                # method_id -- a hardcoded "== 'spec-fixed'" check here
                # would silently skip loading a draft model for every other
                # speculative method (spec-k2, spec-k4, spec-k16, ...),
                # breaking them without a loud error since generate_adaptive
                # only fails once it is actually called with draft_model=None.
                if method.overrides.get("draft_mode") == "model":
                    from afterimage.runtime.streaming_engine import load_draft_model
                    log("  loading resident draft model %s" % config["draft_model"])
                    draft_model = load_draft_model(config["draft_model"], device="cuda")
                rows, metadata = run_afterimage(
                    method, rendered, n_tokens, deadline,
                    draft_model=draft_model,
                    burn_in_rendered=rendered[:1] if warmup_tokens > 0 else None,
                    burn_in_tokens=warmup_tokens,
                    rows_checkpoint=None, repeats=1, repeat_offset=block)
            result = {"rows": rows, "metadata": metadata,
                      "peak_host_rss_bytes": _peak_rss_bytes(),
                      "error": None, "traceback": None}
        except Exception as exc:
            metadata = {"capacity_failure": True} if is_capacity_failure(exc) else {}
            result = {"rows": [], "metadata": metadata,
                      "peak_host_rss_bytes": _peak_rss_bytes(),
                      "error": repr(exc), "traceback": traceback.format_exc()}
        finally:
            if draft_model is not None:
                del draft_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    monitoring = sampler.summary()
    result["thermal_monitoring"] = monitoring
    if config.get("require_thermally_clean") and result["error"] is None:
        thermal_state = monitoring.get("any_thermal_throttle_during_measurement")
        if thermal_state is not False:
            metadata = dict(result.get("metadata") or {})
            discarded_rows = result.get("rows") or []
            metadata["thermal_integrity_rejection"] = {
                "reason": ("thermal throttle observed" if thermal_state is True
                           else "thermal status was not observable"),
                "discarded_row_count": len(discarded_rows),
                "monitoring": monitoring,
                "discarded_rows": discarded_rows,
            }
            result["metadata"] = metadata
            result["rows"] = []
            result["error"] = (
                "ThermalIntegrityError(%r)" %
                metadata["thermal_integrity_rejection"]["reason"])
            result["traceback"] = None
    return result


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
