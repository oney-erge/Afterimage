"""Measured, per-use RAM preparation profiles for offline placement.

A trace can contain multiple prompts and token sweeps. Adding their durations
and charging that sum on EVERY future use exaggerates the cost. Sum split
transfer spans within one tensor use, then take the median across sweeps.
These spans must cover completed allocation/copy, not asynchronous issue time.
"""
from __future__ import annotations

import collections
import hashlib
import math
import pathlib
import statistics


def summarize_ram_prepare(events) -> dict:
    samples = collections.defaultdict(lambda: collections.defaultdict(float))
    for event in events:
        if event.kind != "transfer" or not event.tensor_key:
            continue
        sweep = event.metadata.get("sweep")
        if not isinstance(sweep, int) or isinstance(sweep, bool) or sweep < 1:
            raise ValueError("RAM transfer event needs a positive sweep identifier")
        duration = event.end_s - event.start_s
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("RAM preparation duration must be finite and nonnegative")
        samples[event.tensor_key][sweep] += duration
    if not samples:
        raise ValueError("trace contains no measured RAM transfer events")
    values = {key: list(by_sweep.values()) for key, by_sweep in samples.items()}
    return {
        "aggregation": "median_per_tensor_per_sweep",
        "seconds_per_use": {key: statistics.median(v) for key, v in values.items()},
        "samples_per_tensor": {key: len(v) for key, v in values.items()},
        "samples_seconds": values,
    }


def runtime_source_fingerprint(repo) -> str:
    """Execution and measurement source identity, not merely the git commit."""
    repo = pathlib.Path(repo)
    paths = sorted((repo / "afterimage").rglob("*.py")) + [
        repo / "scripts" / name for name in (
            "run_bounded_suite.py", "run_h65_paper_worker.py",
            "run_paper_comparison_worker.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(repo).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def profile_execution_contract(config: dict, *, warmup_tokens: int,
                               source_sha256: str, h2d_sha256: str) -> dict:
    """Measured costs are conditional on the decoder and measurement lifecycle."""
    return {
        "reuse_decode_tables": config["reuse_decode_tables"],
        "decode_slice_elems": config["decode_slice_elems"],
        "io_prefetch_depth": config["io_prefetch_depth"],
        "storage_read_policy": config["storage_read_policy"],
        "warmup_tokens": warmup_tokens,
        "cache_policy": "cold_page_cache_per_request",
        "allocator_policy": "empty_cache_before_timing",
        "runtime_source_sha256": source_sha256,
        "h2d_artifact_sha256": h2d_sha256,
    }


def validate_ram_profile(profile: dict, *, manifest_sha256: str,
                         vram_gb: float, ram_gb: float, safety_gb: float,
                         evaluation_cases, expected_execution: dict | None = None,
                         tensor_keys=None) -> dict[str, float]:
    """Validate units and provenance; strict callers also require execution v2.

    Schema 1 remains readable for historical diagnostics when no execution
    contract is requested. It must not silently calibrate a new paper run.
    """
    if not isinstance(profile, dict):
        raise ValueError("RAM profile must be an object")
    version = profile.get("schema_version")
    if version not in (1, 2) or "seconds_per_use" not in profile:
        raise ValueError("RAM profile needs schema_version=1 or 2 and seconds_per_use; "
                         "legacy unlabelled maps are not calibrated evidence")
    if expected_execution is not None:
        if version != 2 or profile.get("execution_contract") != expected_execution:
            raise ValueError("RAM profile execution contract mismatch: decoder, warm-up, "
                             "source and H2D artifact must match; recalibrate explicitly")
    if profile.get("aggregation") != "median_per_tensor_per_sweep":
        raise ValueError("RAM profile must aggregate per use, not across prompts")
    if profile.get("manifest_sha256") != manifest_sha256:
        raise ValueError("RAM profile manifest does not match this model store")
    for key, expected in (("vram_budget_gb", vram_gb), ("ram_budget_gb", ram_gb),
                          ("vram_safety_margin_gb", safety_gb)):
        value = profile.get(key)
        if (not isinstance(value, (float, int)) or isinstance(value, bool)
                or not math.isfinite(value) or not math.isclose(value, expected)):
            raise ValueError("RAM profile budget mismatch: " + key)
    cases = profile.get("case_ids")
    if not isinstance(cases, list) or not cases or not all(isinstance(c, str) for c in cases):
        raise ValueError("RAM profile needs calibration case_ids")
    if set(cases) & set(evaluation_cases):
        raise ValueError("RAM profile calibration overlaps evaluation cases")
    if not profile.get("source_trace_sha256"):
        raise ValueError("RAM profile needs source trace provenance")
    costs = profile["seconds_per_use"]
    if not isinstance(costs, dict) or not costs:
        raise ValueError("RAM profile needs nonempty per-use measurements")
    for key, value in costs.items():
        if tensor_keys is not None and key not in tensor_keys:
            raise ValueError("unknown tensor in RAM profile: " + str(key))
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError("invalid RAM preparation cost: " + str(key))
        if version == 2:
            samples = profile.get("samples_seconds", {}).get(key)
            count = profile.get("samples_per_tensor", {}).get(key)
            if (not isinstance(samples, list) or not samples or count != len(samples)
                    or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                           or not math.isfinite(v) or v < 0 for v in samples)
                    or not math.isclose(statistics.median(samples), value)):
                raise ValueError("RAM per-use cost does not match its samples: " + str(key))
    return dict(costs)
