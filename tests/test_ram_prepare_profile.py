import dataclasses

import pytest

from afterimage.bench.ram_prepare import summarize_ram_prepare, validate_ram_profile
from afterimage.runtime.critical_path import TraceEvent


def event(identifier, duration, sweep=1):
    return TraceEvent(identifier, "transfer", "cuda", 0, duration,
                      tensor_key="head", metadata={"sweep": sweep})


def test_multiple_prompts_are_not_charged_as_one_use():
    events = [event("a", 3, 1), event("b", 2, 2), event("c", 4, 3), event("d", 3, 4)]
    result = summarize_ram_prepare(events)
    assert result["seconds_per_use"] == {"head": 3}
    assert result["samples_per_tensor"] == {"head": 4}


def test_split_spans_are_summed_only_within_a_sweep():
    assert summarize_ram_prepare([event("a", 2), event("b", 3), event("c", 7, 2)
                                  ])["seconds_per_use"] == {"head": 6}


@pytest.mark.parametrize("events", [[], [event("a", -1)], [event("a", float("nan"))],
                                   [dataclasses.replace(event("a", 1), metadata={})]])
def test_invalid_trace_rejected(events):
    with pytest.raises(ValueError):
        summarize_ram_prepare(events)


def profile():
    return dict(schema_version=1, aggregation="median_per_tensor_per_sweep",
                seconds_per_use={"head": 3}, manifest_sha256="model",
                vram_budget_gb=6, ram_budget_gb=8, vram_safety_margin_gb=2.2,
                case_ids=["calibration"], source_trace_sha256="trace")


def check(p):
    return validate_ram_profile(p, manifest_sha256="model", vram_gb=6,
                                ram_gb=8, safety_gb=2.2, evaluation_cases=["evaluation"])


def test_valid_profile():
    assert check(profile()) == {"head": 3}


@pytest.mark.parametrize("key,value", [
    ("schema_version", None), ("aggregation", "sum"), ("manifest_sha256", "other"),
    ("vram_budget_gb", 4), ("ram_budget_gb", 16), ("vram_safety_margin_gb", 1),
    ("case_ids", ["evaluation"]), ("source_trace_sha256", None),
    ("seconds_per_use", {"head": float("inf")}),
])
def test_unmatched_or_contaminated_profile_rejected(key, value):
    p = profile()
    p[key] = value
    with pytest.raises(ValueError):
        check(p)


def profile_v2():
    from afterimage.bench.ram_prepare import profile_execution_contract

    execution = profile_execution_contract(
        dict(reuse_decode_tables=True, decode_slice_elems=1024, io_prefetch_depth=2,
             storage_read_policy="per_blob"), warmup_tokens=1,
        source_sha256="execution-source", h2d_sha256="hardware")
    return dict(profile(), schema_version=2, execution_contract=execution,
                samples_seconds={"head": [2, 4]}, samples_per_tensor={"head": 2})


def check_strict(p):
    return validate_ram_profile(p, manifest_sha256="model", vram_gb=6, ram_gb=8,
                                safety_gb=2.2, evaluation_cases=["evaluation"], tensor_keys={"head"},
                                expected_execution=profile_v2()["execution_contract"])


def test_strict_profile_accepts_matching_execution_and_samples():
    assert check_strict(profile_v2()) == {"head": 3}


@pytest.mark.parametrize("field,value", [("reuse_decode_tables", False), ("warmup_tokens", 0),
                                       ("runtime_source_sha256", "changed"),
                                       ("h2d_artifact_sha256", "other-device")])
def test_strict_profile_rejects_wrong_execution(field, value):
    p = profile_v2()
    p["execution_contract"][field] = value
    with pytest.raises(ValueError, match="execution contract"):
        check_strict(p)


def test_historical_profile_cannot_calibrate_strict_paper_run():
    with pytest.raises(ValueError, match="execution contract"):
        check_strict(profile())


@pytest.mark.parametrize("field,value", [("samples_seconds", {"head": [2, 6]}),
                                       ("samples_per_tensor", {"head": 4}),
                                       ("seconds_per_use", {"other": 3})])
def test_strict_profile_rejects_inconsistent_samples_or_unknown_tensor(field, value):
    p = profile_v2()
    p[field] = value
    with pytest.raises(ValueError):
        check_strict(p)
