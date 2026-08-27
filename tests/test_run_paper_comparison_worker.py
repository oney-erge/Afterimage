"""run_cell() dispatches exactly one (block, method) cell and reports a
JSON-serializable result -- this is the function a fresh subprocess runs.
Every external call (tokenizer load, case rendering, the four run_* method
functions, draft model loading) is mocked so these tests exercise the
dispatch/error-handling/override logic without CUDA, network, or a real
model.
"""
from __future__ import annotations

import json
import time

import pytest

from scripts import run_paper_comparison_worker as worker
from scripts.run_bounded_suite import METHODS


class _FakeCase:
    def __init__(self, case_id):
        self.id = case_id


@pytest.fixture(autouse=True)
def _stub_rendering(monkeypatch):
    """Every test needs case rendering and tokenizer loading stubbed; doing
    it as an autouse fixture keeps each test focused on what it actually
    asserts."""
    monkeypatch.setattr(worker, "load_tokenizer", lambda model_id: "FAKE_TOKENIZER")
    monkeypatch.setattr(worker, "prompt_cases",
                        lambda kind: (_FakeCase("a"), _FakeCase("b")))

    def fake_render_cases(tokenizer, cases):
        return [{"case": case, "prompt": "prompt-%s" % case.id, "input_tokens": 5}
                for case in cases]

    monkeypatch.setattr(worker, "render_cases", fake_render_cases)


def _base_config(**overrides):
    config = {
        "method_id": "airllm", "model": "Qwen/Qwen3-14B",
        "dfloat11_model": None, "draft_model": "Qwen/Qwen3-0.6B",
        "store": "/tmp/store", "n_tokens": 4, "block": 2,
        "warmup_tokens": 8, "cooldown_seconds": 5.0, "cooldown_max_temp_c": 70.0,
        "seconds_remaining": 60.0, "case_ids": None,
    }
    config.update(overrides)
    return config


def test_dispatches_airllm_with_repeat_offset_and_warmup(monkeypatch):
    captured = {}

    def fake_run_airllm(method, rendered, n_tokens, deadline, checkpoint_cb,
                        repeats, repeat_offset, warmup_tokens):
        captured.update(method_id=method.id, n_tokens=n_tokens, repeats=repeats,
                        repeat_offset=repeat_offset, warmup_tokens=warmup_tokens,
                        n_cases=len(rendered))
        return [{"case_id": "a"}], {"initialization_seconds": 1.0}

    monkeypatch.setattr(worker, "run_airllm", fake_run_airllm)
    result = worker.run_cell(_base_config(method_id="airllm"))

    assert result["error"] is None
    assert result["rows"] == [{"case_id": "a"}]
    assert captured == {"method_id": "airllm", "n_tokens": 4, "repeats": 1,
                        "repeat_offset": 2, "warmup_tokens": 8, "n_cases": 2}


def test_dispatches_accelerate(monkeypatch):
    calls = []

    def fake_run_accelerate(*args, **kwargs):
        calls.append((args, kwargs))
        return [], {}

    monkeypatch.setattr(worker, "run_accelerate", fake_run_accelerate)
    result = worker.run_cell(_base_config(method_id="accelerate"))
    assert result["error"] is None
    assert len(calls) == 1


def test_dispatches_dfloat11(monkeypatch):
    calls = []

    def fake_run_dfloat11(*args, **kwargs):
        calls.append((args, kwargs))
        return [], {}

    monkeypatch.setattr(worker, "run_dfloat11", fake_run_dfloat11)
    result = worker.run_cell(_base_config(method_id="dfloat11"))
    assert result["error"] is None
    assert len(calls) == 1


def test_dfloat11_model_override_actually_rewrites_the_registered_method(monkeypatch):
    """Regression test for a real bug: run_dfloat11's own
    `method.overrides.get("model_id", DFLOAT11_MODEL)` fallback is dead
    code, because METHODS["dfloat11"].overrides["model_id"] is always
    already present (baked in at import time). Setting the module-level
    DFLOAT11_MODEL global does nothing to an already-built Method. The
    worker must rewrite METHODS[...].overrides directly instead."""
    monkeypatch.setattr(worker, "run_dfloat11", lambda *a, **k: ([], {}))
    original = METHODS["dfloat11"]
    try:
        worker.run_cell(_base_config(
            method_id="dfloat11", dfloat11_model="Some/Other-DF11-Repo"))
        assert METHODS["dfloat11"].overrides["model_id"] == "Some/Other-DF11-Repo"
        assert METHODS["dfloat11-gpu-resident"].overrides["model_id"] == (
            "Some/Other-DF11-Repo")
    finally:
        METHODS["dfloat11"] = original


def test_dispatches_afterimage_methods_without_loading_a_draft_model(monkeypatch):
    """exact-min has no speculative draft step; the worker must not import
    or call load_draft_model for it."""
    captured = {}

    def fake_run_afterimage(method, rendered, n_tokens, deadline, draft_model,
                            burn_in_rendered, burn_in_tokens, rows_checkpoint,
                            repeats, repeat_offset):
        captured["draft_model"] = draft_model
        captured["burn_in_tokens"] = burn_in_tokens
        return [], {}

    monkeypatch.setattr(worker, "run_afterimage", fake_run_afterimage)
    result = worker.run_cell(_base_config(method_id="exact-min"))
    assert result["error"] is None
    assert captured["draft_model"] is None
    assert captured["burn_in_tokens"] == 8


def test_spec_fixed_loads_its_own_fresh_draft_model(monkeypatch):
    """The whole point of subprocess isolation: spec-fixed's draft model is
    loaded inside THIS call, not inherited from a previous cell, and dies
    with this process rather than staying resident for the rest of the
    campaign."""
    import afterimage.runtime.streaming_engine as streaming_engine

    loaded = []
    monkeypatch.setattr(streaming_engine, "load_draft_model",
                        lambda model_id, device: loaded.append((model_id, device))
                        or "FAKE_DRAFT_MODEL")

    captured = {}

    def fake_run_afterimage(method, rendered, n_tokens, deadline, draft_model,
                            burn_in_rendered, burn_in_tokens, rows_checkpoint,
                            repeats, repeat_offset):
        captured["draft_model"] = draft_model
        return [], {}

    monkeypatch.setattr(worker, "run_afterimage", fake_run_afterimage)
    result = worker.run_cell(_base_config(method_id="spec-fixed",
                                          draft_model="Qwen/Qwen3-0.6B"))
    assert result["error"] is None
    assert loaded == [("Qwen/Qwen3-0.6B", "cuda")]
    assert captured["draft_model"] == "FAKE_DRAFT_MODEL"


@pytest.mark.parametrize("method_id", ["spec-k2", "spec-k4", "spec-fixed", "spec-k16"])
def test_every_fixed_k_speculative_method_loads_a_draft_model_not_just_spec_fixed(
        monkeypatch, method_id):
    """Regression test for a real bug: the dispatch used to check
    `method_id == "spec-fixed"` literally, so spec-k2/spec-k4/spec-k16
    (added for the k-ablation series) would silently run with
    draft_model=None despite declaring draft_mode="model" -- a config that
    generate_adaptive only fails on once actually called, not loudly at
    dispatch time. The check must be keyed on the method's own
    draft_mode, not its literal id string."""
    import afterimage.runtime.streaming_engine as streaming_engine

    loaded = []
    monkeypatch.setattr(streaming_engine, "load_draft_model",
                        lambda model_id, device: loaded.append((model_id, device))
                        or "FAKE_DRAFT_MODEL")
    captured = {}

    def fake_run_afterimage(method, rendered, n_tokens, deadline, draft_model,
                            burn_in_rendered, burn_in_tokens, rows_checkpoint,
                            repeats, repeat_offset):
        captured["draft_model"] = draft_model
        return [], {}

    monkeypatch.setattr(worker, "run_afterimage", fake_run_afterimage)
    result = worker.run_cell(_base_config(method_id=method_id,
                                          draft_model="Qwen/Qwen3-0.6B"))
    assert result["error"] is None
    assert loaded == [("Qwen/Qwen3-0.6B", "cuda")]
    assert captured["draft_model"] == "FAKE_DRAFT_MODEL"


def test_spec_k0_the_no_speculation_control_does_not_load_a_draft_model(monkeypatch):
    import afterimage.runtime.streaming_engine as streaming_engine

    loaded = []
    monkeypatch.setattr(streaming_engine, "load_draft_model",
                        lambda model_id, device: loaded.append((model_id, device)))
    captured = {}

    def fake_run_afterimage(method, rendered, n_tokens, deadline, draft_model,
                            burn_in_rendered, burn_in_tokens, rows_checkpoint,
                            repeats, repeat_offset):
        captured["draft_model"] = draft_model
        return [], {}

    monkeypatch.setattr(worker, "run_afterimage", fake_run_afterimage)
    result = worker.run_cell(_base_config(method_id="spec-k0"))
    assert result["error"] is None
    assert loaded == []
    assert captured["draft_model"] is None


class TestMethodSpecPassedThroughConfig:
    """When the orchestrator sends method_overrides/method_kind/etc.
    directly (always, as of the cross-process fix), the worker must build
    its Method from those fields instead of looking method_id up in its
    own METHODS -- the whole point being that a method registered only at
    runtime in the orchestrator's process (budget_method_variants()'s
    exact-<N>gb/accelerate-<N>gb entries) does not exist in this fresh
    subprocess's copy of METHODS at all.
    """

    def test_builds_a_method_for_an_id_absent_from_this_processs_methods(
            self, monkeypatch):
        assert "exact-2gb" not in METHODS  # the exact gap this closes
        captured = {}

        def fake_run_afterimage(method, rendered, n_tokens, deadline, draft_model,
                                burn_in_rendered, burn_in_tokens, rows_checkpoint,
                                repeats, repeat_offset):
            captured["method"] = method
            return [], {}

        monkeypatch.setattr(worker, "run_afterimage", fake_run_afterimage)
        result = worker.run_cell(_base_config(
            method_id="exact-2gb", method_title="Afterimage exact streaming at 2 GB",
            method_kind="afterimage", method_exactness="reference_execution_equivalent",
            method_overrides={"vram_budget_gb": 2.0, "decode_slice_elems": 1 << 22}))
        assert result["error"] is None
        assert captured["method"].id == "exact-2gb"
        assert captured["method"].overrides["vram_budget_gb"] == 2.0

    def test_dfloat11_model_override_still_applies_via_the_config_path(self, monkeypatch):
        captured = {}

        def fake_run_dfloat11(method, rendered, n_tokens, deadline, checkpoint_cb,
                              repeats, repeat_offset, warmup_tokens):
            captured["model_id"] = method.overrides["model_id"]
            return [], {}

        monkeypatch.setattr(worker, "run_dfloat11", fake_run_dfloat11)
        result = worker.run_cell(_base_config(
            method_id="dfloat11", method_title="DFloat11", method_kind="dfloat11",
            method_exactness="reference_greedy",
            method_overrides={"model_id": "DFloat11/Qwen3-14B-DF11", "cpu_offload": True},
            dfloat11_model="Some/Other-DF11-Repo"))
        assert result["error"] is None
        assert captured["model_id"] == "Some/Other-DF11-Repo"

    def test_a_budget_variant_of_spec_fixed_still_loads_a_draft_model(self, monkeypatch):
        """draft_mode routing (the spec-fixed / spec-k* fix) must also work
        for a method built entirely from config, not just for statically
        registered ones."""
        import afterimage.runtime.streaming_engine as streaming_engine

        loaded = []
        monkeypatch.setattr(streaming_engine, "load_draft_model",
                            lambda model_id, device: loaded.append((model_id, device))
                            or "FAKE_DRAFT_MODEL")
        captured = {}

        def fake_run_afterimage(method, rendered, n_tokens, deadline, draft_model,
                                burn_in_rendered, burn_in_tokens, rows_checkpoint,
                                repeats, repeat_offset):
            captured["draft_model"] = draft_model
            return [], {}

        monkeypatch.setattr(worker, "run_afterimage", fake_run_afterimage)
        result = worker.run_cell(_base_config(
            method_id="spec-k2-2gb", method_title="spec-k2 at 2 GB",
            method_kind="afterimage", method_exactness="greedy_token_exact_at_temperature_zero",
            method_overrides={"vram_budget_gb": 2.0, "draft_mode": "model", "spec_k": 2},
            draft_model="Qwen/Qwen3-0.6B"))
        assert result["error"] is None
        assert loaded == [("Qwen/Qwen3-0.6B", "cuda")]
        assert captured["draft_model"] == "FAKE_DRAFT_MODEL"


def test_prompt_suite_selects_which_case_pool_case_ids_are_looked_up_in(monkeypatch):
    """Regression test for a real bug: the worker used to hardcode
    prompt_cases("evaluation") regardless of which suite the orchestrator
    actually selected, so a paper_generation case_id (e.g.
    "explain-bicycle-balance") would KeyError -- it does not exist in the
    "evaluation" split at all. Asserts on the split argument the worker
    actually requests, rather than depending on real prompt_suite.py
    content, since the autouse fixture above already stubs prompt_cases
    for every other test in this file."""
    requested_splits = []

    def fake_prompt_cases(split):
        requested_splits.append(split)
        return (_FakeCase("a"), _FakeCase("b"))

    monkeypatch.setattr(worker, "prompt_cases", fake_prompt_cases)
    monkeypatch.setattr(worker, "run_airllm", lambda *a, **k: ([], {}))
    result = worker.run_cell(_base_config(
        method_id="airllm", prompt_suite="paper_generation"))
    assert result["error"] is None
    assert requested_splits == ["paper_generation"]


def test_missing_prompt_suite_key_defaults_to_evaluation(monkeypatch):
    """Older cell configs (and this file's own _base_config()) have no
    prompt_suite key at all; that must behave exactly as it always did,
    not raise."""
    captured = {}

    def fake_run_airllm(method, rendered, *a, **k):
        captured["case_ids"] = [item["case"].id for item in rendered]
        return [], {}

    monkeypatch.setattr(worker, "run_airllm", fake_run_airllm)
    config = _base_config(method_id="airllm")
    assert "prompt_suite" not in config
    result = worker.run_cell(config)
    assert result["error"] is None
    assert captured["case_ids"]  # the default evaluation split, non-empty


def test_case_ids_filters_to_the_requested_subset(monkeypatch):
    captured = {}

    def fake_run_airllm(method, rendered, *a, **k):
        captured["case_ids"] = [item["case"].id for item in rendered]
        return [], {}

    monkeypatch.setattr(worker, "run_airllm", fake_run_airllm)
    result = worker.run_cell(_base_config(method_id="airllm", case_ids=["b"]))
    assert result["error"] is None
    assert captured["case_ids"] == ["b"]


def test_an_exception_inside_the_method_call_is_captured_not_raised(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated CUDA OOM")

    monkeypatch.setattr(worker, "run_dfloat11", boom)
    result = worker.run_cell(_base_config(method_id="dfloat11"))
    assert result["error"] == "RuntimeError('simulated CUDA OOM')"
    assert "simulated CUDA OOM" in result["traceback"]
    assert result["rows"] == []


class TestThermalMonitorSummary:
    def test_reports_min_clock_median_clock_and_max_temperature(self):
        samples = [
            {"sm_clock_mhz": "1890", "temperature_c": "60", "throttled": False},
            {"sm_clock_mhz": "780", "temperature_c": "62", "throttled": True},
            {"sm_clock_mhz": "1200", "temperature_c": "75", "throttled": False},
        ]
        summary = worker.thermal_monitor_summary(samples)
        assert summary["samples_collected"] == 3
        assert summary["sm_clock_mhz_min"] == 780.0
        assert summary["sm_clock_mhz_median"] == 1200.0
        assert summary["temperature_c_max"] == 75.0
        assert summary["any_throttle_during_measurement"] is True

    def test_a_throttle_that_clears_before_the_last_sample_is_still_caught(self):
        """The exact gap continuous sampling exists to close: a single
        end-of-cell snapshot would see only the clean final sample and
        miss the throttle that happened in between."""
        samples = [
            {"sm_clock_mhz": "1890", "temperature_c": "58", "throttled": False},
            {"sm_clock_mhz": "780", "temperature_c": "61", "throttled": True},
            {"sm_clock_mhz": "1890", "temperature_c": "59", "throttled": False},
        ]
        summary = worker.thermal_monitor_summary(samples)
        assert summary["any_throttle_during_measurement"] is True

    def test_no_throttle_at_all_reports_false_not_none(self):
        samples = [{"sm_clock_mhz": "1890", "temperature_c": "60", "throttled": False}]
        summary = worker.thermal_monitor_summary(samples)
        assert summary["any_throttle_during_measurement"] is False

    def test_unknown_throttle_status_is_none_not_a_false_all_clear(self):
        """No nvidia-smi / non-NVIDIA host: every sample's throttled field
        is None. That must read as "unknown", never as a confident "no
        throttle happened", matching is_throttled()'s own contract in
        run_bounded_suite.py."""
        samples = [{"sm_clock_mhz": None, "temperature_c": None, "throttled": None}]
        summary = worker.thermal_monitor_summary(samples)
        assert summary["any_throttle_during_measurement"] is None

    def test_empty_sample_list_degrades_cleanly(self):
        summary = worker.thermal_monitor_summary([])
        assert summary == {
            "samples_collected": 0, "sm_clock_mhz_min": None,
            "sm_clock_mhz_median": None, "temperature_c_max": None,
            "mean_power_draw_w": None, "energy_joules_estimate": None,
            "any_throttle_during_measurement": None,
        }


class TestThermalSampler:
    def test_collects_multiple_samples_over_its_lifetime(self, monkeypatch):
        calls = {"n": 0}

        def fake_snapshot():
            calls["n"] += 1
            return {"sm_clock_mhz": "1890", "temperature_c": "60", "throttled": False}

        monkeypatch.setattr(worker.bounded, "gpu_thermal_snapshot", fake_snapshot)
        with worker.ThermalSampler(interval_s=0.01) as sampler:
            time.sleep(0.1)
        summary = sampler.summary()
        assert summary["samples_collected"] >= 2

    def test_a_snapshot_exception_does_not_kill_the_sampling_thread(self, monkeypatch):
        """A monitoring thread failing must never take the timed cell down
        with it."""
        state = {"calls": 0}

        def flaky_snapshot():
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("nvidia-smi transient failure")
            return {"sm_clock_mhz": "1890", "temperature_c": "60", "throttled": False}

        monkeypatch.setattr(worker.bounded, "gpu_thermal_snapshot", flaky_snapshot)
        with worker.ThermalSampler(interval_s=0.01) as sampler:
            time.sleep(0.1)
        summary = sampler.summary()
        assert summary["samples_collected"] >= 1  # survived the first exception


def test_peak_rss_bytes_is_an_int_or_none_never_raises():
    value = worker._peak_rss_bytes()
    assert value is None or (isinstance(value, int) and value >= 0)


def test_main_writes_the_result_json_and_exits_nonzero_on_error(tmp_path, monkeypatch):
    config_path = tmp_path / "cell.json"
    out_path = tmp_path / "result.json"
    config_path.write_text(
        json.dumps(_base_config(method_id="dfloat11")), encoding="utf-8")

    monkeypatch.setattr(worker, "run_dfloat11",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
    monkeypatch.setattr(
        "sys.argv",
        ["run_paper_comparison_worker.py", "--config", str(config_path),
         "--out", str(out_path)])
    exit_code = worker.main()
    assert exit_code == 1
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["error"] == "RuntimeError('nope')"
