"""run_cell() dispatches exactly one (block, method) cell and reports a
JSON-serializable result -- this is the function a fresh subprocess runs.
Every external call (tokenizer load, case rendering, the four run_* method
functions, draft model loading) is mocked so these tests exercise the
dispatch/error-handling/override logic without CUDA, network, or a real
model.
"""
from __future__ import annotations

import json

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
