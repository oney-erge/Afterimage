import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
import torch
from safetensors.torch import save_file

from afterimage.runtime.control import JobControl
from afterimage.server.app import (
    ExperimentRunRequest, _specialized_experiment, app,
)


def test_experiment_registry_and_ui_are_exposed():
    client = TestClient(app)
    payload = client.get("/api/experiments").json()
    assert len(payload["hypotheses"]) == 19
    assert {row["id"] for row in payload["hypotheses"]} >= {
        "h0-joint-oracle-gap", "h8-model-based-rl",
        "h9-ram-overlay-head", "h10-replay-cem",
        "h11-neural-utility-spec", "h12-bayesian-prefetch",
        "h13-qubo-residency", "h14-coalesced-storage",
        "h15-extent-qubo-residency", "h16-spec-critical-path",
        "h17-tensor-extents", "h18-rollback-cached-spec"}
    page = client.get("/").text
    assert "Research" in page
    assert 'id="chat-model"' in page
    assert 'id="chat-stop"' in page
    assert 'src="/static/js/app.js"' in page
    assert "Compare execution profiles" in page
    assert "Research Lab" not in page
    definition = client.get("/api/experiments/h12-bayesian-prefetch").json()
    assert definition["protocol"]["id"] == "adaptive-prefetch"


def test_h2_refuses_an_unfrozen_or_missing_calibration_state():
    client = TestClient(app)
    response = client.post("/api/experiments/h2-hazard-cost/runs", json={
        "model_id": "does/not/exist", "draft_model_id": "draft/model",
        "repeats": 5, "max_new_tokens": 128,
    })
    assert response.status_code == 400
    assert "spec_policy_state" in response.text


def test_unknown_hypothesis_is_404():
    assert TestClient(app).get("/api/experiments/not-real").status_code == 404


def test_h0_and_h3_specialized_runners_produce_verdicts():
    oracle_rows = []
    for semantic, system in (("a", "x"), ("b", "y")):
        for profile, reward in (("base", 1.0), ("fast", 2.0)):
            oracle_rows.append({
                "profile": profile, "semantic_bucket": semantic,
                "system_bucket": system, "committed_tokens_per_second": reward,
            })
    h0 = _specialized_experiment(
        "h0-joint-oracle-gap",
        ExperimentRunRequest(inputs={"result_dataset": oracle_rows}), JobControl())
    assert h0.status == "done"

    replay = [{"context": [1.0], "rewards": {"base": 1.0, "fast": 2.0}}
              for _ in range(30)]
    h3 = _specialized_experiment(
        "h3-contextual-bandit",
        ExperimentRunRequest(inputs={"calibration_dataset": replay[:5],
                                     "result_dataset": replay[5:]}), JobControl())
    assert h3.summary["oracle_fraction"] >= 0.95


def test_h6_and_h8_specialized_runners_enforce_their_gates():
    h6 = _specialized_experiment(
        "h6-representations", ExperimentRunRequest(inputs={
            "representation_options": [
                {"tensor_key": "w", "name": "disk", "prepare_s": 2.0},
                {"tensor_key": "w", "name": "ram", "ram_bytes": 16 << 20,
                 "prepare_s": 0.5},
            ],
            "ram_budget_bytes": 16 << 20, "uniform_prepare_s": 2.0,
        }), JobControl())
    assert h6.verdict == "favored"

    h8 = _specialized_experiment(
        "h8-model-based-rl", ExperimentRunRequest(inputs={
            "trace_dataset": [
                {"actual_rewards": {"base": 1.0, "fast": 1.3},
                 "predicted_rewards": {"base": 1.0, "fast": 1.3},
                 "baseline_profile": "base"},
                {"actual_rewards": {"base": 1.0, "fast": 1.2},
                 "predicted_rewards": {"base": 1.0, "fast": 1.2},
                 "baseline_profile": "base"},
            ],
        }), JobControl())
    assert h8.verdict == "favored"


def test_h7_uses_acyclic_bases_and_total_storage(tmp_path):
    tensor_path = tmp_path / "experts.safetensors"
    save_file({"base": torch.zeros(64, dtype=torch.bfloat16),
               "target": torch.zeros(64, dtype=torch.bfloat16)}, str(tensor_path))
    req = ExperimentRunRequest(inputs={
        "expert_tensors": [
            {"id": "base", "path": str(tensor_path), "tensor_key": "base"},
            {"id": "target", "path": str(tensor_path), "tensor_key": "target"},
        ],
        "reference_bases": ["base"],
        "independent_compressed_bytes": {"base": 128, "target": 128},
    })
    run = _specialized_experiment("h7-xor-reference", req, JobControl())
    assert run.summary["candidate_bytes"] <= run.summary["independent_bytes"]
    assert run.summary["audit"]["base"] is None
