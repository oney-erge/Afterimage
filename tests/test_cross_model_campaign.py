import json

import pytest

from afterimage.bench.campaign import (
    load_campaign_config,
    render_campaign_markdown,
    summarize_result,
)
from scripts.run_cross_model_campaign import (
    benchmark_command,
    preserve_interrupted_partial,
)
from scripts.run_bounded_suite import add_comparisons


def test_campaign_config_requires_unique_models_and_stages(tmp_path):
    config = {
        "schema_version": 1,
        "models": [{
            "id": "small", "model_id": "example/model", "store": "/tmp/store",
            "benchmarks": [{"id": "smoke"}, {"id": "smoke"}],
        }],
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark ids"):
        load_campaign_config(path)


def test_summarize_bounded_result_preserves_failures_and_raw_artifact(tmp_path):
    artifact = tmp_path / "bounded.json"
    artifact.write_text(json.dumps({
        "status": "complete",
        "failures": [{"method": "airllm", "error": "unsupported"}],
        "methods": [
            {"method_id": "airllm", "error": "unsupported", "summary": {}},
            {"method_id": "exact-min", "rows": [{"output_token_ids": [1]}],
             "summary": {"completed_cases": 1, "seconds_per_token": 2.5,
                         "peak_vram_gb": 1.8, "expected_match_rate": 1.0}},
        ],
    }), encoding="utf-8")

    summary = summarize_result(artifact)

    assert summary["failure_count"] == 1
    assert summary["methods"][0]["status"] == "failed"
    assert summary["methods"][1]["seconds_per_token"] == 2.5
    assert summary["artifact"] == str(artifact)


def test_interim_markdown_distinguishes_running_from_measured_rows():
    campaign = {
        "status": "running",
        "models": [{
            "id": "small", "model_id": "example/model", "role": "small",
            "stages": [
                {"id": "compress", "status": "running", "log": "compress.log"},
                {"id": "smoke", "status": "passed", "artifact": "smoke.json",
                 "summary": {"methods": [{
                     "method": "exact-min", "seconds_per_token": 2.5,
                     "peak_vram_gb": 1.8, "completed_cases": 1,
                 }]}},
            ],
        }],
        "deferred_models": [],
        "excluded_hypothesis_families": [],
    }

    report = render_campaign_markdown(campaign)

    assert "| example/model | small | compress | running |" in report
    assert "| small | smoke | exact-min | 2.5000 | 1.8000 | 1 |" in report


def test_benchmark_command_passes_cross_model_identity_and_store(tmp_path):
    model = {
        "model_id": "example/model", "store": "/data/example-store",
        "hf_offload_dir": "/data/offload",
    }
    stage = {
        "runner": "regulated", "hypothesis": "H14", "blocks": 1,
        "max_new_tokens": 2, "time_budget_minutes": 10,
        "case_ids": ["fact-gold"],
    }

    command = benchmark_command(
        model, stage, tmp_path / "result.json",
        {"common": {"gpu_memory": "4GB", "cpu_memory": "12GB"}})

    assert command[command.index("--model") + 1] == "example/model"
    assert command[command.index("--store") + 1] == "/data/example-store"
    assert command[command.index("--case-ids") + 1] == "fact-gold"


def test_resume_preserves_interrupted_runner_checkpoint(tmp_path):
    artifact = tmp_path / "broad-l1.json"
    partial = tmp_path / "broad-l1.json.partial"
    partial.write_text('{"status":"running"}', encoding="utf-8")

    archived = preserve_interrupted_partial(artifact, stamp=123)

    assert archived == tmp_path / "broad-l1.json.interrupted-123.partial"
    assert archived.read_text(encoding="utf-8") == '{"status":"running"}'
    assert not partial.exists()


def test_token_exactness_is_separate_from_semantic_prefix_completion():
    result = {"methods": [
        {"method_id": "exact-min", "rows": [
            {"case_id": "retrieval", "output_token_ids": [10, 11],
             "expected_match": False}],
         "summary": {"seconds_per_token": 2.0, "peak_vram_gb": 1.0,
                     "expected_match_rate": 0.0}},
        {"method_id": "candidate", "rows": [
            {"case_id": "retrieval", "output_token_ids": [10, 11],
             "expected_match": False}],
         "summary": {"seconds_per_token": 1.0, "peak_vram_gb": 1.0,
                     "expected_match_rate": 0.0}},
    ]}

    add_comparisons(result)

    assert result["methods"][1]["summary"]["expected_match_rate"] == 0.0
    assert result["methods"][1]["summary"]["token_agreement_vs_exact_min"] == 1.0
