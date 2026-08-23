import json
import types

from afterimage.cli import (
    _estimate_download_bytes, build_parser, cmd_experiments,
    cmd_optimize_residency, cmd_profile_trace, cmd_test_plan,
)
from afterimage.runtime.critical_path import CriticalPathProfile, TraceRecorder


def test_download_estimate_uses_transformers_index_not_duplicate_export(
        tmp_path, monkeypatch):
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors",
        "b": "model-00002-of-00002.safetensors",
    }}), encoding="utf-8")
    siblings = [
        types.SimpleNamespace(rfilename="model.safetensors.index.json", size=100),
        types.SimpleNamespace(
            rfilename="model-00001-of-00002.safetensors", size=11),
        types.SimpleNamespace(
            rfilename="model-00002-of-00002.safetensors", size=13),
        types.SimpleNamespace(rfilename="consolidated.safetensors", size=24),
    ]
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info",
        lambda self, model_id, files_metadata: types.SimpleNamespace(
            siblings=siblings))
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda model_id, filename: str(index_path))

    assert _estimate_download_bytes("fake/duplicated") == 24


def test_research_cli_parses_new_commands():
    parser = build_parser()
    assert parser.parse_args(["research", "experiments", "--json"]).func is cmd_experiments
    plan_args = parser.parse_args(
        ["research", "test-plan", "h12-bayesian-prefetch", "--json"])
    assert plan_args.func is cmd_test_plan
    args = parser.parse_args(
        ["research", "profile-trace", "a.json", "--out", "p.json"])
    assert args.func is cmd_profile_trace
    args = parser.parse_args([
        "research", "optimize-residency", "trace.json", "--manifest", "manifest.json",
        "--out", "plan.json", "--vram-budget-gb", "4",
        "--search-method", "qubo"])
    assert args.func is cmd_optimize_residency
    assert args.search_method == "qubo"
    extent_args = parser.parse_args([
        "research", "optimize-residency", "trace.json", "--manifest", "manifest.json",
        "--out", "plan.json", "--vram-budget-gb", "4",
        "--search-method", "extent-qubo"])
    assert extent_args.search_method == "extent-qubo"
    assert parser.parse_args(
        ["research", "pin-preflight", "--static-only"]).static_only


def test_profile_trace_cli_builds_profile(tmp_path):
    recorder = TraceRecorder()
    read = recorder.record("read", "disk", 0.0, 2.0, tensor_key="w")
    recorder.record("decode", "cuda", 2.0, 3.0, tensor_key="w",
                    dependencies=(read,))
    trace_path = tmp_path / "trace.json"
    profile_path = tmp_path / "profile.json"
    recorder.save(trace_path)
    args = build_parser().parse_args(
        ["research", "profile-trace", str(trace_path), "--out", str(profile_path)])
    assert args.func(args) == 0
    profile = CriticalPathProfile.load(profile_path)
    assert profile.tensors["w"].critical_s == 3.0


def test_experiments_cli_emits_machine_readable_registry(capsys):
    args = build_parser().parse_args(["research", "experiments", "--json"])
    assert args.func(args) == 0
    assert len(json.loads(capsys.readouterr().out)["hypotheses"]) == 19


def test_test_plan_cli_emits_hypothesis_specific_stages(capsys):
    args = build_parser().parse_args([
        "research", "test-plan", "h13-qubo-residency", "--json"])
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"]["id"] == "placement-latency"
    assert payload["protocol"]["stages"][-1]["confirmatory"] is True


def test_test_plan_cli_accepts_short_case_insensitive_hypothesis_id(capsys):
    args = build_parser().parse_args(["research", "test-plan", "H12", "--json"])
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hypothesis"]["id"] == "h12-bayesian-prefetch"
    assert payload["protocol"]["id"] == "adaptive-prefetch"
