import json

from afterimage.cli import (
    build_parser, cmd_experiments, cmd_optimize_residency, cmd_profile_trace,
    cmd_test_plan,
)
from afterimage.runtime.critical_path import CriticalPathProfile, TraceRecorder


def test_research_cli_parses_new_commands():
    parser = build_parser()
    assert parser.parse_args(["experiments", "--json"]).func is cmd_experiments
    plan_args = parser.parse_args(["test-plan", "h12-bayesian-prefetch", "--json"])
    assert plan_args.func is cmd_test_plan
    args = parser.parse_args(["profile-trace", "a.json", "--out", "p.json"])
    assert args.func is cmd_profile_trace
    args = parser.parse_args([
        "optimize-residency", "trace.json", "--manifest", "manifest.json",
        "--out", "plan.json", "--vram-budget-gb", "4",
        "--search-method", "qubo"])
    assert args.func is cmd_optimize_residency
    assert args.search_method == "qubo"
    extent_args = parser.parse_args([
        "optimize-residency", "trace.json", "--manifest", "manifest.json",
        "--out", "plan.json", "--vram-budget-gb", "4",
        "--search-method", "extent-qubo"])
    assert extent_args.search_method == "extent-qubo"
    assert parser.parse_args(["pin-preflight", "--static-only"]).static_only


def test_profile_trace_cli_builds_profile(tmp_path):
    recorder = TraceRecorder()
    read = recorder.record("read", "disk", 0.0, 2.0, tensor_key="w")
    recorder.record("decode", "cuda", 2.0, 3.0, tensor_key="w",
                    dependencies=(read,))
    trace_path = tmp_path / "trace.json"
    profile_path = tmp_path / "profile.json"
    recorder.save(trace_path)
    args = build_parser().parse_args(
        ["profile-trace", str(trace_path), "--out", str(profile_path)])
    assert args.func(args) == 0
    profile = CriticalPathProfile.load(profile_path)
    assert profile.tensors["w"].critical_s == 3.0


def test_experiments_cli_emits_machine_readable_registry(capsys):
    args = build_parser().parse_args(["experiments", "--json"])
    assert args.func(args) == 0
    assert len(json.loads(capsys.readouterr().out)["hypotheses"]) == 16


def test_test_plan_cli_emits_hypothesis_specific_stages(capsys):
    args = build_parser().parse_args([
        "test-plan", "h13-qubo-residency", "--json"])
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"]["id"] == "placement-latency"
    assert payload["protocol"]["stages"][-1]["confirmatory"] is True


def test_test_plan_cli_accepts_short_case_insensitive_hypothesis_id(capsys):
    args = build_parser().parse_args(["test-plan", "H12", "--json"])
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hypothesis"]["id"] == "h12-bayesian-prefetch"
    assert payload["protocol"]["id"] == "adaptive-prefetch"
