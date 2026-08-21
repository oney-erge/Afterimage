import json

from afterimage.cli import (
    build_parser, cmd_experiments, cmd_optimize_residency, cmd_profile_trace,
)
from afterimage.runtime.critical_path import CriticalPathProfile, TraceRecorder


def test_research_cli_parses_new_commands():
    parser = build_parser()
    assert parser.parse_args(["experiments", "--json"]).func is cmd_experiments
    args = parser.parse_args(["profile-trace", "a.json", "--out", "p.json"])
    assert args.func is cmd_profile_trace
    args = parser.parse_args([
        "optimize-residency", "trace.json", "--manifest", "manifest.json",
        "--out", "plan.json", "--vram-budget-gb", "4"])
    assert args.func is cmd_optimize_residency


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
    assert len(json.loads(capsys.readouterr().out)["hypotheses"]) == 12
