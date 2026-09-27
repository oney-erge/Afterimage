"""End-to-end smoke test for `afterimage research h65-plan` and the matching
`afterimage run --representation-policy/--representation-plan-state` flags.

This exercises the CLI wiring added for H6.5 discoverability (docs/H65.md), not
the planner's search or replay math, which tests/test_h65_planner.py already
covers directly. It uses a small hand-built but genuinely scheduler-causal
trace (the same shape as tests/test_h65_planner.py's `_batch_join_fixture`)
so `optimize_h65_plan`'s causal-trace validation is exercised for real.
"""
import dataclasses
import json

from afterimage.cli import main
from afterimage.runtime.critical_path import TraceEvent
from afterimage.runtime.representations import RepresentationPlan

MB = 1_000_000


def _causal_events(read_seconds):
    """One forward sweep over two tensors with a real prefetch/layer_ready/
    decode dependency chain -- satisfies h65_planner._validate_causal_trace.
    """
    keys = ("model.layers.0.a", "model.layers.0.b")
    meta = {"sweep": 1, "layer": 0}
    ra_end = read_seconds["a"]
    rb_end = ra_end + read_seconds["b"]
    return [
        TraceEvent("start", "forward_start", "scheduler", 0, 0),
        TraceEvent("launch", "prefetch_launch", "scheduler", 0, 0,
                   ("start",), metadata={**meta, "lead_layers": 0}),
        TraceEvent("ra", "read", "disk", 0, ra_end, ("launch",), keys[0],
                   15 * MB, metadata={**meta, "prefetch": True}),
        TraceEvent("rb", "read", "disk", ra_end, rb_end, ("launch", "ra"),
                   keys[1], 15 * MB, metadata={**meta, "prefetch": True}),
        TraceEvent("ready", "layer_ready", "scheduler", rb_end, rb_end,
                   ("ra", "rb", "start"), metadata=meta),
        TraceEvent("da", "decode", "cuda", rb_end, rb_end + 0.2,
                   ("ra", "ready"), keys[0], 30 * MB, metadata=meta),
        TraceEvent("db", "decode", "cuda", rb_end + 0.2, rb_end + 0.4,
                   ("rb", "da", "ready"), keys[1], 30 * MB, metadata=meta),
        TraceEvent("compute", "compute", "cuda", rb_end + 0.4, rb_end + 0.5,
                   ("db",)),
        TraceEvent("end", "forward_end", "cuda", rb_end + 0.5, rb_end + 0.5,
                   ("compute",)),
    ]


def _write_trace(path, events):
    payload = {"schema_version": 1,
               "events": [dataclasses.asdict(event) for event in events]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest():
    return {"tensors": {key: {
        "orig_bytes": 30 * MB, "comp_bytes": 15 * MB, "compressed": True,
        "blobs": {"codes": {"offset": i * 15 * MB, "nbytes": 15 * MB}},
    } for i, key in enumerate(("model.layers.0.a", "model.layers.0.b"))}}


def test_h65_plan_cli_runs_end_to_end_and_produces_a_loadable_plan(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    h2d_path = tmp_path / "h2d.json"
    h2d_path.write_text(json.dumps({"median_stable_gbps": 1.0,
                                    "memory_mode": "pageable"}), encoding="utf-8")

    trace_paths = []
    for index, mix in enumerate((
            {"a": 0.1, "b": 3.0}, {"a": 0.1, "b": 3.0}, {"a": 0.1, "b": 3.0})):
        path = tmp_path / ("trace-%d.json" % index)
        _write_trace(path, _causal_events(mix))
        trace_paths.append(str(path))

    out_path = tmp_path / "plan.json"
    rc = main([
        "research", "h65-plan", *trace_paths,
        "--manifest", str(manifest_path), "--h2d", str(h2d_path),
        "--vram-budget-gb", "0.2", "--ram-budget-gb", "0.2",
        "--vram-safety-margin-gb", "0.0", "--decode-slice-elems", "1",
        "--search-iterations", "8", "--minimum-trace-count", "3",
        "--out", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()

    plan = RepresentationPlan.load(out_path)
    assert set(plan.choices) == set(_manifest()["tensors"])


def test_run_exposes_representation_plan_flags():
    parser_args = main.__globals__["build_parser"]().parse_args(
        ["run", "some/model", "prompt",
         "--representation-policy", "per_tensor",
         "--representation-plan-state", "plan.json"])
    assert parser_args.representation_policy == "per_tensor"
    assert parser_args.representation_plan_state == "plan.json"
