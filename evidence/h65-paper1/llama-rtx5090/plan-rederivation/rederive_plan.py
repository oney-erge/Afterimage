#!/usr/bin/env python3
"""Rebuild the frozen H6.5 placement-only endpoint plan (SHA-256 a31a0d4b...,
the plan D1 and D2 confirmed against) from its recorded calibration traces,
and compare it tensor-by-tensor against the published plan.

Everything this script needs except one file is already in this repository:

- disk-trace-{0,1,2}.json: the three raw calibration traces (copied verbatim
  from the private campaign output, verified secret- and path-free).
- ram-prepare-seconds.json: the decoded-RAM preparation cost profile, read
  directly out of the run's own recorded planning report (not re-measured).
- ../frozen-plans/h65-endpoint-plan.json: the plan to compare against.

**What you must supply**: the Llama-3.3-70B compressed store's manifest.json,
SHA-256 abb500ea82e7b979f20a32ffd466de639e31773678091ad823586c5a6798ac92.
It is not published here (the store itself is tens of gigabytes; only its
small metadata file is needed, but even that was not captured in this
repository's evidence snapshot -- see evidence/h65-paper1/README.md).

Run:
    python evidence/h65-paper1/llama-rtx5090/plan-rederivation/rederive_plan.py \\
        --manifest /path/to/manifest.json

This needs no GPU: optimize_h65_plan is pure offline replay and search over
the recorded traces. It does take a minute or two (256 search iterations
over 723 tensors).

Every keyword argument below is copied verbatim from this run's own recorded
`status.json` planning report (not guessed), specifically the "placement"
arm (enable_compressed_ram=False), which is the arm whose candidate became
the published h65-endpoint-plan.json. The matching "full" arm additionally
seeds its search with this placement arm's result
(extra_seed_choices=placement.candidate_plan.choices) and adds
compressed_ram to the option space; that produces h65-full-plan.json, used
only by the four-token ablation, and is not reproduced by this script.

A plan is not deterministic byte-for-byte across `RepresentationPlan.save()`
calls in general (dict key order is insertion order, not sorted), so this
compares the semantic content -- one representation choice per tensor, and
the aggregate byte/time totals -- not raw file bytes.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

EXPECTED_MANIFEST_SHA256 = (
    "abb500ea82e7b979f20a32ffd466de639e31773678091ad823586c5a6798ac92")
PUBLISHED_PLAN = HERE.parent / "frozen-plans" / "h65-endpoint-plan.json"

# Copied verbatim from this run's status.json -> planning.placement.
# seed=202609093 as recorded; not a typo for 20260909.
PARAMS = dict(
    vram_budget_gb=8.0,
    ram_budget_gb=16.0,
    vram_safety_margin_gb=0.5,
    h2d_gbps=17.110620973580367,
    h2d_memory_mode="pageable_blocking",
    decode_slice_elems=1 << 22,
    search_iterations=256,
    seed=202609093,
    minimum_predicted_improvement=0.0,
    minimum_live_improvement=0.01,
    enable_compressed_ram=False,
)


def sha256_file(path: pathlib.Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True,
                        help="the Llama-3.3-70B store's manifest.json")
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    got_sha = sha256_file(manifest_path)
    if got_sha != EXPECTED_MANIFEST_SHA256:
        print("WARNING: manifest SHA-256 does not match the frozen run.")
        print("  expected:", EXPECTED_MANIFEST_SHA256)
        print("  got:     ", got_sha)
        print("  Continuing anyway, but a mismatch here explains any plan "
              "difference below.")

    from afterimage.runtime.critical_path import TraceRecorder
    from afterimage.runtime.h65_planner import optimize_h65_plan
    from afterimage.runtime.representations import RepresentationPlan

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    traces = [TraceRecorder.load(HERE / f"disk-trace-{i}.json") for i in range(3)]
    ram_prepare_seconds = json.loads(
        (HERE / "ram-prepare-seconds.json").read_text(encoding="utf-8"))

    print("Running optimize_h65_plan (this takes a minute or two)...")
    try:
        result = optimize_h65_plan(
            manifest, traces, ram_prepare_seconds=ram_prepare_seconds, **PARAMS)
    except ValueError as exc:
        print("optimize_h65_plan rejected this manifest: %s" % exc)
        if got_sha != EXPECTED_MANIFEST_SHA256:
            print("This is almost certainly the SHA-256 mismatch flagged above: "
                  "you need the real Llama-3.3-70B store manifest, not a "
                  "placeholder.")
        return 1
    rebuilt = result.candidate_plan

    published = RepresentationPlan.load(PUBLISHED_PLAN)

    print("\nRebuilt vs. published (%s):" % PUBLISHED_PLAN.name)
    print("  candidates_scored:", result.report.candidates_scored,
          "(paper Table 3: 1,817)")
    print("  vram_bytes:  rebuilt %d  published %d" %
          (rebuilt.vram_bytes, published.vram_bytes))
    print("  ram_bytes:   rebuilt %d  published %d" %
          (rebuilt.ram_bytes, published.ram_bytes))
    print("  predicted_prepare_s: rebuilt %.6f  published %.6f" %
          (rebuilt.predicted_prepare_s, published.predicted_prepare_s))

    keys = sorted(set(rebuilt.choices) | set(published.choices))
    mismatches = [
        key for key in keys
        if rebuilt.choices.get(key) is None
        or published.choices.get(key) is None
        or rebuilt.choices[key].name != published.choices[key].name
    ]
    print("  tensors compared: %d, mismatched representation: %d"
          % (len(keys), len(mismatches)))
    if mismatches:
        print("  first mismatches:")
        for key in mismatches[:10]:
            r = rebuilt.choices.get(key)
            p = published.choices.get(key)
            print("    %s: rebuilt=%s published=%s"
                  % (key, r.name if r else "MISSING", p.name if p else "MISSING"))

    if not mismatches and rebuilt.vram_bytes == published.vram_bytes:
        print("\nMATCH: the rebuilt plan selects the same representation for "
              "every tensor as the published, D1/D2-confirmed plan.")
        return 0
    print("\nDID NOT MATCH EXACTLY. This can be a genuine discrepancy, or "
          "environment drift (Python version, dependency versions, or a "
          "manifest that isn't byte-identical to the one this run used -- "
          "see the SHA-256 check above). Report either as a reproduction "
          "issue: see .github/ISSUE_TEMPLATE/h65_reproduction_report.md")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
