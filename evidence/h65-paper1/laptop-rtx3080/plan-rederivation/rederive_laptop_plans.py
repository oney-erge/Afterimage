#!/usr/bin/env python3
"""Rebuild the H6.5 plans behind the paper's RTX 3080 Laptop results (Table 9
and Figures 7-8) from their recorded inputs, on a CPU, and check them byte for
byte against the frozen plans the timed runs actually loaded.

Everything the search consumed is in this folder: each model's store
manifest, its three calibration traces, and the measured pageable H2D rate.
The search settings below are copied from each run's own planner record.

These plans were built by the H6.5 planner at public commit 6c37700, which
predates two later planner changes (the 2026-09-09 candidate-retention fix and
the 2026-09-10 output-head search seed). So this script runs the planner from
that commit, exported with `git archive`, not from your working tree. That
needs this repository's full history (a plain `git clone`, not a shallow one).

Run (no GPU, about 20 seconds):
    python evidence/h65-paper1/laptop-rtx3080/plan-rederivation/rederive_laptop_plans.py
"""
from __future__ import annotations

import hashlib
import importlib
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
FROZEN = HERE.parent / "frozen-plans"
PLANNER_COMMIT = "6c377007ef834ab99159d49096e91c8383ccbda5"

# Copied from each run's planner record (planner/*-pilot-*.json): identical for
# both models. The timed H6.5 arm loaded the full (compressed-RAM-enabled)
# search's candidate. The planner at this commit predates extra_seed_choices,
# so that search ran on its own, not seeded by the placement-only search.
SETTINGS = dict(vram_budget_gb=4.0, ram_budget_gb=8.0, vram_safety_margin_gb=0.5,
                decode_slice_elems=4194304, search_iterations=128, seed=20260909,
                minimum_predicted_improvement=0.0, minimum_live_improvement=0.0,
                h2d_memory_mode="pageable_blocking", ram_prepare_seconds=None)
MODELS = {
    # model: (frozen plan the timed H6.5 arm loaded, candidates the search scored)
    "qwen3-14b": ("qwen3-14b-h65-selected-2382b744dacf.json", 1290),
    "gemma2-27b": ("gemma2-27b-h65-selected-51a2846ce1a2.json", 1694),
}
TRACES = ("calibration-trace-b0-summary-photosynthesis.json",
          "calibration-trace-b1-logic-glippets.json",
          "calibration-trace-b2-copy-nonce.json")


def _import_planner_at_commit(workdir: pathlib.Path):
    archive = workdir / "afterimage.tar"
    with archive.open("wb") as out:
        subprocess.run(["git", "archive", PLANNER_COMMIT, "afterimage"],
                       cwd=REPO, stdout=out, check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(workdir, filter="data")
    sys.path.insert(0, str(workdir))
    for name in [m for m in sys.modules if m == "afterimage" or m.startswith("afterimage.")]:
        del sys.modules[name]
    planner = importlib.import_module("afterimage.runtime.h65_planner")
    critical = importlib.import_module("afterimage.runtime.critical_path")
    assert pathlib.Path(planner.__file__).resolve().is_relative_to(workdir.resolve())
    return planner.optimize_h65_plan, critical.TraceRecorder


def main() -> int:
    h2d = json.loads((HERE / "pageable-h2d-rtx3080-20260828.json").read_text(encoding="utf-8"))
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        try:
            optimize_h65_plan, TraceRecorder = _import_planner_at_commit(workdir)
        except subprocess.CalledProcessError:
            print("Could not export commit %s. This needs the repository's full "
                  "history: git fetch --unshallow" % PLANNER_COMMIT[:7])
            return 1
        for model, (frozen_name, recorded_scored) in MODELS.items():
            folder = HERE / model
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            traces = [TraceRecorder.load(folder / name) for name in TRACES]
            params = dict(SETTINGS, h2d_gbps=float(h2d["median_stable_gbps"]))
            full = optimize_h65_plan(manifest, traces, enable_compressed_ram=True, **params)
            rebuilt = workdir / (model + ".json")
            full.candidate_plan.save(rebuilt)
            # Plans saved before this repository wrote them with LF on every
            # platform come out CRLF on Windows; compare the LF form.
            got = hashlib.sha256(rebuilt.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            want = hashlib.sha256((FROZEN / frozen_name).read_bytes()).hexdigest()
            ok = got == want and full.report.candidates_scored == recorded_scored
            failures += not ok
            print("[%s] %s: rebuilt plan %s, published %s; %d candidates scored "
                  "(recorded %d)" % ("OK  " if ok else "FAIL", model, got[:12], want[:12],
                                     full.report.candidates_scored, recorded_scored))
    print("ALL PLANS REPRODUCED BYTE FOR BYTE" if not failures
          else "%d PLAN(S) DID NOT REPRODUCE" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
