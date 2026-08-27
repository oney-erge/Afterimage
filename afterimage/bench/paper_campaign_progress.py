"""Shared, dependency-light parsing of run_paper_comparison.py's result
files -- one place both scripts/campaign_status.py (a terminal-facing CLI)
and afterimage/server/app.py's /api/campaigns (the web UI's monitoring
panel) read from, so the two never drift into disagreeing about what a
"done" or "failed" cell means.

Deliberately stdlib-only (no torch/transformers import): the server
process must be able to poll this on every request without pulling in
anything GPU-related, and a CLI user should be able to run
scripts/campaign_status.py without the project's [bench] extras.

Every function here is pure and read-only -- it only ever reads the
.partial/final JSON files a campaign already checkpoints after each cell
(see run_paper_comparison.py's run_one_token_length), never writes to
them or touches the running process.
"""
from __future__ import annotations

import json
import pathlib
import statistics
from typing import Any


def human_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return "%dh%02dm%02ds" % (hours, minutes, seconds)
    if minutes:
        return "%dm%02ds" % (minutes, seconds)
    return "%ds" % seconds


def find_length_files(out_dir: pathlib.Path, run_label: str | None = None) -> list[pathlib.Path]:
    """Every {label}-{n}tok.json[.partial] under out_dir, one entry per
    token length, final result preferred over a stale .partial of the
    same length if somehow both exist."""
    if not out_dir.exists():
        return []
    pattern = "%s-*tok.json*" % (run_label or "*")
    candidates = sorted(out_dir.glob(pattern))
    by_length: dict[tuple[str, int], pathlib.Path] = {}
    for path in candidates:
        name = path.name
        core = name[:-len(".partial")] if name.endswith(".partial") else name
        if not core.endswith("tok.json"):
            continue
        stem = core[: -len("tok.json")]
        try:
            label, n_str = stem.rsplit("-", 1)
            n_tokens = int(n_str)
        except ValueError:
            continue
        key = (label, n_tokens)
        if key in by_length and by_length[key].name.endswith(".partial") is False:
            continue  # already have the final version, keep it
        by_length[key] = path
    return [by_length[key] for key in sorted(by_length, key=lambda k: (k[0], k[1]))]


def completed_cells(result: dict) -> set[tuple[int, str]]:
    """Mirrors run_paper_comparison.completed_cells exactly."""
    return {(cell["block"], cell["method"]) for cell in result.get("cells", [])
           if cell.get("error") is None}


def failed_cells(result: dict) -> list[dict]:
    return [cell for cell in result.get("cells", []) if cell.get("error") is not None]


def method_stats(result: dict) -> dict[str, dict]:
    """Handles both result shapes: an in-progress .partial file keeps raw
    rows in rows_by_method, while a finalized (non-.partial) file
    replaces that with a "methods" list of already-aggregated per-method
    summaries (see aggregate() in run_paper_comparison.py)."""
    stats: dict[str, dict] = {}
    if "rows_by_method" in result:
        for method_id, rows in result["rows_by_method"].items():
            spt = [row["seconds_per_token"] for row in rows if row.get("seconds_per_token") is not None]
            stats[method_id] = {
                "rows": len(rows),
                "mean_seconds_per_token": statistics.mean(spt) if spt else None,
            }
    else:
        for entry in result.get("methods", []):
            summary = entry.get("summary") or {}
            stats[entry["method_id"]] = {
                "rows": len(entry.get("rows") or []),
                "mean_seconds_per_token": summary.get("seconds_per_token"),
            }
    return stats


def summarize_length(path: pathlib.Path) -> dict[str, Any]:
    """One JSON-safe summary dict for a single token-length result file --
    the shape both the CLI renderer and the /api/campaigns endpoint build
    on."""
    result = json.loads(path.read_text(encoding="utf-8"))
    is_final = not path.name.endswith(".partial")
    blocks = result.get("blocks_requested", 0)
    selected = result.get("selected_methods", [])
    required = blocks * len(selected)
    done = completed_cells(result)
    failed = failed_cells(result)
    attempted = len(done) + len(failed)
    elapsed = result.get("elapsed_seconds")

    eta_seconds = None
    if not is_final and attempted and elapsed:
        avg_per_cell = elapsed / attempted
        remaining = required - attempted
        eta_seconds = avg_per_cell * remaining if remaining > 0 else 0.0

    stats = method_stats(result)
    methods = []
    for method_id in selected:
        s = stats.get(method_id, {})
        n_fail = sum(1 for c in failed if c["method"] == method_id)
        n_done = sum(1 for b, m in done if m == method_id)
        marker = "ok" if n_done and not n_fail else ("fail" if n_fail and not n_done else
                 ("mixed" if n_fail else "pending"))
        methods.append({
            "method_id": method_id, "marker": marker,
            "blocks_done": n_done, "blocks_failed": n_fail,
            "rows": s.get("rows", 0),
            "mean_seconds_per_token": s.get("mean_seconds_per_token"),
        })

    recent_failures = [{
        "block": cell["block"], "method": cell["method"],
        "error": (str(cell.get("error", ""))[:200]),
    } for cell in failed[-5:]]

    return {
        "file": path.name,
        "status": "complete" if is_final else "in_progress",
        "max_new_tokens": result.get("max_new_tokens"),
        "prompt_suite": result.get("prompt_suite"),
        "workload": result.get("workload"),
        "blocks_requested": blocks,
        "cells_done": len(done),
        "cells_failed": len(failed),
        "cells_required": required,
        "elapsed_seconds": elapsed,
        "elapsed_human": human_seconds(elapsed),
        "eta_seconds": eta_seconds,
        "eta_human": human_seconds(eta_seconds) if eta_seconds is not None else None,
        "paper_eligible": result.get("paper_eligible"),
        "paper_eligibility_reason": result.get("paper_eligibility_reason"),
        "methods": methods,
        "recent_failures": recent_failures,
    }


def summarize_campaign(out_dir: pathlib.Path, run_label: str | None = None) -> list[dict[str, Any]]:
    """One summary per token-length file found under out_dir, in length
    order. A file that fails to parse (e.g. read mid-write) is skipped
    rather than raising -- a live campaign's .partial file is rewritten
    frequently and a single torn read must not break the whole view."""
    summaries = []
    for path in find_length_files(out_dir, run_label):
        try:
            summaries.append(summarize_length(path))
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return summaries
