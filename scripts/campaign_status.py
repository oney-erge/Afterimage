#!/usr/bin/env python3
"""Read-only, real-time progress view for a run_paper_comparison.py
campaign -- answers "where is it right now" without touching the running
process or its files.

This is deliberately dependency-light (stdlib only, no torch/transformers
import) so it stays cheap to run every few seconds from a second terminal
while the actual campaign occupies the GPU: it only reads the .partial/
final JSON result files the campaign already checkpoints after every
single (block, method) cell (see run_paper_comparison.py's
run_one_token_length -- checkpoint(partial, result) is called right after
each cell, win or lose, which is also what makes --resume safe: killing
the campaign process at any point loses at most the one cell that was
in flight, never previously-completed work).

Usage:
    # one-shot snapshot
    python scripts/campaign_status.py --out-dir results/paper-comparison

    # auto-refresh every 15s until Ctrl-C (a real "live dashboard" in a
    # spare terminal, independent of anyone polling it)
    python scripts/campaign_status.py --out-dir results/paper-comparison --watch 15

    # also show the last few lines of a run's own stdout/stderr log
    python scripts/campaign_status.py --out-dir results/paper-comparison \\
        --log /root/afterimage/logs/run1_paper_generation.log
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import statistics
import sys
import time


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


def load_result(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_length_files(out_dir: pathlib.Path, run_label: str | None) -> list[pathlib.Path]:
    """Every {label}-{n}tok.json[.partial] under out_dir, one entry per
    token length, final result preferred over a stale .partial of the
    same length if somehow both exist."""
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
        # Prefer the final (non-.partial) file; otherwise take the .partial.
        if key not in by_length or name.endswith(".partial") is False:
            if key in by_length and by_length[key].name.endswith(".partial") is False:
                continue  # already have the final version, keep it
            by_length[key] = path
    return [by_length[key] for key in sorted(by_length, key=lambda k: (k[0], k[1]))]


def completed_cells(result: dict) -> set[tuple[int, str]]:
    """Mirrors run_paper_comparison.completed_cells exactly (kept as an
    inline copy, not an import, so this tool never has to pull in torch
    just to read a JSON file -- see module docstring)."""
    return {(cell["block"], cell["method"]) for cell in result.get("cells", [])
           if cell.get("error") is None}


def failed_cells(result: dict) -> list[dict]:
    return [cell for cell in result.get("cells", []) if cell.get("error") is not None]


def method_stats(result: dict) -> dict[str, dict]:
    """Handles both result shapes: an in-progress .partial file keeps raw
    rows in rows_by_method (built up cell by cell), while a finalized
    (non-.partial) file replaces that with a "methods" list of
    already-aggregated per-method summaries (see aggregate() in
    run_paper_comparison.py) and drops rows_by_method entirely."""
    stats = {}
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


def render_length(path: pathlib.Path, out_dir: pathlib.Path) -> str:
    result = load_result(path)
    is_final = not path.name.endswith(".partial")
    n_tokens = result.get("max_new_tokens")
    blocks = result.get("blocks_requested", 0)
    selected = result.get("selected_methods", [])
    required = blocks * len(selected)
    done = completed_cells(result)
    failed = failed_cells(result)
    attempted = len(done) + len(failed)
    elapsed = result.get("elapsed_seconds")

    lines = []
    state = "COMPLETE" if is_final else "IN PROGRESS"
    lines.append("-- %s (%d tokens, %s, workload=%s) [%s] --" % (
        path.name, n_tokens, result.get("prompt_suite", "?"), result.get("workload", "?"), state))
    lines.append("   cells: %d/%d done, %d failed, %d attempted of %d required  (elapsed %s)" % (
        len(done), required, len(failed), attempted, required, human_seconds(elapsed)))
    if is_final and "paper_eligible" in result:
        lines.append("   paper_eligible: %s (%s)" % (
            result["paper_eligible"], result.get("paper_eligibility_reason", "")))

    if not is_final and attempted and elapsed:
        avg_per_cell = elapsed / attempted
        remaining = required - attempted
        eta = avg_per_cell * remaining if remaining > 0 else 0.0
        lines.append("   pace: avg %.1fs/cell across %d attempted cells (all methods blended) "
                     "-> ~%s remaining for this length" % (avg_per_cell, attempted, human_seconds(eta)))

    stats = method_stats(result)
    for method_id in selected:
        s = stats.get(method_id, {})
        n_fail = sum(1 for c in failed if c["method"] == method_id)
        n_done = sum(1 for b, m in done if m == method_id)
        marker = "OK" if n_done and not n_fail else ("FAIL" if n_fail and not n_done else
                 ("MIXED" if n_fail else "..."))
        spt = s.get("mean_seconds_per_token")
        spt_str = ("%.2f s/tok" % spt) if spt is not None else "--"
        lines.append("     %-16s %-5s  blocks done=%d fail=%d  rows=%-3d  %s" % (
            method_id, marker, n_done, n_fail, s.get("rows", 0), spt_str))

    if failed:
        lines.append("   recent failures:")
        for cell in failed[-3:]:
            err = str(cell.get("error", ""))
            err = err if len(err) <= 140 else err[:137] + "..."
            lines.append("     block %d / %-16s : %s" % (cell["block"], cell["method"], err))

    return "\n".join(lines)


def tail_log(log_path: pathlib.Path, n_lines: int = 8) -> str:
    if not log_path.exists():
        return "  (log not found: %s)" % log_path
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return "  (could not read log: %r)" % exc
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join("  | " + line for line in lines[-n_lines:])


def render_snapshot(out_dir: pathlib.Path, run_label: str | None,
                    log_paths: list[pathlib.Path]) -> str:
    out = []
    out.append("=" * 72)
    out.append("Campaign status @ %s  (%s)" %
               (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), out_dir))
    out.append("=" * 72)
    length_files = find_length_files(out_dir, run_label)
    if not length_files:
        out.append("  no result files found yet matching %s-*tok.json*" % (run_label or "*"))
    for path in length_files:
        try:
            out.append(render_length(path, out_dir))
        except (json.JSONDecodeError, OSError) as exc:
            out.append("-- %s : could not read (%r) --" % (path.name, exc))
    for log_path in log_paths:
        out.append("")
        out.append("recent log lines (%s):" % log_path)
        out.append(tail_log(log_path))
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="results/paper-comparison")
    parser.add_argument("--run-label", default=None,
                        help="only show files for this run label; default shows every "
                             "*-<n>tok.json[.partial] under --out-dir")
    parser.add_argument("--log", action="append", default=[], dest="logs",
                        help="path to a campaign stdout/stderr log to tail; repeatable")
    parser.add_argument("--watch", type=float, default=None,
                        help="refresh every N seconds until Ctrl-C, instead of printing once")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    log_paths = [pathlib.Path(p) for p in args.logs]
    if not out_dir.exists():
        print("out-dir does not exist yet: %s" % out_dir, file=sys.stderr)
        return 1

    if args.watch is None:
        print(render_snapshot(out_dir, args.run_label, log_paths))
        return 0

    try:
        while True:
            # Clear-ish: a blank separator rather than an ANSI clear, so
            # scrollback still shows history if redirected to a file.
            print("\n\n")
            print(render_snapshot(out_dir, args.run_label, log_paths))
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
