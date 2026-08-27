#!/usr/bin/env python3
"""Read-only, real-time progress view for a run_paper_comparison.py
campaign -- answers "where is it right now" without touching the running
process or its files.

The parsing itself lives in afterimage.bench.paper_campaign_progress (a
stdlib-only module also used by the web UI's /api/campaigns endpoint), so
the CLI view here and the browser view never disagree about what "done"
means. This module is only the terminal-facing rendering on top of it.

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
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from afterimage.bench.paper_campaign_progress import summarize_campaign


def render_length(summary: dict) -> str:
    lines = []
    state = "COMPLETE" if summary["status"] == "complete" else "IN PROGRESS"
    lines.append("-- %s (%s tokens, %s, workload=%s) [%s] --" % (
        summary["file"], summary["max_new_tokens"], summary.get("prompt_suite", "?"),
        summary.get("workload", "?"), state))
    lines.append("   cells: %d/%d done, %d failed, of %d required  (elapsed %s)" % (
        summary["cells_done"], summary["cells_required"], summary["cells_failed"],
        summary["cells_required"], summary["elapsed_human"]))
    if summary.get("paper_eligible") is not None:
        lines.append("   paper_eligible: %s (%s)" % (
            summary["paper_eligible"], summary.get("paper_eligibility_reason", "")))
    if summary.get("eta_human"):
        lines.append("   pace: -> ~%s remaining for this length" % summary["eta_human"])

    for method in summary["methods"]:
        spt = method["mean_seconds_per_token"]
        spt_str = ("%.2f s/tok" % spt) if spt is not None else "--"
        lines.append("     %-16s %-6s  blocks done=%d fail=%d  rows=%-3d  %s" % (
            method["method_id"], method["marker"].upper(), method["blocks_done"],
            method["blocks_failed"], method["rows"], spt_str))

    if summary["recent_failures"]:
        lines.append("   recent failures:")
        for cell in summary["recent_failures"][-3:]:
            lines.append("     block %d / %-16s : %s" % (cell["block"], cell["method"], cell["error"]))

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
    summaries = summarize_campaign(out_dir, run_label)
    if not summaries:
        out.append("  no result files found yet matching %s-*tok.json*" % (run_label or "*"))
    for summary in summaries:
        out.append(render_length(summary))
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
            print("\n\n")
            print(render_snapshot(out_dir, args.run_label, log_paths))
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
