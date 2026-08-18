"""Plain-text table formatting for run_matrix results. Deliberately not a
plotting library dependency -- IMPLEMENTATION_PLAN.md's deliverables are
tables and a primary metric, not charts, and keeping this dependency-free
means bench/harness.py output can be inspected in a terminal over SSH on the
actual benchmarking rig.
"""
from __future__ import annotations

from .harness import MetricResult


def format_table(results: dict[str, dict[str, MetricResult]], metric_order: list[str] | None = None) -> str:
    if not results:
        return "(no results)"
    metrics = metric_order or sorted(next(iter(results.values())).keys())
    configs = list(results.keys())

    col_widths = [max(len(c), 16) for c in configs]
    header = "metric".ljust(24) + "".join(c.rjust(w + 2) for c, w in zip(configs, col_widths))
    lines = [header, "-" * len(header)]

    for m in metrics:
        row = m.ljust(24)
        for cfg, w in zip(configs, col_widths):
            r = results[cfg].get(m)
            if r is None:
                cell = "n/a"
            else:
                flag = "" if r.stable else "*"
                cell = f"{r.median:.4g}{flag}"
            row += cell.rjust(w + 2)
        lines.append(row)

    lines.append("")
    lines.append("* = IQR exceeds 15% of median (IMPLEMENTATION_PLAN.md #4.3) -- re-run before trusting")
    return "\n".join(lines)
