"""Regenerate docs/assets/results-chart.svg from the README's own results table.

Stdlib only, like the other scripts here. This is the single place that reads
the six-row table in README.md's "## Results" section; tests/test_docs_numbers.py
imports read_results_table() from here instead of parsing the table a second
time, so there's one parser to keep in sync with the table's format.

    python scripts/make_results_chart.py           # write docs/assets/results-chart.svg
    python scripts/make_results_chart.py --check    # exit 1 if the committed file is stale
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
OUT = ROOT / "docs" / "assets" / "results-chart.svg"

# (name prefix used to match a README row, exactness contract, marker shape)
EXACTNESS = {
    "Afterimage + fixed speculation": ("Greedy-token exact at T=0", "diamond"),
    "Hugging Face Accelerate": ("Same checkpoint and token IDs", "circle"),
    "Afterimage exact + 4 GB residency": ("Reference-execution equivalent", "square"),
    "AirLLM": ("Same checkpoint and token IDs", "circle"),
    "Afterimage chunked output head": ("Approximate BF16 matmul", "hollow"),
    "Afterimage exact minimum-memory": ("Reference-execution equivalent", "square"),
}


def read_results_table(text: str | None = None) -> dict[str, tuple[float, float, float]]:
    """Parse README.md's "| Configuration | Peak VRAM | ..." table.

    Returns {name: (peak_vram_gb, seconds_per_token, ratio_vs_airllm)}, in the
    table's own row order (relies on dict insertion order).
    """
    text = README.read_text(encoding="utf-8") if text is None else text
    lines = text[text.index("| Configuration | Peak VRAM"):].splitlines()[2:]
    table: dict[str, tuple[float, float, float]] = {}
    for line in lines:
        if not line.startswith("|"):
            break
        cells = [c.strip().strip("*").strip() for c in line.strip("|").split("|")]
        vram = float(cells[1].split()[0].strip("*"))
        table[cells[0]] = (vram, float(cells[2]), float(cells[3].rstrip("x")))
    return table


def _exactness(name: str) -> tuple[str, str]:
    for prefix, value in EXACTNESS.items():
        if name.startswith(prefix):
            return value
    raise KeyError("no exactness mapping for README row %r -- add one to EXACTNESS" % name)


# Per-point (short label, dx, dy, text-anchor) in SVG px, tuned by hand for
# these six rows and this axis scale so labels don't overlap the plot area,
# each other, or another point's marker. Re-tune (and re-run --check) if the
# table's rows, values, or axis range change.
LABEL = {
    "Afterimage + fixed speculation": ("fixed speculation", 0, 20, "middle"),
    "Hugging Face Accelerate GPU/CPU/disk": ("Accelerate", 0, 20, "middle"),
    "Afterimage exact + 4 GB residency": ("exact + 4 GB residency", 0, -14, "middle"),
    "AirLLM 3.2.0": ("AirLLM 3.2.0", 0, 20, "middle"),
    "Afterimage chunked output head": ("chunked output head", 0, -14, "middle"),
    "Afterimage exact minimum-memory": ("exact minimum-memory", 40, 6, "start"),
}

COLOR = {
    "diamond": "var(--spec)",
    "square": "var(--exact)",
    "circle": "var(--base)",
    "hollow": "var(--approx)",
}

W, H = 720, 510
PAD_L, PAD_R, PAD_T, PAD_B = 60, 30, 26, 120
PLOT_W, PLOT_H = W - PAD_L - PAD_R, H - PAD_T - PAD_B
X_MAX, Y_MAX = 4.5, 35.0


def _x(vram: float) -> float:
    return PAD_L + (vram / X_MAX) * PLOT_W


def _y(seconds: float) -> float:
    return PAD_T + (1 - seconds / Y_MAX) * PLOT_H


def _marker(shape: str, cx: float, cy: float, color: str) -> str:
    r = 7
    if shape == "circle":
        return '<circle cx="%.1f" cy="%.1f" r="%d" fill="%s"/>' % (cx, cy, r, color)
    if shape == "hollow":
        return ('<circle cx="%.1f" cy="%.1f" r="%d" fill="none" stroke="%s" '
                 'stroke-width="2.5" stroke-dasharray="3 3"/>') % (cx, cy, r, color)
    if shape == "square":
        s = r * 1.6
        return '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="2" fill="%s"/>' % (
            cx - s / 2, cy - s / 2, s, s, color)
    if shape == "diamond":
        s = r * 1.25
        pts = "%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f" % (
            cx, cy - s, cx + s, cy, cx, cy + s, cx - s, cy)
        return '<polygon points="%s" fill="%s"/>' % (pts, color)
    raise ValueError(shape)


def build_svg(table: dict[str, tuple[float, float, float]]) -> str:
    grid = []
    for gx in (0, 1, 2, 3, 4):
        x = _x(gx)
        grid.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="grid"/>' % (x, PAD_T, x, H - PAD_B))
        grid.append('<text x="%.1f" y="%d" class="tick">%d</text>' % (x, H - PAD_B + 18, gx))
    for gy in (0, 5, 10, 15, 20, 25, 30, 35):
        y = _y(gy)
        grid.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>' % (PAD_L, y, W - PAD_R, y))
        grid.append('<text x="%d" y="%.1f" class="tick" text-anchor="end">%d</text>' % (PAD_L - 8, y + 4, gy))

    points = []
    for name, (vram, seconds, _ratio) in table.items():
        _label, shape = _exactness(name)
        cx, cy = _x(vram), _y(seconds)
        color = COLOR[shape]
        points.append(_marker(shape, cx, cy, color))
        text, dx, dy, anchor = LABEL[name]
        points.append('<text x="%.1f" y="%.1f" class="label" text-anchor="%s">%s</text>' % (
            cx + dx, cy + dy, anchor, text))

    legend_items = [
        ("diamond", "Speculative (greedy-exact)"),
        ("square", "Exact (reference-equivalent)"),
        ("circle", "External baseline (same tokens)"),
        ("hollow", "Approximate (not lossless)"),
    ]
    legend = []
    cols = [PAD_L, PAD_L + 330]
    rows = [H - 48, H - 24]
    for i, (shape, text) in enumerate(legend_items):
        lx, ly = cols[i % 2], rows[i // 2]
        legend.append(_marker(shape, lx + 7, ly - 4, COLOR[shape]))
        legend.append('<text x="%d" y="%.1f" class="legend">%s</text>' % (lx + 20, ly, text))

    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="-apple-system,Segoe UI,sans-serif">
<title>Afterimage vs. baselines: peak VRAM and seconds per token, Qwen3-14B on an RTX 3080 Laptop GPU</title>
<style>
  :root { --ink:#1b1d2a; --muted:#5b6074; --rule:#d7d9e3; --bg:#ffffff;
          --spec:#5b3fe6; --exact:#0b8fb8; --base:#8a8f9e; --approx:#c9871f; }
  @media (prefers-color-scheme: dark) {
    :root { --ink:#e6e8f0; --muted:#9aa0b4; --rule:#333747; --bg:#161923;
            --spec:#a593ff; --exact:#43c6ee; --base:#9aa0b4; --approx:#e6b84d; }
  }
  rect.bg { fill: var(--bg); }
  .grid { stroke: var(--rule); stroke-width: 1; }
  .axis { stroke: var(--muted); stroke-width: 1.2; }
  .tick { fill: var(--muted); font-size: 12px; }
  .label { fill: var(--ink); font-size: 12px; }
  .legend { fill: var(--muted); font-size: 11.5px; }
  .axis-title { fill: var(--muted); font-size: 12.5px; }
</style>
<rect class="bg" x="0" y="0" width="%d" height="%d"/>
%s
<line class="axis" x1="%d" y1="%d" x2="%d" y2="%d"/>
<line class="axis" x1="%d" y1="%d" x2="%d" y2="%d"/>
<text class="axis-title" x="%d" y="%d" text-anchor="middle">Peak VRAM (GB)</text>
<text class="axis-title" x="14" y="%d" text-anchor="middle" transform="rotate(-90 14 %d)">Seconds / token</text>
%s
%s
</svg>
""" % (
        W, H, W, H, "\n".join(grid),
        PAD_L, PAD_T, PAD_L, H - PAD_B,
        PAD_L, H - PAD_B, W - PAD_R, H - PAD_B,
        PAD_L + PLOT_W / 2, (H - PAD_B) + 42,
        PAD_T + PLOT_H / 2, PAD_T + PLOT_H / 2,
        "\n".join(points), "\n".join(legend),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                         help="exit 1 if docs/assets/results-chart.svg is stale, write nothing")
    args = parser.parse_args()

    svg = build_svg(read_results_table())
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else None
        if current != svg:
            print("docs/assets/results-chart.svg is stale -- run "
                  "python scripts/make_results_chart.py", file=sys.stderr)
            return 1
        print("docs/assets/results-chart.svg is up to date")
        return 0

    OUT.write_text(svg, encoding="utf-8", newline="\n")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
