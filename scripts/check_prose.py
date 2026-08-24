"""Keeps the user-facing docs from drifting back into machine-written prose.

Em dash density is the single most reliable, cheap-to-check signal: LLM
output uses far more em dashes than people writing by hand, almost always
in places a person would reach for a period, a comma, or parentheses
instead. Style guidance nobody enforces reverts within a month, so this
runs in CI.

Deliberately narrow: the research docs (RESEARCH_METHODS.md, RESULTS_LOG.md,
HYPOTHESIS_LINEAGE.md, and everything under docs/archive/) are exempt.
Precise, qualified language is correct there, not a style problem -- this
check is about the docs a new user actually reads first.
"""
from __future__ import annotations

import pathlib
import sys

BUDGET_PER_1000_WORDS = 3.0

CHECKED_FILES = [
    "README.md",
    "docs/USAGE.md",
    "docs/CONFIGURATION.md",
    "docs/TROUBLESHOOTING.md",
    "docs/FAQ.md",
]


def em_dash_rate(text: str) -> tuple[int, int, float]:
    words = len(text.split())
    em_dashes = text.count("—")
    rate = (em_dashes / words * 1000) if words else 0.0
    return em_dashes, words, rate


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    failed = False
    for relpath in CHECKED_FILES:
        path = root / relpath
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        em_dashes, words, rate = em_dash_rate(text)
        status = "FAIL" if rate > BUDGET_PER_1000_WORDS else "ok"
        print("%-28s %4d em-dashes / %5d words = %5.1f/1k  [%s]"
              % (relpath, em_dashes, words, rate, status))
        if rate > BUDGET_PER_1000_WORDS:
            failed = True
    if failed:
        print("\nOver budget (%.1f per 1000 words). Replace the flagged em "
              "dashes with a period, a comma, or parentheses -- if the "
              "sentence still reads fine, it never needed the dash."
              % BUDGET_PER_1000_WORDS)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
