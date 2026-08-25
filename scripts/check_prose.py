"""Keeps the docs from drifting back into machine-written prose.

Em dash density is the cheapest reliable signal. LLM output uses far more em
dashes than people writing by hand, almost always where a person would reach
for a period, a comma, a colon, or parentheses. Style guidance nobody enforces
reverts within a month, so this runs in CI.

Every tracked Markdown file is checked, at zero tolerance, because the repo is
currently at zero. Recasting the sentence is nearly always an improvement; if a
quoted title genuinely needs the character, raise the budget deliberately
rather than reaching for it by habit.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

EM_DASH = "—"
BUDGET = 0


def tracked_markdown(root: pathlib.Path) -> list[pathlib.Path]:
    """Every Markdown file git knows about, so a new doc is covered on day one."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.md"],
            cwd=root, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        # No git available (a source tarball, say). Fall back to a walk.
        return sorted(p for p in root.rglob("*.md") if ".git" not in p.parts)
    return [root / line for line in out.splitlines() if line]


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    checked = 0
    for path in tracked_markdown(root):
        if not path.exists():
            continue
        checked += 1
        count = path.read_text(encoding="utf-8").count(EM_DASH)
        if count > BUDGET:
            offenders.append((path.relative_to(root).as_posix(), count))

    if not offenders:
        print("%d Markdown files, no em dashes." % checked)
        return 0

    for relpath, count in sorted(offenders, key=lambda x: -x[1]):
        print("%-46s %3d em-dashes  [FAIL]" % (relpath, count))
    print("\nReplace each one with a period, a comma, a colon, or parentheses. "
          "If the sentence still reads fine, it never needed the dash.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
