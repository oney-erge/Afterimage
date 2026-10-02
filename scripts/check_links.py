"""Fail if any tracked Markdown file links to a repository path that does not
exist. Moving or deleting a document is the usual way a reader lands on a 404;
this runs in CI so that never ships. External URLs and in-page anchors are not
checked.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import urllib.parse

LINK = re.compile(r'\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)|(?:href|src)="([^"]+)"')


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    files = subprocess.run(["git", "ls-files", "*.md"], cwd=root,
                           capture_output=True, text=True, check=True).stdout.split()
    broken = []
    for name in files:
        path = root / name
        if not path.exists():
            continue
        for match in LINK.finditer(path.read_text(encoding="utf-8")):
            target = next(group for group in match.groups() if group)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = urllib.parse.unquote(target.split("#", 1)[0])
            if relative and not (path.parent / relative).exists():
                broken.append("%s -> %s" % (name, target))
    if broken:
        print("\n".join(broken))
        print("\n%d broken link(s)." % len(broken))
        return 1
    print("%d Markdown files, no broken repository links." % len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
