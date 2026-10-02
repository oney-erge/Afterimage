"""Catch invisible corruption in tracked text files.

A lone carriage return (a CR byte not followed by LF) renders as nothing in
most viewers and silently eats the character after it: a scripted edit once
turned the Windows path `.\\run.ps1` in docs/TROUBLESHOOTING.md into `.` + CR +
`un.ps1`, and it shipped. CRLF line endings are fine (git normalizes them);
a lone CR never is. Frozen, hash-pinned documents are exempt because their
bytes must stay exactly as recorded.
"""
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".md", ".py", ".sh", ".ps1", ".bat", ".command", ".yml", ".yaml",
                 ".toml", ".cff", ".txt", ".html", ".js", ".css", ".cfg", ""}
EXEMPT_PREFIXES = ("docs/h65/protocols/", "evidence/")
LONE_CR = re.compile(rb"\r(?!\n)")


def _tracked_text_files():
    names = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    for name in names:
        path = ROOT / name
        if (path.suffix in TEXT_SUFFIXES and not name.startswith(EXEMPT_PREFIXES)
                and path.is_file()):
            yield name, path


def test_no_tracked_text_file_contains_a_lone_carriage_return():
    offenders = []
    for name, path in _tracked_text_files():
        data = path.read_bytes()
        for match in LONE_CR.finditer(data):
            context = data[max(0, match.start() - 30):match.end() + 10]
            offenders.append("%s: ...%r..." % (name, context))
    assert not offenders, "\n".join(offenders)
