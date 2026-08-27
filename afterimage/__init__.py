"""Single-sourced from pyproject.toml's [project].version, not duplicated --
an editable install with no build metadata yet (e.g. running straight from a
checkout before `pip install -e .`) falls back to "0.0.0.dev0" rather than a
second hand-maintained number that can silently drift from the real one."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("afterimage-llm")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
