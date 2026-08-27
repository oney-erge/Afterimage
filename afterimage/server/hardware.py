"""Cross-platform hardware facts for the CLI and web application."""
from __future__ import annotations

import pathlib
import shutil
from typing import Any


def memory_info() -> dict[str, float | None]:
    """Return total and currently available host memory in GiB.

    psutil is a core dependency because its platform implementations are more
    reliable than maintaining separate Windows, macOS, Linux, and WSL probes in
    Afterimage.  The defensive fallback keeps doctor usable in a partially
    installed environment.
    """

    try:
        import psutil

        value = psutil.virtual_memory()
        gib = 1024**3
        return {
            "total_gib": round(value.total / gib, 2),
            "available_gib": round(value.available / gib, 2),
        }
    except (ImportError, OSError):
        try:
            fields: dict[str, int] = {}
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    name, _, raw = line.partition(":")
                    if name in {"MemTotal", "MemAvailable"}:
                        fields[name] = int(raw.split()[0]) * 1024
            gib = 1024**3
            return {
                "total_gib": round(fields["MemTotal"] / gib, 2),
                "available_gib": (
                    round(fields["MemAvailable"] / gib, 2)
                    if "MemAvailable" in fields
                    else None
                ),
            }
        except (FileNotFoundError, KeyError, OSError, ValueError):
            return {"total_gib": None, "available_gib": None}


def disk_info(path: pathlib.Path) -> dict[str, Any]:
    """Return capacity facts for the filesystem that stores model data."""

    target = pathlib.Path(path).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return {
            "path": str(target),
            "total_gib": None,
            "free_gib": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    gib = 1024**3
    return {
        "path": str(target.resolve()),
        "total_gib": round(usage.total / gib, 2),
        "free_gib": round(usage.free / gib, 2),
        "error": None,
    }
