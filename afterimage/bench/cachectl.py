"""Page-cache control (IMPLEMENTATION_PLAN.md #4.1 -- "the one that
invalidates most disk-offload benchmarks").

If the OS caches the model file in free RAM, a second run reads from RAM at
RAM speed while the benchmark reports it as an NVMe number. On a machine with
enough free RAM to hold the model, EVERY run after the first is measuring the
wrong thing. This module does not estimate or work around that -- it either
proves the cache was actually dropped, or refuses to certify the run.

Linux: `echo 3 > /proc/sys/vm/drop_caches` requires root. Verified, not
assumed: caller must confirm via `verify_cache_dropped`, which re-reads a
throwaway file and checks that FIRST-read timing (cold) roughly matches
SECOND-read timing (would be warm if drop_caches silently failed).

Windows: there is no equivalent syscall exposed without a kernel driver.
`is_cache_control_available()` returns False on Windows, and
`bench/harness.py` must refuse to certify NVMe-tier numbers on a platform
where this returns False (IMPLEMENTATION_PLAN.md #4.1: "If neither is
available, do not report Windows disk numbers"). This is a real limitation
of this codebase on Windows, not a placeholder to be silently ignored.
"""
from __future__ import annotations

import ctypes
import pathlib
import platform
import subprocess
import time


def is_cache_control_available() -> bool:
    return platform.system() == "Linux"


def drop_caches() -> bool:
    """Best-effort on Linux; returns whether it plausibly succeeded (no
    exception raised talking to /proc). Returns False immediately on any
    other platform -- callers must not proceed to certify NVMe numbers."""
    if platform.system() != "Linux":
        return False
    try:
        subprocess.run(["sync"], check=True, timeout=30)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except (PermissionError, FileNotFoundError, OSError, subprocess.SubprocessError):
        return False


def verify_cache_dropped(probe_path: pathlib.Path, probe_bytes: int = 64 * 1024 * 1024,
                          warm_vs_cold_ratio_floor: float = 2.0) -> tuple[bool, float, float]:
    """Writes a throwaway file, reads it once (should be cold if drop_caches
    worked), reads it again immediately (will be warm, from the page cache,
    regardless). If the OS actually dropped caches before the first read,
    the first read should be markedly slower than the second. Returns
    (looks_cold, first_read_seconds, second_read_seconds)."""
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(probe_path, "wb") as f:
        f.write(b"\0" * probe_bytes)

    drop_caches()

    t0 = time.perf_counter()
    with open(probe_path, "rb") as f:
        f.read()
    first = time.perf_counter() - t0

    t0 = time.perf_counter()
    with open(probe_path, "rb") as f:
        f.read()
    second = time.perf_counter() - t0

    looks_cold = second > 0 and (first / max(second, 1e-9)) >= warm_vs_cold_ratio_floor
    return looks_cold, first, second


def constrain_available_memory_linux(cgroup_path: pathlib.Path, max_bytes: int) -> bool:
    """Writes memory.max for a cgroup v2 path the caller has already created
    and moved this process into. Returns False (does nothing) off Linux or
    if the cgroup doesn't exist -- this function does not create cgroups or
    require root elevation itself, it only writes the limit file."""
    if platform.system() != "Linux":
        return False
    target = cgroup_path / "memory.max"
    if not target.exists():
        return False
    try:
        target.write_text(str(max_bytes))
        return True
    except OSError:
        return False


def free_ram_bytes() -> int | None:
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb * 1024
        except (FileNotFoundError, OSError, ValueError):
            return None
        return None
    if platform.system() == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullAvailPhys
        return None
    return None
