"""Peak-memory measurement (VALIDATION_PLAN.md #5).

Three numbers must be reported separately, because conflating them is the
standard way memory claims get overstated:

  - weights on disk   -- what the checkpoint costs to store
  - PEAK VRAM         -- what must fit on the GPU; decides whether it runs
  - host RAM high-water -- what offloading pushes onto the CPU side

Offloading does NOT reduce total bytes. It reduces peak VRAM by moving bytes
to RAM/NVMe. A table showing only "VRAM went down" without the host-RAM
column is misleading, so this module reports all three or none.

torch.cuda.max_memory_allocated() only counts allocations PyTorch made
through its own caching allocator. It misses the CUDA context, cuBLAS/cuDNN
workspaces, fragmentation, and anything another process holds. nvidia-smi
sees actual device usage but is process-wide and only samples when polled.
Neither alone is trustworthy, so both are captured and reported, and
nvidia-smi is sampled on a background thread rather than read once at the
end (a single reading after the fact misses the peak entirely).
"""
from __future__ import annotations

import dataclasses
import pathlib
import shutil
import subprocess
import threading
import time

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep elsewhere
    torch = None  # type: ignore


def nvidia_smi_used_mb() -> int | None:
    """Process-wide GPU memory in MiB, or None if nvidia-smi is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def host_rss_bytes() -> int | None:
    """Resident set size of this process, without requiring psutil."""
    try:
        import resource  # Linux/macOS only
        # ru_maxrss is KB on Linux, bytes on macOS. Assume Linux (the
        # benchmarking platform per EXECUTION_PLAN.md Stage A).
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        pass
    try:
        import ctypes
        import ctypes.wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    except Exception:
        pass
    return None


def checkpoint_bytes(path: str | pathlib.Path) -> int:
    """Total size of a model directory or single file, following the same
    convention as `du -b`: sum of regular file sizes, no block rounding."""
    p = pathlib.Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


@dataclasses.dataclass
class MemoryReport:
    torch_peak_vram_bytes: int | None
    smi_peak_used_mb: int | None
    smi_baseline_used_mb: int | None
    host_rss_peak_bytes: int | None
    n_samples: int

    @property
    def torch_peak_vram_gb(self) -> float | None:
        return self.torch_peak_vram_bytes / 1e9 if self.torch_peak_vram_bytes else None

    @property
    def smi_delta_gb(self) -> float | None:
        """Peak minus baseline -- what THIS workload added on top of whatever
        the desktop/other processes were already holding. On a laptop with a
        display attached to the same GPU (the EXECUTION_PLAN.md rig holds
        ~1.5 GB for the Windows desktop), the raw peak overstates the model's
        own cost, and the delta is the honest number to compare across
        configs."""
        if self.smi_peak_used_mb is None or self.smi_baseline_used_mb is None:
            return None
        return max(0, self.smi_peak_used_mb - self.smi_baseline_used_mb) / 1000.0

    @property
    def host_rss_peak_gb(self) -> float | None:
        return self.host_rss_peak_bytes / 1e9 if self.host_rss_peak_bytes else None


class MemoryProbe:
    """Context manager sampling GPU memory on a background thread.

    A single nvidia-smi reading taken after the workload finishes will miss
    the peak -- allocations are transient and the caching allocator may have
    already released them. Sampling at a fixed interval and keeping the max
    is the only way to catch it without profiler instrumentation.
    """

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_mb: int | None = None
        self._baseline_mb: int | None = None
        self._n_samples = 0

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            val = nvidia_smi_used_mb()
            if val is not None:
                self._n_samples += 1
                if self._peak_mb is None or val > self._peak_mb:
                    self._peak_mb = val
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "MemoryProbe":
        self._baseline_mb = nvidia_smi_used_mb()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        # Synchronize BEFORE stopping the sampler: CUDA work queued but not
        # finished would otherwise peak after sampling has already stopped.
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep(self.interval_s * 2)  # let the sampler catch the final state
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def report(self) -> MemoryReport:
        torch_peak = None
        if torch is not None and torch.cuda.is_available():
            torch_peak = int(torch.cuda.max_memory_allocated())
        return MemoryReport(
            torch_peak_vram_bytes=torch_peak,
            smi_peak_used_mb=self._peak_mb,
            smi_baseline_used_mb=self._baseline_mb,
            host_rss_peak_bytes=host_rss_bytes(),
            n_samples=self._n_samples,
        )
