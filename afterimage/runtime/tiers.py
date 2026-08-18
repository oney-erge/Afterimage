"""Storage-tier abstraction: VRAM / pinned host RAM / NVMe.

Byte accounting here is the source of truth for the project's primary metric
(GB transferred per accepted token, IMPLEMENTATION_PLAN.md #3.1). Every read
and write that crosses a tier boundary must be counted here, not estimated
after the fact.

On a CUDA machine, VRAM is real GPU memory and NVMe reads are real file I/O
staged through a pinned host buffer. On this development machine (CPU-only
torch, see IMPLEMENTATION_STATUS.md) VRAM and RAM both fall back to plain CPU
tensors, but NVMe is still real disk I/O -- the accounting and the streaming
logic are exercised for real, only the VRAM/RAM speed gap is not physically
present.
"""
from __future__ import annotations

import dataclasses
import enum
import pathlib
import threading
import time

import numpy as np
import torch

from . import directio


class Tier(enum.Enum):
    VRAM = "vram"
    RAM = "ram"
    NVME = "nvme"


@dataclasses.dataclass
class TierStats:
    bytes_read: int = 0
    bytes_written: int = 0
    read_count: int = 0
    write_count: int = 0
    read_seconds: float = 0.0
    write_seconds: float = 0.0

    def reset(self) -> None:
        self.bytes_read = 0
        self.bytes_written = 0
        self.read_count = 0
        self.write_count = 0
        self.read_seconds = 0.0
        self.write_seconds = 0.0

    @property
    def read_bandwidth_gbps(self) -> float:
        if self.read_seconds <= 0:
            return 0.0
        return (self.bytes_read / 1e9) / self.read_seconds


def _device_for(tier: Tier) -> torch.device:
    if tier == Tier.VRAM and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TieredStore:
    """Keys are tensor identities (layer name + tensor name). Values may live
    in VRAM/RAM as resident torch tensors, or on NVMe as files, never both at
    once -- residency is a placement decision made by resident.py, not an
    automatic cache.

    direct_io: use O_DIRECT for NVMe reads (see runtime/directio.py). Measured
    necessary, not optional: on the development rig, buffered reads reported
    up to 7x the true storage bandwidth because the WSL VHDX is cached by
    Windows underneath Linux, where `drop_caches` cannot reach it. Any run
    reporting NVMe-tier bandwidth must have `direct_io_effective` True.
    """

    def __init__(self, nvme_root: str | pathlib.Path, direct_io: bool = False):
        self.nvme_root = pathlib.Path(nvme_root)
        self.nvme_root.mkdir(parents=True, exist_ok=True)
        self.stats: dict[Tier, TierStats] = {t: TierStats() for t in Tier}
        self._resident: dict[str, tuple[Tier, torch.Tensor]] = {}
        self._lock = threading.Lock()
        self.direct_io = direct_io
        self._direct_reads = 0
        self._buffered_reads = 0

    @property
    def direct_io_effective(self) -> bool:
        """True only if direct I/O was requested AND every NVMe read actually
        used it. False means the numbers include page-cache effects and must
        not be presented as storage measurements."""
        return self.direct_io and self._buffered_reads == 0 and self._direct_reads > 0

    def io_mode_report(self) -> str:
        if not self.direct_io:
            return "buffered (direct_io disabled) -- NOT a valid storage measurement"
        if self._buffered_reads:
            return (f"MIXED: {self._direct_reads} direct, {self._buffered_reads} "
                    f"buffered fallback -- NOT a valid storage measurement")
        return f"O_DIRECT ({self._direct_reads} reads) -- valid storage measurement"

    def reset_stats(self) -> None:
        for s in self.stats.values():
            s.reset()

    # -- placement ---------------------------------------------------------

    def put_resident(self, key: str, tensor: torch.Tensor, tier: Tier) -> None:
        assert tier in (Tier.VRAM, Tier.RAM), "NVMe placement is via write_nvme"
        device = _device_for(tier)
        t0 = time.perf_counter()
        t = tensor.detach().to(device=device, copy=True)
        if tier == Tier.RAM and device.type == "cpu" and torch.cuda.is_available():
            t = t.pin_memory()
        dt = time.perf_counter() - t0
        nbytes = t.element_size() * t.nelement()
        with self._lock:
            self._resident[key] = (tier, t)
            st = self.stats[tier]
            st.bytes_written += nbytes
            st.write_count += 1
            st.write_seconds += dt

    def write_nvme(self, key: str, tensor: torch.Tensor) -> None:
        t0 = time.perf_counter()
        if self.direct_io:
            nbytes = directio.write_tensor_raw(self._nvme_path(key), tensor)
        else:
            arr = tensor.detach().cpu().numpy()
            np.save(self._nvme_path(key), arr, allow_pickle=False)
            nbytes = arr.nbytes
        dt = time.perf_counter() - t0
        with self._lock:
            st = self.stats[Tier.NVME]
            st.bytes_written += nbytes
            st.write_count += 1
            st.write_seconds += dt

    def _nvme_path(self, key: str) -> pathlib.Path:
        safe = key.replace("/", "__")
        ext = ".bin" if self.direct_io else ".npy"
        return self.nvme_root / f"{safe}{ext}"

    # -- retrieval ------------------------------------------------------

    def get(self, key: str) -> torch.Tensor:
        with self._lock:
            hit = self._resident.get(key)
        if hit is not None:
            tier, t = hit
            nbytes = t.element_size() * t.nelement()
            st = self.stats[tier]
            st.bytes_read += nbytes
            st.read_count += 1
            return t
        return self._read_nvme(key)

    def _read_nvme(self, key: str) -> torch.Tensor:
        path = self._nvme_path(key)
        t0 = time.perf_counter()
        if self.direct_io:
            t, result = directio.read_tensor_raw(path)
            nbytes = result.bytes_read
            used_direct = result.used_direct
        else:
            arr = np.load(path, allow_pickle=False)
            t = torch.from_numpy(arr)
            nbytes = arr.nbytes
            used_direct = False
        dt = time.perf_counter() - t0
        with self._lock:
            st = self.stats[Tier.NVME]
            st.bytes_read += nbytes
            st.read_count += 1
            st.read_seconds += dt
            if self.direct_io:
                if used_direct:
                    self._direct_reads += 1
                else:
                    self._buffered_reads += 1
        return t

    def read_nvme_raw(self, key: str) -> torch.Tensor:
        """Reads a key directly from NVMe, bypassing residency. Used for
        bit-plane ladder reads (layout.py), which are stored as individual
        plane artifacts on NVMe and never promoted to a resident tier as a
        whole -- only whichever planes precision escalation decides to fetch."""
        return self._read_nvme(key)

    def is_resident(self, key: str) -> bool:
        with self._lock:
            return key in self._resident

    def residency_tier(self, key: str) -> Tier | None:
        with self._lock:
            hit = self._resident.get(key)
        return hit[0] if hit else None
