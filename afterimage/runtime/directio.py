"""O_DIRECT weight reads.

Added after empirical measurement on the development rig (2026-08-17)
disproved the assumption in IMPLEMENTATION_PLAN.md #4.1 that `drop_caches`
alone is sufficient to certify NVMe numbers. Measured on WSL2 ext4, same
file, same drive:

    2 GB file, buffered after drop_caches :  3.5 GB/s
    2 GB file, buffered warm              : 14.8 GB/s
    24 GB file, buffered                  :  4.3 GB/s
    24 GB file, O_DIRECT                  :  2.0 GB/s   <- the true rate

The buffered numbers are inflated by two caches, only one of which Linux
controls: its own page cache (droppable) and the Windows-side cache of the
VHDX file backing the WSL filesystem (not droppable from inside Linux). A
benchmark trusting the buffered path would have reported up to 7x the real
storage bandwidth.

O_DIRECT bypasses the page cache at the syscall level, so it cannot be
defeated by a cache layer above or below us. It is what databases use for
exactly this reason. The cost is that reads must be aligned: offset, length,
and the destination buffer address all need to be multiples of the device
logical block size (512 or 4096 bytes).

Falls back to buffered reads where O_DIRECT is unavailable (Windows, DrvFs,
some network filesystems) and reports which mode was used -- callers must
not present buffered-mode numbers as storage measurements.
"""
from __future__ import annotations

import dataclasses
import json
import mmap
import os
import pathlib
import platform

import numpy as np
import torch

_O_DIRECT = getattr(os, "O_DIRECT", None)
DEFAULT_ALIGNMENT = 4096


@dataclasses.dataclass
class ReadResult:
    data: bytes
    used_direct: bool
    bytes_read: int


def direct_io_available() -> bool:
    return _O_DIRECT is not None and platform.system() == "Linux"


class AlignedBuffer:
    """Page-aligned buffer. mmap of an anonymous region is guaranteed
    page-aligned, which satisfies O_DIRECT's memory-alignment requirement --
    a plain bytearray does not."""

    def __init__(self, size: int, alignment: int = DEFAULT_ALIGNMENT):
        self.size = ((size + alignment - 1) // alignment) * alignment
        self._mm = mmap.mmap(-1, self.size)

    def readinto_from(self, fd: int, nbytes: int) -> int:
        # The memoryview must be released before the mmap can be closed, or
        # CPython raises "BufferError: cannot close exported pointers exist".
        # This only fires where O_DIRECT is actually taken (Linux), so it is
        # invisible to a Windows test run -- found by
        # scripts/verify_odirect.py on WSL2.
        view = memoryview(self._mm)
        try:
            total = 0
            while total < nbytes:
                chunk = os.readv(fd, [view[total:self.size]])
                if chunk <= 0:
                    break
                total += chunk
            return total
        finally:
            view.release()

    def tobytes(self, nbytes: int) -> bytes:
        return self._mm[:nbytes]

    def close(self) -> None:
        self._mm.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_file(path: str | pathlib.Path, force_buffered: bool = False,
              alignment: int = DEFAULT_ALIGNMENT) -> ReadResult:
    """Reads an entire file, preferring O_DIRECT. Never raises on O_DIRECT
    being unsupported -- falls back and records that it did."""
    path = pathlib.Path(path)
    nbytes = path.stat().st_size

    if not force_buffered and direct_io_available():
        try:
            fd = os.open(str(path), os.O_RDONLY | _O_DIRECT)
            try:
                with AlignedBuffer(nbytes, alignment) as buf:
                    got = buf.readinto_from(fd, nbytes)
                    return ReadResult(buf.tobytes(min(got, nbytes)), True, min(got, nbytes))
            finally:
                os.close(fd)
        except OSError:
            pass  # filesystem refused O_DIRECT; fall through to buffered

    with open(path, "rb") as f:
        data = f.read()
    return ReadResult(data, False, len(data))


# -- raw tensor format ---------------------------------------------------
#
# .npy carries a variable-length header, which makes aligned partial reads
# awkward. Weights are stored instead as a flat .bin of raw little-endian
# values plus a .json sidecar holding dtype and shape, so an O_DIRECT read
# of the .bin needs no parsing beyond a frombuffer.

_TORCH_TO_NP = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "float32",  # bf16 has no numpy equivalent; widen on write
    torch.int8: "int8",
    torch.int32: "int32",
    torch.int64: "int64",
}


def write_tensor_raw(path: str | pathlib.Path, tensor: torch.Tensor) -> int:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = tensor.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    np_dtype = _TORCH_TO_NP.get(t.dtype)
    if np_dtype is None:
        raise TypeError(f"unsupported dtype for raw storage: {t.dtype}")

    arr = t.numpy()
    with open(path, "wb") as f:
        f.write(arr.tobytes(order="C"))
    path.with_suffix(".json").write_text(json.dumps({
        "dtype": np_dtype,
        "shape": list(arr.shape),
    }))
    return arr.nbytes


def read_tensor_raw(path: str | pathlib.Path, force_buffered: bool = False
                    ) -> tuple[torch.Tensor, ReadResult]:
    path = pathlib.Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    result = read_file(path, force_buffered=force_buffered)
    expected = int(np.prod(meta["shape"])) if meta["shape"] else 1
    arr = np.frombuffer(result.data, dtype=meta["dtype"], count=expected)
    arr = arr.reshape(meta["shape"])
    return torch.from_numpy(arr.copy()), result
