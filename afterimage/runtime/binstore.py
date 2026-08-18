"""Raw binary weight store -- replaces the .npz-per-tensor format
(LOSSLESS_ENGINE.md Improvement Plan, lever 1).

Measured problem with .npz: np.load on a zip container topped out at
~1.03 GB/s even from page cache, against 2.0 GB/s O_DIRECT on this
machine's disk -- the container format was the ceiling, not the hardware.
It also made the I/O/decode split unmeasurable: np.load's zip members are
read lazily, so array materialization happened INSIDE whatever timing block
called np.load, silently attributing disk time to decode time.

This format is one flat file per model, `weights.bin`, holding every
tensor's byte-arrays concatenated with no container overhead. The manifest
gains one new field per array: its (offset, nbytes) within that file. A
read is exactly `seek(offset); read(nbytes)` -- no parsing, no compression
of the metadata itself, and the byte-for-byte boundary between "reading
bytes" and "decoding symbols" becomes an explicit, separately-timed step.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import torch


@dataclasses.dataclass
class BlobRef:
    offset: int
    nbytes: int
    dtype: str
    shape: tuple


class BinaryWeightWriter:
    """Appends arrays to one file, recording where each one landed."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self._fh = open(self.path, "wb")
        self._pos = 0

    def write(self, arr: np.ndarray) -> BlobRef:
        arr = np.ascontiguousarray(arr)
        data = arr.tobytes()
        ref = BlobRef(offset=self._pos, nbytes=len(data),
                     dtype=str(arr.dtype), shape=tuple(arr.shape))
        self._fh.write(data)
        self._pos += len(data)
        return ref

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BinaryWeightReader:
    """One open file handle for the whole model's lifetime, reused across
    every layer load -- opening/closing per-tensor was never the bottleneck
    the .npz container was, but avoiding it is free and one less syscall
    per tensor per token."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self._fh = open(self.path, "rb")

    def read(self, ref: dict) -> np.ndarray:
        self._fh.seek(ref["offset"])
        data = self._fh.read(ref["nbytes"])
        # np.frombuffer over the bytes object file.read() returns is
        # read-only; torch.from_numpy on a read-only array is UB by
        # PyTorch's own documentation, which is not a risk worth taking in
        # an engine whose entire premise is bit-exactness. The .npz path
        # this replaced didn't have this issue (np.load always returns
        # writable arrays), so it's the one behavior change this format
        # swap needs to account for, not a pre-existing condition.
        arr = np.frombuffer(data, dtype=ref["dtype"]).copy()
        return arr.reshape(ref["shape"])

    def read_row(self, base_offset: int, row_index: int, row_nbytes: int,
                dtype: str) -> np.ndarray:
        """Reads exactly one row of a row-major 2D blob, without touching
        the rest of it -- what makes embedding row-gather possible
        (streaming_engine.py lever 2). A (vocab, hidden) embedding table
        stored row-major has row i contiguous at
        base_offset + i * row_nbytes, so no index or metadata beyond that
        arithmetic is needed to seek directly to it.
        """
        self._fh.seek(base_offset + row_index * row_nbytes)
        data = self._fh.read(row_nbytes)
        return np.frombuffer(data, dtype=dtype).copy()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def blobref_to_dict(ref: BlobRef) -> dict:
    return {"offset": ref.offset, "nbytes": ref.nbytes,
            "dtype": ref.dtype, "shape": list(ref.shape)}
