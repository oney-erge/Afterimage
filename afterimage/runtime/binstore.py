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

Integrity
---------
A product whose entire claim is bit-exactness had, for a while, no way to
detect a corrupted weights.bin -- silent disk corruption would silently
become a silently wrong weight, with no error anywhere. Every blob now
carries a CRC32 computed at write time (cheap: ~1-3 GB/s in Python's zlib
binding, computed once during the already-generous compression pass, not
on the hot per-token read path). Verification is opt-in per read
(`read(ref, verify=True)`) rather than always-on, because CRC32 on every
read would compete with disk bandwidth for the exact I/O path that was
just fixed to stop being the bottleneck -- `verify_store()` below is the
place to spend that cost, once, explicitly.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import zlib

import numpy as np
import torch


@dataclasses.dataclass
class BlobRef:
    offset: int
    nbytes: int
    dtype: str
    shape: tuple
    crc32: int = 0


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
                     dtype=str(arr.dtype), shape=tuple(arr.shape),
                     crc32=zlib.crc32(data))
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

    def read(self, ref: dict, verify: bool = False) -> np.ndarray:
        self._fh.seek(ref["offset"])
        data = self._fh.read(ref["nbytes"])
        if verify:
            got = zlib.crc32(data)
            want = ref.get("crc32", 0)
            if got != want:
                raise ValueError(
                    "checksum mismatch at offset %d (%d bytes): store is "
                    "corrupted -- got crc32=%d, manifest says %d"
                    % (ref["offset"], ref["nbytes"], got, want))
        # np.frombuffer over the bytes object file.read() returns is
        # read-only; torch.from_numpy on a read-only array is UB by
        # PyTorch's own documentation, which is not a risk worth taking in
        # an engine whose entire premise is bit-exactness. The .npz path
        # this replaced didn't have this issue (np.load always returns
        # writable arrays), so it's the one behavior change this format
        # swap needs to account for, not a pre-existing condition.
        arr = np.frombuffer(data, dtype=ref["dtype"]).copy()
        return arr.reshape(ref["shape"])

    def read_many(self, refs: list[dict], *, max_gap_bytes: int = 0,
                  max_extent_bytes: int = 1 << 28,
                  verify: bool = False) -> tuple[list[np.ndarray], int, int]:
        """Read nearby blobs through bounded contiguous storage extents.

        The ordinary hot path issues one seek/read pair per compressed array.
        A layer contains many physically adjacent arrays, so that policy pays
        repeated fixed request overhead even though ``weights.bin`` already
        preserves model order.  This method merges adjacent requests without
        changing a single stored or decoded bit.  It returns arrays in the
        caller's original order plus the actual read-call and byte counts.

        A blob larger than ``max_extent_bytes`` remains one indivisible read;
        the bound only prevents *merging* additional blobs into that extent.

        Memory contract -- deliberately different from ``read()``: each
        returned array is a non-owning ``np.frombuffer`` view over one shared
        per-extent ``bytearray`` (``owndata=False``), and it is not aligned
        to its dtype's itemsize whenever the blob's on-disk offset within the
        extent isn't a multiple of that itemsize. Values are exact and the
        buffer is writable either way; measured, ``torch.from_numpy`` accepts
        an unaligned writable array without raising for every dtype this
        store uses. But do not assume ``owndata`` or alignment here the way
        ``read()`` guarantees them -- multiple returned arrays can alias the
        same underlying buffer, so mutating one can corrupt another.
        """
        if max_gap_bytes < 0:
            raise ValueError("max_gap_bytes must be non-negative")
        if max_extent_bytes < 1:
            raise ValueError("max_extent_bytes must be positive")
        if not refs:
            return [], 0, 0

        ordered = sorted(enumerate(refs), key=lambda item: int(item[1]["offset"]))
        extents: list[dict] = []
        for original_index, ref in ordered:
            start = int(ref["offset"])
            end = start + int(ref["nbytes"])
            if end < start:
                raise ValueError("blob extent overflows its offset")
            if extents:
                previous = extents[-1]
                gap = start - previous["end"]
                merged_end = max(previous["end"], end)
                if (gap >= 0 and gap <= max_gap_bytes
                        and merged_end - previous["start"] <= max_extent_bytes):
                    previous["end"] = merged_end
                    previous["items"].append((original_index, ref))
                    continue
            extents.append({"start": start, "end": end,
                            "items": [(original_index, ref)]})

        output: list[np.ndarray | None] = [None] * len(refs)
        bytes_read = 0
        for extent in extents:
            start = extent["start"]
            nbytes = extent["end"] - start
            self._fh.seek(start)
            data = bytearray(nbytes)
            got_bytes = self._fh.readinto(data)
            if got_bytes != nbytes:
                raise ValueError(
                    "short read at offset %d: wanted %d bytes, got %d" %
                    (start, nbytes, got_bytes))
            bytes_read += nbytes
            view = memoryview(data)
            for original_index, ref in extent["items"]:
                relative = int(ref["offset"]) - start
                blob = view[relative:relative + int(ref["nbytes"])]
                if verify:
                    got = zlib.crc32(blob)
                    want = ref.get("crc32", 0)
                    if got != want:
                        raise ValueError(
                            "checksum mismatch at offset %d (%d bytes): store is "
                            "corrupted -- got crc32=%d, manifest says %d" %
                            (ref["offset"], ref["nbytes"], got, want))
                # bytearray-backed memoryviews are writable, so torch.from_numpy
                # can safely consume these arrays without the second full copy
                # that erased the fixed-request benefit in the first H14 screen.
                array = np.frombuffer(blob, dtype=ref["dtype"])
                output[original_index] = array.reshape(ref["shape"])

        return output, len(extents), bytes_read  # type: ignore[return-value]

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
            "dtype": ref.dtype, "shape": list(ref.shape), "crc32": ref.crc32}


def verify_store(store_dir) -> tuple[bool, list[str]]:
    """Reads every blob in weights.bin once and checks it against the
    manifest's stored CRC32s. This is the expensive, thorough check --
    O(file size) disk read plus O(file size) CRC computation -- meant to be
    run explicitly (a CLI `afterimage verify` command, or once at the start
    of a long batch job), not on every engine startup. See
    StreamingLosslessModel's cheap, always-on truncation check for what
    DOES run on every startup.

    Returns (ok, bad_keys): ok is True iff every blob's checksum matched;
    bad_keys lists which tensor keys failed, empty if ok.
    """
    store_dir = pathlib.Path(store_dir)
    manifest = json.loads((store_dir / "manifest.json").read_text())
    bad = []
    with BinaryWeightReader(store_dir / "weights.bin") as reader:
        for key, meta in manifest["tensors"].items():
            for ref in meta.get("blobs", {}).values():
                if "crc32" not in ref:
                    continue  # store predates checksums; nothing to check
                try:
                    reader.read(ref, verify=True)
                except ValueError:
                    bad.append(key)
                    break
    return (len(bad) == 0, bad)
