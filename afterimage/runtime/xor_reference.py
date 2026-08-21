"""Bit-exact reference/delta coding for same-shaped expert tensors.

The transform is XOR, not subtraction: every bit pattern (including NaNs and
signed zero) round-trips exactly.  Compression is useful only when experts
share bit-level structure, so callers should run ``audit_reference_candidates``
before creating a dependent store.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import struct
import zlib

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class XORReferenceBlob:
    shape: tuple[int, ...]
    dtype: str
    compressed_delta: bytes
    raw_nbytes: int
    base_sha256: str
    target_crc32: int
    compression: str = "zlib"

    @property
    def compressed_nbytes(self) -> int:
        return len(self.compressed_delta)

    @property
    def artifact_nbytes(self) -> int:
        return len(_MAGIC) + 8 + len(_header_bytes(self)) + len(self.compressed_delta)


_MAGIC = b"AIXOR1\0"


def _header_bytes(blob: XORReferenceBlob) -> bytes:
    header = dataclasses.asdict(blob)
    header.pop("compressed_delta")
    return json.dumps(header, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def encode_xor_reference(base: torch.Tensor, target: torch.Tensor,
                         level: int = 6) -> XORReferenceBlob:
    if base.shape != target.shape or base.dtype != target.dtype:
        raise ValueError("base and target must have identical shape and dtype")
    base_raw = _raw_bytes(base)
    target_raw = _raw_bytes(target)
    left = np.frombuffer(base_raw, dtype=np.uint8)
    right = np.frombuffer(target_raw, dtype=np.uint8)
    delta = np.bitwise_xor(left, right).tobytes()
    return XORReferenceBlob(
        shape=tuple(target.shape), dtype=str(target.dtype),
        compressed_delta=zlib.compress(delta, level=level),
        raw_nbytes=len(target_raw), base_sha256=hashlib.sha256(base_raw).hexdigest(),
        target_crc32=zlib.crc32(target_raw))


_TORCH_DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
}


def decode_xor_reference(base: torch.Tensor, blob: XORReferenceBlob) -> torch.Tensor:
    if tuple(base.shape) != blob.shape or str(base.dtype) != blob.dtype:
        raise ValueError("base does not match reference blob shape/dtype")
    base_raw = _raw_bytes(base)
    if hashlib.sha256(base_raw).hexdigest() != blob.base_sha256:
        raise ValueError("base tensor checksum mismatch")
    delta = zlib.decompress(blob.compressed_delta)
    if len(delta) != blob.raw_nbytes:
        raise ValueError("reference delta is truncated or malformed")
    recovered = np.bitwise_xor(np.frombuffer(base_raw, dtype=np.uint8),
                               np.frombuffer(delta, dtype=np.uint8)).copy()
    raw = recovered.tobytes()
    if zlib.crc32(raw) != blob.target_crc32:
        raise ValueError("reconstructed target checksum mismatch")
    dtype = _TORCH_DTYPES.get(blob.dtype)
    if dtype is None:
        raise ValueError("unsupported tensor dtype %s" % blob.dtype)
    # View through uint8 preserves every underlying bit. clone owns writable
    # storage before the dtype reinterpretation.
    out = torch.from_numpy(recovered).clone().view(dtype)
    return out.reshape(blob.shape)


def save_xor_reference(blob: XORReferenceBlob, path) -> None:
    """Persist a self-describing dependent artifact atomically."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = blob.compressed_delta
    header_bytes = _header_bytes(blob)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(_MAGIC)
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload)
    tmp.replace(path)


def load_xor_reference(path) -> XORReferenceBlob:
    with pathlib.Path(path).open("rb") as handle:
        if handle.read(len(_MAGIC)) != _MAGIC:
            raise ValueError("not an Afterimage XOR-reference artifact")
        size_raw = handle.read(8)
        if len(size_raw) != 8:
            raise ValueError("truncated XOR-reference header")
        header_size = struct.unpack("<Q", size_raw)[0]
        header_raw = handle.read(header_size)
        if len(header_raw) != header_size:
            raise ValueError("truncated XOR-reference header")
        header = json.loads(header_raw)
        payload = handle.read()
    if header.get("compression") != "zlib":
        raise ValueError("unsupported XOR-reference compression")
    return XORReferenceBlob(shape=tuple(header["shape"]), dtype=header["dtype"],
                            compressed_delta=payload,
                            raw_nbytes=int(header["raw_nbytes"]),
                            base_sha256=header["base_sha256"],
                            target_crc32=int(header["target_crc32"]),
                            compression=header["compression"])


def audit_reference_candidates(tensors: dict[str, torch.Tensor], level: int = 1,
                               base_keys=None) -> dict:
    """Find the smallest exact base/delta pairing for each compatible tensor."""
    bases = set(tensors) if base_keys is None else set(base_keys)
    unknown = bases - set(tensors)
    if unknown:
        raise ValueError("unknown reference bases: %s" % sorted(unknown))
    result = {}
    for target_key, target in tensors.items():
        if base_keys is not None and target_key in bases:
            result[target_key] = None
            continue
        raw_nbytes = len(_raw_bytes(target))
        best = None
        for base_key, base in tensors.items():
            if (base_key not in bases or base_key == target_key
                    or base.shape != target.shape or base.dtype != target.dtype):
                continue
            blob = encode_xor_reference(base, target, level=level)
            candidate = {"base": base_key, "compressed_bytes": blob.artifact_nbytes,
                         "payload_bytes": blob.compressed_nbytes,
                         "ratio": raw_nbytes / max(blob.artifact_nbytes, 1)}
            if best is None or candidate["compressed_bytes"] < best["compressed_bytes"]:
                best = candidate
        result[target_key] = best
    return result
