#!/usr/bin/env python3
"""Stdlib-only validation of the O_DIRECT read path used by
afterimage/runtime/directio.py. Runs without numpy or torch so it can be
executed inside a fresh WSL2 install before the heavy CUDA dependencies are
present.

Verifies three things:
  1. O_DIRECT opens succeed on this filesystem
  2. A page-aligned mmap buffer satisfies O_DIRECT's alignment requirement
  3. Data read via O_DIRECT is byte-identical to data read normally

Run inside WSL2:  python3 scripts/verify_odirect.py
"""
import hashlib
import mmap
import os
import pathlib
import platform
import sys
import time

ALIGNMENT = 4096


def aligned_read(path: pathlib.Path, nbytes: int) -> bytes:
    o_direct = getattr(os, "O_DIRECT", None)
    if o_direct is None:
        raise RuntimeError("os.O_DIRECT not available on this platform")

    size = ((nbytes + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    fd = os.open(str(path), os.O_RDONLY | o_direct)
    try:
        mm = mmap.mmap(-1, size)  # anonymous mmap is page-aligned
        try:
            view = memoryview(mm)
            try:
                total = 0
                while total < nbytes:
                    got = os.readv(fd, [view[total:size]])
                    if got <= 0:
                        break
                    total += got
                out = bytes(view[:min(total, nbytes)])
            finally:
                # Every derived memoryview must be released before the mmap
                # can be closed, or CPython raises
                # "BufferError: cannot close exported pointers exist".
                view.release()
            return out
        finally:
            mm.close()
    finally:
        os.close(fd)


def main():
    print("=" * 62)
    print("O_DIRECT path validation")
    print("=" * 62)
    print(f"platform      : {platform.system()} {platform.release()}")
    print(f"os.O_DIRECT   : {'present' if hasattr(os, 'O_DIRECT') else 'ABSENT'}")

    if not hasattr(os, "O_DIRECT"):
        print("\nO_DIRECT unavailable -- afterimage falls back to buffered reads")
        print("and will report 'NOT a valid storage measurement'. Correct")
        print("behaviour, but this platform cannot certify NVMe numbers.")
        return 0

    target_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path.home() / "afterimage_odirect"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "odirect_test.bin"

    size_mb = 256
    nbytes = size_mb * 1024 * 1024
    print(f"test file     : {path}  ({size_mb} MB)")

    payload = os.urandom(1024 * 1024)
    with open(path, "wb") as f:
        for _ in range(size_mb):
            f.write(payload)
    os.sync()

    with open(path, "rb") as f:
        buffered = f.read()
    buffered_hash = hashlib.sha256(buffered).hexdigest()

    try:
        t0 = time.perf_counter()
        direct = aligned_read(path, nbytes)
        dt = time.perf_counter() - t0
    except OSError as e:
        print(f"\nO_DIRECT read FAILED on this filesystem: {e}")
        print("This filesystem (likely DrvFs /mnt/*) does not support O_DIRECT.")
        print("Move the weight store to native ext4 -- see the archived Phase-0 execution plan A.5.")
        path.unlink(missing_ok=True)
        return 1

    direct_hash = hashlib.sha256(direct).hexdigest()

    print(f"\nbytes read    : {len(direct):,} (expected {nbytes:,})")
    print(f"elapsed       : {dt:.3f} s  ->  {nbytes / dt / 1e9:.2f} GB/s")
    print(f"buffered sha  : {buffered_hash[:16]}...")
    print(f"O_DIRECT sha  : {direct_hash[:16]}...")

    ok = direct_hash == buffered_hash and len(direct) == nbytes
    print(f"\nRESULT: {'PASS -- O_DIRECT reads are correct and usable' if ok else 'FAIL -- data mismatch'}")

    path.unlink(missing_ok=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
