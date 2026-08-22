"""Environment gates for experiments that require page-locked host RAM."""
from __future__ import annotations

import dataclasses
import gc
import sys


@dataclasses.dataclass(frozen=True)
class PinnedMemoryPreflight:
    requested_bytes: int
    memlock_soft_bytes: int | None
    memlock_hard_bytes: int | None
    cuda_available: bool
    allocation_attempted: bool
    success: bool
    reason: str


def _memlock_limits() -> tuple[int | None, int | None]:
    if sys.platform == "win32":
        return None, None
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        infinity = resource.RLIM_INFINITY
        return (None if soft == infinity else int(soft),
                None if hard == infinity else int(hard))
    except (ImportError, OSError, ValueError):
        return None, None


def pinned_memory_preflight(requested_bytes: int, *,
                            attempt_allocation: bool = True) -> PinnedMemoryPreflight:
    """Prove that the requested pinned allocation is possible or fail closed."""
    if requested_bytes < 1:
        raise ValueError("requested_bytes must be positive")
    soft, hard = _memlock_limits()
    import torch
    cuda_available = bool(torch.cuda.is_available())
    if hard is not None and hard < requested_bytes:
        return PinnedMemoryPreflight(
            requested_bytes, soft, hard, cuda_available, False, False,
            "hard memlock limit is below the requested pinned allocation")

    if not cuda_available:
        return PinnedMemoryPreflight(
            requested_bytes, soft, hard, False, False, False,
            "CUDA is unavailable, so pin_memory cannot be validated")
    if not attempt_allocation:
        return PinnedMemoryPreflight(
            requested_bytes, soft, hard, True, False, True,
            "static limits permit the request; allocation was not attempted")

    allocation = None
    try:
        allocation = torch.empty(requested_bytes, dtype=torch.uint8,
                                 device="cpu", pin_memory=True)
        return PinnedMemoryPreflight(
            requested_bytes, soft, hard, True, True, True,
            "requested pinned allocation succeeded")
    except RuntimeError as exc:
        return PinnedMemoryPreflight(
            requested_bytes, soft, hard, True, True, False,
            "pin_memory allocation failed: %s" % exc)
    finally:
        del allocation
        gc.collect()
