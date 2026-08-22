"""Byte-accounting summary (IMPLEMENTATION_PLAN.md #3.1-#3.2).

Thin formatting layer over TieredStore.stats -- the primary metric (GB per
accepted token) and the diagnostic secondaries all derive from the same
counters, so there is exactly one source of truth for "how many bytes moved."
"""
from __future__ import annotations

import dataclasses

from ..runtime.tiers import TieredStore


@dataclasses.dataclass
class IOSummary:
    bytes_read: dict[str, int]
    bytes_written: dict[str, int]
    read_bandwidth_gbps: dict[str, float]
    total_bytes_read: int

    def gb_per_token(self, tokens: int) -> float:
        return (self.total_bytes_read / 1e9) / tokens if tokens else 0.0


def summarize(store: TieredStore) -> IOSummary:
    reads = {t.value: s.bytes_read for t, s in store.stats.items()}
    writes = {t.value: s.bytes_written for t, s in store.stats.items()}
    bw = {t.value: s.read_bandwidth_gbps for t, s in store.stats.items()}
    total = sum(reads.values())
    return IOSummary(bytes_read=reads, bytes_written=writes, read_bandwidth_gbps=bw, total_bytes_read=total)
