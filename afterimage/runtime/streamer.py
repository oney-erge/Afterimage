"""Double-buffered async layer streaming (IMPLEMENTATION_PLAN.md Phase 2).

On a CUDA machine this would use a dedicated torch.cuda.Stream to overlap
PCIe transfer with GPU compute on the default stream. This development
machine has no CUDA (see IMPLEMENTATION_STATUS.md), so overlap is achieved
with a background thread instead: a producer thread issues NVMe reads ahead
of the consumer. The queueing/prefetch-depth logic and the real disk I/O are
identical to what the CUDA version would do; only the specific mechanism used
to run "the next fetch" concurrently with "the current compute" differs.
GPUDirect Storage does not exist on consumer GeForce cards (LITERATURE.md
#10), so even the CUDA version stages through a pinned host buffer -- this
threaded version is a reasonable stand-in for that staging step, not a
different architecture.
"""
from __future__ import annotations

import queue
import threading

import torch

from .tiers import TieredStore

_SENTINEL = object()


class AsyncStreamer:
    def __init__(self, store: TieredStore, keys: list[str], prefetch_depth: int = 2):
        self.store = store
        self.keys = keys
        self.prefetch_depth = max(1, prefetch_depth)

    def _producer(self, q: "queue.Queue"):
        for k in self.keys:
            t = self.store.get(k)
            q.put((k, t))
        q.put(_SENTINEL)

    def __iter__(self):
        q: queue.Queue = queue.Queue(maxsize=self.prefetch_depth)
        thread = threading.Thread(target=self._producer, args=(q,), daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            yield item
        thread.join()


def stream_sequential(store: TieredStore, keys: list[str]):
    """No prefetch -- fetch, use, fetch, use. This is the AirLLM-equivalent
    baseline path (baselines/b3_sequential.py): I/O and compute never
    overlap, by construction."""
    for k in keys:
        yield k, store.get(k)
