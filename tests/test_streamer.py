import time

import torch

from afterimage.runtime.resident import LayerInfo, plan_static_residency
from afterimage.runtime.streamer import AsyncStreamer, stream_sequential
from afterimage.runtime.tiers import TieredStore


import pytest

pytestmark = pytest.mark.archive  # Phase-0 subspace-activation-cache branch, killed per its own gate
FETCH_DELAY = 0.02
COMPUTE_DELAY = 0.02
N_LAYERS = 8


class SlowStore(TieredStore):
    """Adds an artificial fetch delay so the overlap between prefetch and
    compute is measurable on this dev machine, where real local-disk reads
    of small test tensors are too fast to show a timing difference."""

    def get(self, key: str) -> torch.Tensor:
        t = super().get(key)
        time.sleep(FETCH_DELAY)
        return t


def _make_store(tmp_path) -> SlowStore:
    store = SlowStore(tmp_path / "nvme")
    for i in range(N_LAYERS):
        store.write_nvme(f"layer{i}", torch.randn(4, 4))
    return store


def test_async_streamer_overlaps_fetch_with_compute(tmp_path):
    store = _make_store(tmp_path)
    keys = [f"layer{i}" for i in range(N_LAYERS)]

    t0 = time.perf_counter()
    for _key, _tensor in stream_sequential(store, keys):
        time.sleep(COMPUTE_DELAY)
    sequential_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    streamer = AsyncStreamer(store, keys, prefetch_depth=2)
    for _key, _tensor in streamer:
        time.sleep(COMPUTE_DELAY)
    overlapped_time = time.perf_counter() - t0

    expected_sequential = N_LAYERS * (FETCH_DELAY + COMPUTE_DELAY)
    expected_overlapped_floor = N_LAYERS * max(FETCH_DELAY, COMPUTE_DELAY)

    assert sequential_time >= expected_sequential * 0.8
    assert overlapped_time < sequential_time * 0.85, (
        f"overlap not observed: sequential={sequential_time:.3f}s "
        f"overlapped={overlapped_time:.3f}s"
    )
    assert overlapped_time >= expected_overlapped_floor * 0.7


def test_async_streamer_yields_every_key_in_order(tmp_path):
    store = _make_store(tmp_path)
    keys = [f"layer{i}" for i in range(N_LAYERS)]
    seen = [k for k, _t in AsyncStreamer(store, keys, prefetch_depth=3)]
    assert seen == keys


def test_residency_plan_respects_budget_and_favours_priority():
    layers = [
        LayerInfo("a", nbytes=100, priority=1.0),
        LayerInfo("b", nbytes=100, priority=5.0),
        LayerInfo("c", nbytes=100, priority=1.0),
    ]
    resident, streamed = plan_static_residency(layers, budget_bytes=150)
    assert "b" in resident  # highest priority density must be kept
    assert set(resident) | set(streamed) == {"a", "b", "c"}
    assert sum(l.nbytes for l in layers if l.key in resident) <= 150


def test_residency_plan_empty_budget_streams_everything():
    layers = [LayerInfo("a", nbytes=10), LayerInfo("b", nbytes=10)]
    resident, streamed = plan_static_residency(layers, budget_bytes=0)
    assert resident == []
    assert set(streamed) == {"a", "b"}
