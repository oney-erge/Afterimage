import torch

from afterimage.runtime.config import EngineConfig
from afterimage.runtime.streaming_engine import StreamStats, StreamingLosslessModel


class _FakeDynamicCache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length

    def crop(self, maximum_length):
        self.length = min(self.length, maximum_length)


def _engine(cache_length):
    engine = StreamingLosslessModel.__new__(StreamingLosslessModel)
    engine.config = EngineConfig(draft_mode="model", spec_target_cache=True)
    engine.stats = StreamStats()
    engine._kv_cache = _FakeDynamicCache(cache_length)
    return engine


def test_cached_speculation_feeds_only_last_committed_token_and_proposals():
    engine = _engine(cache_length=3)
    seq = torch.tensor([[10, 11, 12, 13]])

    target_input, base = engine._spec_target_input(seq, [20, 21])

    assert target_input.tolist() == [[13, 20, 21]]
    assert base == 0
    assert engine.stats.spec_cached_prefix_tokens == 3


def test_cached_speculation_crops_rejected_lookahead_exactly():
    engine = _engine(cache_length=9)

    engine._crop_kv_cache(5)

    assert engine._kv_cache_length() == 5
    assert engine.stats.spec_cache_crops == 1


def test_first_cached_speculation_sweep_uses_the_full_prefix():
    engine = _engine(cache_length=0)
    engine._kv_cache = None
    seq = torch.tensor([[10, 11, 12]])

    target_input, base = engine._spec_target_input(seq, [20, 21])

    assert target_input.tolist() == [[10, 11, 12, 20, 21]]
    assert base == 2
