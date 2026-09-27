"""A draft model with a different vocabulary than the target silently
produces wrong speculative-decoding output rather than an error -- this is
exactly what --profile fast / --auto do for any non-Qwen3 target, since
their draft is hardcoded to Qwen/Qwen3-0.6B. generate_speculative and
generate_adaptive must both refuse a mismatched draft up front instead.
"""
import types

import pytest
import torch

from afterimage.runtime.config import EngineConfig
from afterimage.runtime.streaming_engine import StreamStats, StreamingLosslessModel


def _engine(draft_mode="model"):
    engine = StreamingLosslessModel.__new__(StreamingLosslessModel)
    engine.config = EngineConfig(draft_mode=draft_mode)
    engine.stats = StreamStats()
    engine.adapter = types.SimpleNamespace(
        output_head=types.SimpleNamespace(weight=torch.zeros(151936, 8)))
    return engine


def _draft(vocab_size):
    return types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=vocab_size))


def test_generate_adaptive_rejects_a_mismatched_draft_vocab():
    engine = _engine()
    with pytest.raises(ValueError, match="vocabulary"):
        engine.generate_adaptive(
            torch.tensor([[1, 2, 3]]), max_new_tokens=1,
            draft_model=_draft(128256))  # a Llama-family vocab size, not Qwen3's


def test_generate_adaptive_accepts_a_matching_draft_vocab():
    engine = _engine()
    # Vocab matches; the guard passes and execution reaches real generation
    # logic this fixture does not stub, so a different (unrelated) error is
    # the expected outcome here -- the point is it is NOT the vocab guard.
    with pytest.raises(Exception) as exc_info:
        engine.generate_adaptive(
            torch.tensor([[1, 2, 3]]), max_new_tokens=1,
            draft_model=_draft(151936))
    assert "vocabulary" not in str(exc_info.value)


def test_generate_speculative_rejects_a_mismatched_draft_vocab():
    engine = _engine()
    with pytest.raises(ValueError, match="vocabulary"):
        engine.generate_speculative(
            torch.tensor([[1, 2, 3]]), max_new_tokens=1,
            draft_model=_draft(128256))
