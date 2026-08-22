import pytest
import torch

from afterimage.runtime import memory_preflight


def test_pin_preflight_fails_before_allocation_when_hard_limit_is_too_low(
        monkeypatch):
    monkeypatch.setattr(memory_preflight, "_memlock_limits",
                        lambda: (64 << 20, 64 << 20))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    called = False

    def forbidden_empty(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("allocation should not be attempted")

    monkeypatch.setattr(torch, "empty", forbidden_empty)
    report = memory_preflight.pinned_memory_preflight(1_600_000_000)
    assert not report.success
    assert not report.allocation_attempted
    assert not called
    assert "memlock" in report.reason


def test_pin_preflight_rejects_nonpositive_request():
    with pytest.raises(ValueError, match="positive"):
        memory_preflight.pinned_memory_preflight(0)
