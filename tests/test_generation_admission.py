from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from afterimage.server import app as app_module
from afterimage.server.jobs import JobRegistry


def test_stream_owns_gpu_until_producer_finishes_without_a_consumer(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    registry = JobRegistry()
    monkeypatch.setattr(app_module, "registry", registry)
    monkeypatch.setattr(app_module, "_stats_usage", lambda *_args: {"completion_tokens": 1})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Engine:
        stats = SimpleNamespace(reset=lambda: None)

        def generate_greedy(self, ids, **kwargs):
            entered.set()
            assert release.wait(5)
            kwargs["on_token"](7)
            return ids

    tokenizer = SimpleNamespace(decode=lambda _tokens, **_kwargs: "token")
    monkeypatch.setattr(
        app_module, "_prepare_generation",
        lambda _req: (Engine(), tokenizer, torch.tensor([[1]]), set(), None, 0),
    )
    request = app_module.ChatCompletionRequest(
        model="test", messages=[app_module.ChatMessage(role="user", content="hello")], stream=True
    )
    response = app_module.chat_completions(request)
    assert response.media_type == "text/event-stream"
    assert entered.wait(5)
    lane = registry._lane_lock("gpu")
    assert not lane.acquire(blocking=False)
    release.set()
    assert lane.acquire(timeout=5)
    lane.release()


def test_failed_preparation_releases_gpu_lane(monkeypatch):
    registry = JobRegistry()
    monkeypatch.setattr(app_module, "registry", registry)

    def fail(_req):
        raise RuntimeError("model failed to load")

    monkeypatch.setattr(app_module, "_prepare_generation", fail)
    request = app_module.ChatCompletionRequest(
        model="test", messages=[app_module.ChatMessage(role="user", content="hello")]
    )
    with pytest.raises(RuntimeError, match="failed to load"):
        app_module.chat_completions(request)
    lane = registry._lane_lock("gpu")
    assert lane.acquire(blocking=False)
    lane.release()
