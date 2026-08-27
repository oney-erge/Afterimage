"""ModelRegistry leaked one open sqlite3 connection (and its OS file handle)
per call: every method used ``with self._connect() as connection:``, and
``sqlite3.Connection.__exit__`` only commits or rolls back a transaction, it
never closes the connection. In a server meant to run for days, with a
download job alone calling ``upsert_model`` once per file (see
afterimage/server/acquisition.py), that leaks one handle per shard of every
model ever downloaded. These tests spy on the real close() calls rather than
just exercising the methods, since the bug was that the code ran perfectly
fine while leaking.
"""
from __future__ import annotations

import sqlite3

import pytest

from afterimage.server.model_registry import ModelRegistry


class _CountingConnection:
    """Wraps a real sqlite3.Connection and records whether close() was
    called, while delegating everything else unchanged."""

    def __init__(self, real: sqlite3.Connection, counter: dict):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_counter", counter)
        counter["opened"] += 1

    def close(self):
        self._counter["closed"] += 1
        return self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._real.__exit__(*exc_info)


@pytest.fixture
def counted_registry(tmp_path, monkeypatch):
    counter = {"opened": 0, "closed": 0}
    real_connect = sqlite3.connect

    def spy_connect(*args, **kwargs):
        return _CountingConnection(real_connect(*args, **kwargs), counter)

    monkeypatch.setattr(
        "afterimage.server.model_registry.sqlite3.connect", spy_connect)
    registry = ModelRegistry(tmp_path / "registry.sqlite3")
    return registry, counter


def test_upsert_and_get_model_close_every_connection_they_open(counted_registry):
    registry, counter = counted_registry
    registry.upsert_model("org/model", state="remote")
    registry.get_model("org/model")
    registry.list_models()
    assert counter["opened"] > 0
    assert counter["closed"] == counter["opened"], (
        f"opened {counter['opened']} connections but only closed "
        f"{counter['closed']} -- leaking {counter['opened'] - counter['closed']}")


def test_job_lifecycle_closes_every_connection_it_opens(counted_registry):
    registry, counter = counted_registry
    registry.create_job("job-1", kind="download", lane="primary", model_id="org/model")
    registry.update_job("job-1", status="running", progress={"bytes_done": 10})
    registry.get_job("job-1")
    registry.list_jobs()
    assert counter["opened"] > 0
    assert counter["closed"] == counter["opened"]


def test_delete_model_closes_its_connection(counted_registry):
    registry, counter = counted_registry
    registry.upsert_model("org/model", state="remote")
    before_delete = counter["opened"]
    registry.delete_model("org/model")
    assert counter["opened"] > before_delete
    assert counter["closed"] == counter["opened"]


def test_delete_job_removes_the_row_and_closes_its_connection(counted_registry):
    registry, counter = counted_registry
    registry.create_job("job-1", kind="acquire", lane="model-lifecycle",
                        model_id="org/model")
    before_delete = counter["opened"]
    removed = registry.delete_job("job-1")
    assert removed is True
    assert registry.get_job("job-1") is None
    assert counter["opened"] > before_delete
    assert counter["closed"] == counter["opened"]


def test_delete_job_on_an_unknown_id_returns_false(counted_registry):
    registry, _counter = counted_registry
    assert registry.delete_job("no-such-job") is False


def test_a_realistic_download_style_call_volume_leaks_nothing(counted_registry):
    """Mirrors afterimage/server/acquisition.py's download_snapshot(), which
    calls upsert_model once per file in a model with potentially hundreds of
    shards -- the exact call pattern that made the original leak add up
    fastest."""
    registry, counter = counted_registry
    for i in range(50):
        registry.upsert_model("org/big-model", bytes_done=i * 1000, state="downloading")
    assert counter["closed"] == counter["opened"]


def test_the_spy_fixture_actually_catches_the_original_bug(tmp_path, monkeypatch):
    """Sanity check on the test suite itself: reproduce the exact pre-fix
    pattern (a bare ``with self._connect() as connection:``, no closing())
    against the same counting spy, and confirm it reports a leak. If this
    test ever failed to detect a leak here, the tests above proving zero
    leaks would not be trustworthy."""
    counter = {"opened": 0, "closed": 0}
    real_connect = sqlite3.connect

    def spy_connect(*args, **kwargs):
        return _CountingConnection(real_connect(*args, **kwargs), counter)

    monkeypatch.setattr("sqlite3.connect", spy_connect)
    path = tmp_path / "leaky.sqlite3"

    def leaky_call():
        # The exact pattern every ModelRegistry method used before the fix.
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS t(x)")
            connection.execute("INSERT INTO t VALUES (1)")

    for _ in range(3):
        leaky_call()
    assert counter["opened"] == 3
    assert counter["closed"] == 0, (
        "the bare with-block closed its connections, which would mean this "
        "spy cannot actually detect the bug the fix above resolves")
