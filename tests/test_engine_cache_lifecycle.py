"""_EngineCache had no way to explicitly release a loaded model (the only
way to free its VRAM/RAM was to load a DIFFERENT model, which evicted the
old one as a side effect) and a real bug in the failure path: get()
assigned self._sm = None while evicting the previous engine but only set
self._key to the NEW key at the very end, after every fallible step
(store-existence check, tokenizer/config load, engine construction). If
any of those raised, self._key was left pointing at the OLD model while
self._sm was actually None -- the next request for that old model matched
_key, skipped reloading, and returned None, 500ing every request until a
THIRD, different model was requested.

These tests exercise both fixes without a real model, CUDA, or a real
compressed store: StreamingLosslessModel and the transformers loaders are
monkeypatched with cheap fakes, and only the cache's own bookkeeping is
under test.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from afterimage.runtime.config import EngineConfig
from afterimage.server import app as app_module


class _FakeEngine:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeConfig:
    pass


class _FakeTokenizer:
    pass


def _prepare_store(tmp_path, name):
    store = tmp_path / name
    store.mkdir()
    (store / "manifest.json").write_text("{}", encoding="utf-8")
    return store


def _patch_loading(monkeypatch, store_by_model, engine_factory):
    """Stub out every fallible step inside _EngineCache.get() with a cheap
    fake, so the test controls exactly which model (if any) fails to
    "load" without touching a real model, disk store, or transformers
    download."""
    monkeypatch.setattr(app_module, "_store_dir_for", lambda model_id: store_by_model[model_id])
    monkeypatch.setattr(app_module.model_registry, "get_model", lambda model_id: {})
    monkeypatch.setattr(app_module, "classify_config", lambda cfg: {"modality": "text"})

    import transformers
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        staticmethod(lambda *a, **k: _FakeConfig()))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _FakeTokenizer()))

    import afterimage.runtime.streaming_engine as streaming_engine
    monkeypatch.setattr(streaming_engine, "StreamingLosslessModel", engine_factory)


class TestFailedLoadDoesNotPoisonTheCache:
    def test_a_failed_load_leaves_the_cache_empty_not_pointing_at_the_old_model(
            self, tmp_path, monkeypatch):
        store_a = _prepare_store(tmp_path, "a")
        store_b = _prepare_store(tmp_path, "b")
        cache = app_module._EngineCache()

        def engine_factory(model_id, store_dir, **kwargs):
            if model_id == "b":
                raise RuntimeError("simulated failure loading b")
            return _FakeEngine()

        _patch_loading(monkeypatch, {"a": store_a, "b": store_b}, engine_factory)
        cfg = EngineConfig()

        sm_a, _tok_a = cache.get("a", cfg)
        assert sm_a is not None
        assert cache._key == ("a", cfg.fingerprint())

        with pytest.raises(RuntimeError):
            cache.get("b", cfg)

        # The bug this guards against: _key stuck at ("a", ...) while _sm
        # was None.
        assert cache._key is None
        assert cache._sm is None
        assert cache._loading_key is None  # cleared even on failure

    def test_the_previous_model_reloads_cleanly_after_a_failed_switch(
            self, tmp_path, monkeypatch):
        store_a = _prepare_store(tmp_path, "a")
        store_b = _prepare_store(tmp_path, "b")
        cache = app_module._EngineCache()
        reload_count = {"a": 0}

        def engine_factory(model_id, store_dir, **kwargs):
            if model_id == "b":
                raise RuntimeError("simulated failure loading b")
            reload_count["a"] += 1
            return _FakeEngine()

        _patch_loading(monkeypatch, {"a": store_a, "b": store_b}, engine_factory)
        cfg = EngineConfig()

        cache.get("a", cfg)
        with pytest.raises(RuntimeError):
            cache.get("b", cfg)

        # Previously: this returned (None, <b's tokenizer>) instead of
        # reloading -- every downstream caller (chat, /api/stats) would
        # then crash on a None engine.
        sm_a2, _tok = cache.get("a", cfg)
        assert sm_a2 is not None
        assert reload_count["a"] == 2
        assert cache._key == ("a", cfg.fingerprint())


class TestUnload:
    def test_unload_closes_the_engine_and_clears_every_field(self, tmp_path, monkeypatch):
        store_a = _prepare_store(tmp_path, "a")
        cache = app_module._EngineCache()
        engines = []

        def engine_factory(model_id, store_dir, **kwargs):
            engine = _FakeEngine()
            engines.append(engine)
            return engine

        _patch_loading(monkeypatch, {"a": store_a}, engine_factory)
        cfg = EngineConfig()
        cache.get("a", cfg)
        cache._last_completion_len = 42

        unloaded = cache.unload()

        assert unloaded == "a"
        assert engines[0].closed is True
        assert cache._key is None
        assert cache._sm is None
        assert cache._tok is None
        assert cache._last_completion_len is None

    def test_unload_also_releases_the_draft_model(self, tmp_path, monkeypatch):
        store_a = _prepare_store(tmp_path, "a")
        cache = app_module._EngineCache()
        _patch_loading(monkeypatch, {"a": store_a}, lambda *a, **k: _FakeEngine())

        import afterimage.runtime.streaming_engine as streaming_engine
        monkeypatch.setattr(streaming_engine, "load_draft_model",
                            lambda draft_model_id, device: object())

        cfg = EngineConfig()
        cache.get("a", cfg)
        cache.get_draft("draft-x", "cpu")
        assert cache._draft is not None

        cache.unload()
        assert cache._draft is None
        assert cache._draft_key is None

    def test_unload_on_an_empty_cache_is_a_harmless_no_op(self):
        cache = app_module._EngineCache()
        assert cache.unload() is None
        assert cache._sm is None


class TestUnloadEndpoint:
    @pytest.fixture
    def isolated_client(self, monkeypatch):
        monkeypatch.setattr(app_module, "_engine_cache", app_module._EngineCache())
        return TestClient(app_module.app)

    def test_returns_409_when_the_requested_model_is_not_loaded(self, isolated_client):
        resp = isolated_client.post("/api/models/org%2Fmodel/unload")
        assert resp.status_code == 409

    def test_returns_409_when_a_different_model_is_loaded(self, isolated_client):
        app_module._engine_cache._key = ("org/other", "fingerprint")
        app_module._engine_cache._sm = _FakeEngine()
        resp = isolated_client.post("/api/models/org%2Fmodel/unload")
        assert resp.status_code == 409
        assert app_module._engine_cache._sm is not None  # untouched

    def test_unloads_the_currently_loaded_model_and_frees_it(self, isolated_client):
        engine = _FakeEngine()
        app_module._engine_cache._key = ("org/model", "fingerprint")
        app_module._engine_cache._sm = engine
        resp = isolated_client.post("/api/models/org%2Fmodel/unload")
        assert resp.status_code == 200
        assert resp.json() == {"model_id": "org/model", "unloaded": True}
        assert engine.closed is True
        assert app_module._engine_cache._sm is None
        assert app_module._engine_cache._key is None


class TestHealthReportsLoadingState:
    def test_health_shows_loading_model_while_a_load_is_in_flight(self, monkeypatch):
        monkeypatch.setattr(app_module, "_engine_cache", app_module._EngineCache())
        app_module._engine_cache._loading_key = ("org/model", "fingerprint")
        client = TestClient(app_module.app)
        payload = client.get("/health").json()
        assert payload["loading_model"] == "org/model"
        assert payload["loaded_model"] is None  # still not committed

    def test_health_reports_no_loading_model_when_idle(self, monkeypatch):
        monkeypatch.setattr(app_module, "_engine_cache", app_module._EngineCache())
        client = TestClient(app_module.app)
        assert client.get("/health").json()["loading_model"] is None


class TestModelsListReportsLoadedFlag:
    def test_the_currently_loaded_model_is_flagged_loaded_true(self, monkeypatch):
        monkeypatch.setattr(app_module, "_engine_cache", app_module._EngineCache())
        app_module._engine_cache._key = ("org/model", "fingerprint")
        app_module._engine_cache._sm = _FakeEngine()
        monkeypatch.setattr(
            app_module.model_registry, "list_models",
            lambda: [{"model_id": "org/model", "state": "ready", "updated_at": "2026-01-01",
                     "metadata": {}}])
        monkeypatch.setattr(app_module, "_scan_store_root", lambda root, by_id, mtimes: None)
        monkeypatch.setattr(app_module, "_extra_store_roots", lambda: [])
        client = TestClient(app_module.app)
        payload = client.get("/api/models").json()
        assert payload["models"][0]["loaded"] is True

    def test_a_different_ready_model_is_flagged_loaded_false(self, monkeypatch):
        monkeypatch.setattr(app_module, "_engine_cache", app_module._EngineCache())
        monkeypatch.setattr(
            app_module.model_registry, "list_models",
            lambda: [{"model_id": "org/model", "state": "ready", "updated_at": "2026-01-01",
                     "metadata": {}}])
        monkeypatch.setattr(app_module, "_scan_store_root", lambda root, by_id, mtimes: None)
        monkeypatch.setattr(app_module, "_extra_store_roots", lambda: [])
        client = TestClient(app_module.app)
        payload = client.get("/api/models").json()
        assert payload["models"][0]["loaded"] is False


class TestRemoveModelEvictsALoadedEngineFirst:
    @pytest.fixture
    def isolated_client(self, tmp_path, monkeypatch):
        store_root = tmp_path / "stores"
        store_root.mkdir()
        monkeypatch.setattr(app_module, "DEFAULT_STORE_ROOT", store_root)
        monkeypatch.setattr(app_module, "_store_dir_for",
                            lambda model_id: store_root / model_id.replace("/", "__"))
        monkeypatch.setattr(app_module, "registry", type("R", (), {"list": staticmethod(list)})())
        monkeypatch.setattr(app_module.model_registry, "delete_model", lambda model_id: True)
        monkeypatch.setattr(app_module, "_engine_cache", app_module._EngineCache())
        return TestClient(app_module.app), store_root

    def test_deleting_the_loaded_model_closes_its_engine_before_rmtree(
            self, isolated_client):
        client, store_root = isolated_client
        store = store_root / "org__model"
        store.mkdir()
        (store / "weights.bin").write_bytes(b"x")

        engine = _FakeEngine()
        app_module._engine_cache._key = ("org/model", "fingerprint")
        app_module._engine_cache._sm = engine

        resp = client.delete(
            "/api/models/org%2Fmodel?confirm_model_id=org%2Fmodel")

        assert resp.status_code == 200
        assert engine.closed is True  # closed BEFORE the store was removed
        assert not store.exists()
        assert app_module._engine_cache._sm is None
        assert app_module._engine_cache._key is None

    def test_deleting_an_unrelated_model_does_not_touch_the_loaded_engine(
            self, isolated_client):
        client, store_root = isolated_client
        store = store_root / "org__other"
        store.mkdir()

        engine = _FakeEngine()
        app_module._engine_cache._key = ("org/model", "fingerprint")
        app_module._engine_cache._sm = engine

        resp = client.delete(
            "/api/models/org%2Fother?confirm_model_id=org%2Fother")

        assert resp.status_code == 200
        assert engine.closed is False
        assert app_module._engine_cache._sm is engine
