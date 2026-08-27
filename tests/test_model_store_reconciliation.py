"""A real, on-disk compressed store built outside DEFAULT_STORE_ROOT (e.g.
this project's own GPU benchmark tooling, which hard-codes a WSL2 path like
/root/afterimage/store_14b, entirely outside ~/.afterimage/stores) was
invisible to /api/models: nothing ever scanned it, so a model the user knew
existed on disk simply never appeared in the web UI's library. These tests
build a real manifest.json on disk and verify list_models() actually finds
it through AFTERIMAGE_EXTRA_STORE_ROOTS -- this is the same reconciliation
mechanism DEFAULT_STORE_ROOT itself already used, generalized to additional
configurable roots rather than a single hard-coded one.

Every test uses its own ModelRegistry pointed at a tmp_path database and
monkeypatches it onto afterimage.server.app's module-level singleton, so
none of these ever read or write the real user registry at
~/.afterimage/state/afterimage.sqlite3.
"""
from __future__ import annotations

import json
import os

import pytest

from afterimage.server import app as app_module
from afterimage.server.model_registry import ModelRegistry


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    registry = ModelRegistry(tmp_path / "registry.sqlite3")
    monkeypatch.setattr(app_module, "model_registry", registry)
    monkeypatch.setattr(app_module, "DEFAULT_STORE_ROOT", tmp_path / "default-store-root")
    monkeypatch.delenv("AFTERIMAGE_EXTRA_STORE_ROOTS", raising=False)
    return registry


def _write_manifest(store_dir, model_id, orig_bytes=100, comp_bytes=50):
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / "manifest.json").write_text(json.dumps({
        "model_id": model_id, "total_orig_bytes": orig_bytes,
        "total_comp_bytes": comp_bytes, "ratio": orig_bytes / comp_bytes,
    }), encoding="utf-8")


def test_a_store_outside_default_root_is_invisible_without_the_env_var(
        isolated_registry, tmp_path):
    """The exact bug being fixed, reproduced first so the fix below is
    proven to matter, not just proven to not crash."""
    extra_root = tmp_path / "wsl2-benchmark-stores"
    _write_manifest(extra_root / "Qwen__Qwen3-14B", "Qwen/Qwen3-14B")

    result = app_module.list_models()
    assert "Qwen/Qwen3-14B" not in [m["model_id"] for m in result["models"]]


def test_a_store_under_an_extra_root_is_found_and_reported_ready(
        isolated_registry, tmp_path, monkeypatch):
    extra_root = tmp_path / "wsl2-benchmark-stores"
    _write_manifest(extra_root / "Qwen__Qwen3-14B", "Qwen/Qwen3-14B",
                    orig_bytes=28_000_000_000, comp_bytes=19_600_000_000)
    monkeypatch.setenv("AFTERIMAGE_EXTRA_STORE_ROOTS", str(extra_root))

    result = app_module.list_models()
    by_id = {m["model_id"]: m for m in result["models"]}
    assert "Qwen/Qwen3-14B" in by_id
    assert by_id["Qwen/Qwen3-14B"]["state"] == "ready"
    assert by_id["Qwen/Qwen3-14B"]["orig_gb"] == pytest.approx(28.0)


def test_multiple_extra_roots_are_all_scanned(isolated_registry, tmp_path, monkeypatch):
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    _write_manifest(root_a / "Org__ModelA", "Org/ModelA")
    _write_manifest(root_b / "Org__ModelB", "Org/ModelB")
    monkeypatch.setenv("AFTERIMAGE_EXTRA_STORE_ROOTS",
                       os.pathsep.join([str(root_a), str(root_b)]))

    ids = {m["model_id"] for m in app_module.list_models()["models"]}
    assert {"Org/ModelA", "Org/ModelB"} <= ids


def test_default_root_and_extra_roots_are_both_scanned(
        isolated_registry, tmp_path, monkeypatch):
    _write_manifest(tmp_path / "default-store-root" / "Org__InDefault", "Org/InDefault")
    extra_root = tmp_path / "extra"
    _write_manifest(extra_root / "Org__InExtra", "Org/InExtra")
    monkeypatch.setenv("AFTERIMAGE_EXTRA_STORE_ROOTS", str(extra_root))

    ids = {m["model_id"] for m in app_module.list_models()["models"]}
    assert {"Org/InDefault", "Org/InExtra"} <= ids


def test_a_missing_or_bogus_extra_root_does_not_break_the_scan(
        isolated_registry, tmp_path, monkeypatch):
    real_root = tmp_path / "real"
    _write_manifest(real_root / "Org__Real", "Org/Real")
    monkeypatch.setenv(
        "AFTERIMAGE_EXTRA_STORE_ROOTS",
        os.pathsep.join([str(tmp_path / "does-not-exist"), str(real_root)]))

    ids = {m["model_id"] for m in app_module.list_models()["models"]}
    assert "Org/Real" in ids


def test_empty_env_var_scans_nothing_extra(isolated_registry, tmp_path, monkeypatch):
    monkeypatch.setenv("AFTERIMAGE_EXTRA_STORE_ROOTS", "")
    result = app_module.list_models()
    assert result["models"] == []


def test_a_directory_without_a_manifest_is_silently_skipped(
        isolated_registry, tmp_path, monkeypatch):
    extra_root = tmp_path / "extra"
    (extra_root / "not-a-store").mkdir(parents=True)
    monkeypatch.setenv("AFTERIMAGE_EXTRA_STORE_ROOTS", str(extra_root))
    result = app_module.list_models()
    assert result["models"] == []


def test_a_corrupt_manifest_json_is_skipped_not_a_crash(
        isolated_registry, tmp_path, monkeypatch):
    extra_root = tmp_path / "extra"
    store_dir = extra_root / "Org__Broken"
    store_dir.mkdir(parents=True)
    (store_dir / "manifest.json").write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv("AFTERIMAGE_EXTRA_STORE_ROOTS", str(extra_root))
    result = app_module.list_models()  # must not raise
    assert result["models"] == []
