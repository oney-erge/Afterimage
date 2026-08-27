"""Activity feed rows (backed by the durable ``jobs`` table -- see
JobRegistry.list() in afterimage/server/jobs.py, which always reads through
model_registry.list_jobs()) had no way to be cleared once finished,
interrupted, or cancelled: pause/resume/cancel/retry existed, delete did
not. A user stuck with old failed downloads permanently cluttering the
Activity feed had no in-product way to remove them.

These tests isolate both the JobRegistry (in-memory) and its backing
ModelRegistry (SQLite) from the real singletons so nothing here touches
~/.afterimage/state/afterimage.sqlite3.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from afterimage.server import app as app_module
from afterimage.server import jobs as jobs_module
from afterimage.server.jobs import JobRegistry
from afterimage.server.model_registry import ModelRegistry


@pytest.fixture
def isolated_job_registry(tmp_path, monkeypatch):
    model_registry = ModelRegistry(tmp_path / "registry.sqlite3")
    monkeypatch.setattr(jobs_module, "model_registry", model_registry)
    job_registry = JobRegistry()
    return job_registry, model_registry


def _terminal_job(job_registry, model_registry, status, job_id="job-1",
                  kind="acquire", model_id="org/model"):
    model_registry.create_job(job_id, kind=kind, lane="model-lifecycle",
                              model_id=model_id)
    model_registry.update_job(job_id, status=status)
    return job_id


class TestJobRegistryDelete:
    def test_deletes_a_terminal_job(self, isolated_job_registry):
        job_registry, model_registry = isolated_job_registry
        job_id = _terminal_job(job_registry, model_registry, "interrupted")
        assert job_registry.delete(job_id) is True
        assert model_registry.get_job(job_id) is None

    @pytest.mark.parametrize("status", ["done", "error", "cancelled", "interrupted"])
    def test_deletes_every_terminal_status(self, isolated_job_registry, status):
        job_registry, model_registry = isolated_job_registry
        job_id = _terminal_job(job_registry, model_registry, status)
        assert job_registry.delete(job_id) is True

    @pytest.mark.parametrize(
        "status", ["queued", "running", "paused", "pause_requested", "cancelling"])
    def test_refuses_to_delete_an_active_job(self, isolated_job_registry, status):
        job_registry, model_registry = isolated_job_registry
        job_id = _terminal_job(job_registry, model_registry, status)
        assert job_registry.delete(job_id) is False
        assert model_registry.get_job(job_id) is not None

    def test_checks_persisted_status_even_when_not_held_in_memory(
            self, isolated_job_registry):
        """The gap this test exists for: a job whose background thread no
        longer exists (e.g. left over from a server restart) is not in
        JobRegistry._jobs at all -- delete() must still consult its
        persisted status via self.get(), not skip the active check just
        because there is no in-memory Job object."""
        job_registry, model_registry = isolated_job_registry
        job_id = _terminal_job(job_registry, model_registry, "running")
        assert job_id not in job_registry._jobs  # never touched .create()
        assert job_registry.delete(job_id) is False

    def test_deleting_an_unknown_job_id_is_a_harmless_no_op(self, isolated_job_registry):
        job_registry, _model_registry = isolated_job_registry
        assert job_registry.delete("no-such-job") is False


@pytest.fixture
def isolated_client(tmp_path, monkeypatch):
    model_registry = ModelRegistry(tmp_path / "registry.sqlite3")
    monkeypatch.setattr(jobs_module, "model_registry", model_registry)
    monkeypatch.setattr(app_module, "model_registry", model_registry)
    job_registry = JobRegistry()
    monkeypatch.setattr(app_module, "registry", job_registry)
    return TestClient(app_module.app), job_registry, model_registry


class TestDeleteJobsEndpoint:
    def test_deletes_a_terminal_job_and_returns_200(self, isolated_client):
        client, job_registry, model_registry = isolated_client
        job_id = _terminal_job(job_registry, model_registry, "cancelled")
        resp = client.delete("/api/jobs/%s" % job_id)
        assert resp.status_code == 200
        assert resp.json() == {"deleted": True, "id": job_id}
        assert client.get("/api/jobs").json()["jobs"] == []

    def test_refuses_an_active_job_with_409(self, isolated_client):
        client, job_registry, model_registry = isolated_client
        job_id = _terminal_job(job_registry, model_registry, "running")
        resp = client.delete("/api/jobs/%s" % job_id)
        assert resp.status_code == 409

    def test_unknown_job_id_is_404(self, isolated_client):
        client, _job_registry, _model_registry = isolated_client
        resp = client.delete("/api/jobs/does-not-exist")
        assert resp.status_code == 404
