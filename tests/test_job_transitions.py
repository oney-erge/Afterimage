"""Controlled races between job controls and terminal outcomes."""

from __future__ import annotations

import threading

import pytest

from afterimage.runtime.control import JobControl
from afterimage.server import jobs as jobs_module
from afterimage.server.jobs import JobRegistry
from afterimage.server.model_registry import ModelRegistry


@pytest.mark.parametrize("outcome", ["done", "error"])
def test_cancel_and_completion_agree_at_the_terminal_boundary(tmp_path, monkeypatch, outcome):
    storage = ModelRegistry(tmp_path / "registry.sqlite3")
    monkeypatch.setattr(jobs_module, "model_registry", storage)
    registry = JobRegistry()
    work_returned = threading.Event()
    checked = threading.Event()
    release = threading.Event()
    cancel_started = threading.Event()
    cancel_returned = threading.Event()
    result = {}
    original = JobControl.is_cancelled.fget

    def inspect_cancel(control):
        value = original(control)
        if work_returned.is_set() and not checked.is_set():
            checked.set()
            if not release.wait(5):
                raise TimeoutError("test did not release the terminal boundary")
        return value

    monkeypatch.setattr(JobControl, "is_cancelled", property(inspect_cancel))

    def work(_control):
        work_returned.set()
        if outcome == "error":
            raise RuntimeError("controlled failure")
        return "completed result"

    job = registry.create("generate", work, lane="gpu")

    def cancel():
        cancel_started.set()
        result["status"] = registry.cancel(job.id).status
        cancel_returned.set()

    thread = threading.Thread(target=cancel)
    try:
        assert checked.wait(5)
        thread.start()
        assert cancel_started.wait(5)
        returned_before_completion = cancel_returned.wait(1)
    finally:
        release.set()
    assert job.worker_finished.wait(5)
    assert cancel_returned.wait(5)
    thread.join(timeout=5)
    assert not thread.is_alive()
    expected = "cancelled" if returned_before_completion else outcome
    assert result["status"] == expected
    assert registry.get(job.id).status == expected
    assert storage.get_job(job.id)["status"] == expected


def test_cancel_before_work_returns_preserves_cancelled_outcome(tmp_path, monkeypatch):
    storage = ModelRegistry(tmp_path / "registry.sqlite3")
    monkeypatch.setattr(jobs_module, "model_registry", storage)
    registry = JobRegistry()
    entered, release = threading.Event(), threading.Event()

    def work(_control):
        entered.set()
        assert release.wait(5)
        return "late result"

    job = registry.create("generate", work, lane="gpu")
    try:
        assert entered.wait(5)
        assert registry.cancel(job.id).status == "cancelled"
    finally:
        release.set()
    assert job.worker_finished.wait(5)
    assert registry.get(job.id).status == "cancelled"
    assert storage.get_job(job.id)["status"] == "cancelled"


def test_late_progress_cannot_restore_a_cancelled_job(tmp_path, monkeypatch):
    storage = ModelRegistry(tmp_path / "registry.sqlite3")
    monkeypatch.setattr(jobs_module, "model_registry", storage)
    registry = JobRegistry()
    entered = threading.Event()
    release_report = threading.Event()
    report_finished = threading.Event()
    release_work = threading.Event()
    cancel_started = threading.Event()
    cancel_returned = threading.Event()
    report_started = threading.Event()
    original = JobControl.is_paused.fget

    def inspect_pause(control):
        value = original(control)
        if report_started.is_set() and not entered.is_set():
            entered.set()
            if not release_report.wait(5):
                raise TimeoutError("test did not release progress")
        return value

    monkeypatch.setattr(JobControl, "is_paused", property(inspect_pause))

    def work(control):
        report_started.set()
        control.report(stage="working")
        report_finished.set()
        assert release_work.wait(5)
        control.checkpoint()

    job = registry.create("generate", work, lane="gpu")

    def cancel():
        cancel_started.set()
        registry.cancel(job.id)
        cancel_returned.set()

    thread = threading.Thread(target=cancel)
    try:
        assert entered.wait(5)
        thread.start()
        assert cancel_started.wait(5)
        cancel_returned.wait(1)
        release_report.set()
        assert report_finished.wait(5)
        assert cancel_returned.wait(5)
        assert job.status == "cancelled"
        assert storage.get_job(job.id)["status"] == "cancelled"
    finally:
        release_report.set()
        release_work.set()
    assert job.worker_finished.wait(5)
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_api_reports_cancelled_while_worker_cleans_up(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from afterimage.server import app as app_module

    storage = ModelRegistry(tmp_path / "registry.sqlite3")
    monkeypatch.setattr(jobs_module, "model_registry", storage)
    registry = JobRegistry()
    monkeypatch.setattr(app_module, "registry", registry)
    monkeypatch.setattr(app_module, "model_registry", storage)
    entered, release = threading.Event(), threading.Event()

    def work(_control):
        entered.set()
        assert release.wait(5)
        return {"text": "late result"}

    job = registry.create("generate", work, lane="gpu")
    try:
        assert entered.wait(5)
        client = TestClient(app_module.app)
        response = client.post(f"/api/jobs/{job.id}/cancel")
        assert response.status_code == 200 and response.json()["status"] == "cancelled"
        stopping = client.get(f"/api/jobs/{job.id}").json()
        assert stopping["status"] == "cancelled" and stopping["cleanup_pending"] is True
        assert stopping["result"] is None
    finally:
        release.set()
    assert job.worker_finished.wait(5)
    finished = client.get(f"/api/jobs/{job.id}").json()
    assert finished["status"] == "cancelled" and finished["cleanup_pending"] is False
    assert finished["result"] is None
    assert storage.get_job(job.id)["status"] == "cancelled"


def test_progress_does_not_claim_pause_before_a_worker_checkpoint(tmp_path, monkeypatch):
    storage = ModelRegistry(tmp_path / "registry.sqlite3")
    monkeypatch.setattr(jobs_module, "model_registry", storage)
    registry = JobRegistry()
    entered = threading.Event()
    release_report = threading.Event()
    report_finished = threading.Event()
    release_checkpoint = threading.Event()
    pause_observed = threading.Event()
    update = storage.update_job

    def record_update(job_id, **fields):
        update(job_id, **fields)
        if fields.get("status") == "paused":
            pause_observed.set()

    monkeypatch.setattr(storage, "update_job", record_update)

    def work(control):
        entered.set()
        assert release_report.wait(5)
        control.report(stage="working")
        report_finished.set()
        assert release_checkpoint.wait(5)
        control.checkpoint()
        return "finished"

    job = registry.create("generate", work, lane="gpu")
    try:
        assert entered.wait(5)
        assert registry.pause(job.id).status == "pause_requested"
        release_report.set()
        assert report_finished.wait(5)
        assert registry.get(job.id).status == "pause_requested"
        assert storage.get_job(job.id)["status"] == "pause_requested"
        release_checkpoint.set()
        assert pause_observed.wait(5)
        assert registry.get(job.id).status == "paused"
        assert registry.resume(job.id).status == "running"
    finally:
        release_report.set()
        release_checkpoint.set()
        registry.resume(job.id)
    assert job.worker_finished.wait(5)
    assert storage.get_job(job.id)["status"] == "done"


@pytest.mark.parametrize("stale_state", ["paused", "running"])
def test_stale_checkpoint_callback_cannot_reverse_a_newer_control(tmp_path, monkeypatch, stale_state):
    storage = ModelRegistry(tmp_path / "registry.sqlite3")
    monkeypatch.setattr(jobs_module, "model_registry", storage)
    registry = JobRegistry()
    entered = threading.Event()
    checkpoint = threading.Event()
    paused = threading.Event()
    callback_entered = threading.Event()
    release_callback = threading.Event()
    callback_done = threading.Event()
    release_worker = threading.Event()

    def work(control):
        entered.set()
        assert checkpoint.wait(5)
        control.checkpoint()
        return "finished"

    job = registry.create("generate", work, lane="gpu")
    original = job.control.state_callback

    def gate(status):
        if status == stale_state:
            callback_entered.set()
            assert release_callback.wait(5)
        original(status)
        if status == "paused":
            paused.set()
        if status == stale_state:
            callback_done.set()
            assert release_worker.wait(5)

    job.control.state_callback = gate
    try:
        assert entered.wait(5)
        assert registry.pause(job.id).status == "pause_requested"
        checkpoint.set()
        if stale_state == "running":
            assert paused.wait(5)
            assert registry.resume(job.id).status == "running"
            assert callback_entered.wait(5)
            assert registry.pause(job.id).status == "pause_requested"
            expected = "pause_requested"
        else:
            assert callback_entered.wait(5)
            assert registry.resume(job.id).status == "running"
            expected = "running"
        release_callback.set()
        assert callback_done.wait(5)
        assert registry.get(job.id).status == expected
        assert storage.get_job(job.id)["status"] == expected
    finally:
        checkpoint.set()
        release_callback.set()
        registry.resume(job.id)
        release_worker.set()
    assert job.worker_finished.wait(5)
