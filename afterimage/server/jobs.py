"""In-process job registry for compress/generate operations.

Each job runs its work function in a background thread against its own
runtime.control.JobControl, so the HTTP layer can return immediately with a
job id and poll (or open a WebSocket on) /api/jobs/{id} instead of holding
an HTTP request open for a multi-minute compression pass.

Deliberately in-process, not a task queue: this project targets one
consumer GPU running one job at a time, not a multi-worker deployment --
see the archived master plan's product scope. A real multi-node deployment would
replace this with Celery/RQ/etc without changing JobControl or the engine
at all, since JobControl already only depends on threading primitives.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
import uuid
from typing import Any, Callable, Optional

from afterimage.runtime.control import JobCancelled, JobControl
from afterimage.server.model_registry import model_registry

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Job:
    id: str
    kind: str
    model_id: str | None = None
    lane: str = "default"
    created_at: float = dataclasses.field(default_factory=time.time)
    status: str = "running"  # running | paused | done | error | cancelled
    progress: dict = dataclasses.field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    control: JobControl = dataclasses.field(default_factory=JobControl)


class JobRegistry:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._lanes: dict[str, threading.Lock] = {}

    def create(
        self,
        kind: str,
        fn: Callable[[JobControl], Any],
        *,
        model_id: str | None = None,
        lane: str = "default",
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12], kind=kind, model_id=model_id, lane=lane,
            status="queued",
        )
        model_registry.create_job(job.id, kind, lane, model_id)

        def on_progress(fields: dict) -> None:
            job.progress = {**job.progress, **fields}
            if job.status not in ("cancelled", "done", "error"):
                job.status = "paused" if job.control.is_paused else "running"
            model_registry.update_job(job.id, status=job.status, progress=job.progress)

        def on_state(status: str) -> None:
            if job.status in {"cancelled", "done", "error"}:
                return
            job.status = status
            model_registry.update_job(job.id, status=status, progress=job.progress)

        job.control.progress_callback = on_progress
        job.control.state_callback = on_state

        def run() -> None:
            lane_lock = None
            while lane_lock is None:
                if job.control.is_cancelled:
                    job.status = "cancelled"
                    model_registry.update_job(job.id, status=job.status)
                    return
                candidate = self._lane_lock(lane)
                if not candidate.acquire(timeout=0.1):
                    continue
                with self._lock:
                    current = self._lanes.get(lane)
                if current is not candidate:
                    candidate.release()
                    continue
                lane_lock = candidate
            try:
                if job.control.is_cancelled:
                    job.status = "cancelled"
                    model_registry.update_job(job.id, status=job.status)
                    return
                job.status = "running"
                model_registry.update_job(job.id, status=job.status)
                logger.info("job %s (%s) started", job.id, job.kind)
                try:
                    job.result = fn(job.control)
                    if job.control.is_cancelled:
                        raise JobCancelled()
                    job.status = "done"
                    model_registry.update_job(
                        job.id, status=job.status, progress=job.progress,
                        result=job.result,
                    )
                    logger.info("job %s (%s) done", job.id, job.kind)
                except JobCancelled:
                    job.status = "cancelled"
                    model_registry.update_job(job.id, status=job.status)
                    logger.info("job %s (%s) cancelled", job.id, job.kind)
                except Exception as e:  # noqa: BLE001 -- surfaced to the API caller
                    job.status = "error"
                    job.error = "%s: %s" % (type(e).__name__, e)
                    model_registry.update_job(
                        job.id, status=job.status, progress=job.progress,
                        error=job.error,
                    )
                    logger.exception("job %s (%s) failed", job.id, job.kind)
            finally:
                lane_lock.release()

        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=run, daemon=True).start()
        return job

    def _lane_lock(self, lane: str) -> threading.Lock:
        with self._lock:
            return self._lanes.setdefault(lane, threading.Lock())

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        saved = model_registry.get_job(job_id)
        if saved is None:
            return None
        return Job(
            id=saved["id"], kind=saved["kind"], model_id=saved["model_id"],
            lane=saved["lane"], created_at=saved["created_at"],
            status=saved["status"], progress=saved["progress"],
            result=saved["result"], error=saved["error"],
        )

    def list(self) -> list[Job]:
        return [self.get(row["id"]) for row in model_registry.list_jobs()]

    def pause(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.status not in {"queued", "running"}:
            return job
        job.control.pause()
        job.status = "pause_requested"
        model_registry.update_job(job.id, status=job.status, progress=job.progress)
        return job

    def resume(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.status not in {"paused", "pause_requested"}:
            return job
        job.control.resume()
        job.status = "running"
        model_registry.update_job(job.id, status=job.status, progress=job.progress)
        return job

    def cancel(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.status in {"done", "error", "cancelled"}:
            return job
        job.control.cancel()
        # Cancellation is a user-facing terminal state. A network read in a
        # third-party downloader may take time to unwind after the cooperative
        # signal. Leaving the API in ``cancelling`` strands the UI and every
        # job behind the same lane. Hugging Face partials are resumable, so a
        # download can release the public lane while its old request unwinds.
        # GPU work remains serialized.
        job.status = "cancelled"
        if job.lane == "model-lifecycle" and (
            job.progress.get("stage") in {None, "downloading"}
            or not job.progress
        ):
            with self._lock:
                self._lanes[job.lane] = threading.Lock()
        model_registry.update_job(job.id, status=job.status, progress=job.progress)
        return job


registry = JobRegistry()
