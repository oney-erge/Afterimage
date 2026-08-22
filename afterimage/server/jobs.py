"""In-process job registry for compress/generate operations.

Each job runs its work function in a background thread against its own
runtime.control.JobControl, so the HTTP layer can return immediately with a
job id and poll (or open a WebSocket on) /api/jobs/{id} instead of holding
an HTTP request open for a multi-minute compression pass.

Deliberately in-process, not a task queue: this project targets one
consumer GPU running one job at a time, not a multi-worker deployment --
see docs/archive/MASTER_PLAN.md's product scope. A real multi-node deployment would
replace this with Celery/RQ/etc without changing JobControl or the engine
at all, since JobControl already only depends on threading primitives.
"""
from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from typing import Any, Callable, Optional

from afterimage.runtime.control import JobCancelled, JobControl


@dataclasses.dataclass
class Job:
    id: str
    kind: str
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

    def create(self, kind: str, fn: Callable[[JobControl], Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)

        def on_progress(fields: dict) -> None:
            job.progress = fields
            if job.status not in ("cancelled", "done", "error"):
                job.status = "paused" if job.control.is_paused else "running"

        job.control.progress_callback = on_progress

        def run() -> None:
            try:
                job.result = fn(job.control)
                job.status = "done"
            except JobCancelled:
                job.status = "cancelled"
            except Exception as e:  # noqa: BLE001 -- surfaced to the API caller, not swallowed
                job.status = "error"
                job.error = "%s: %s" % (type(e).__name__, e)

        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=run, daemon=True).start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())


registry = JobRegistry()
