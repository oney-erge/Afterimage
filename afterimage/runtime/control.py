"""Pause / resume / cancel for long-running engine operations, plus a
structured progress callback -- what the FastAPI server's job control and
WebSocket progress stream are built on.

A layer-streaming loop has a natural, cheap pause point: the boundary
between one layer's work and the next. checkpoint() is called there (inside
StreamingLosslessModel._load_layer) and once per generated token/sweep
(inside generate_greedy/generate_speculative/compress_model_to_disk's
per-tensor loop), so pausing takes effect within about one layer's worth of
I/O -- seconds, not "however long the whole generation takes" -- rather than
only at the very end of a request.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional


class JobCancelled(Exception):
    """Raised out of checkpoint() when a caller requested cancellation.
    Callers that want cleanup on cancellation should catch this, not a bare
    Exception -- it is the one exception this module raises deliberately."""


class JobControl:
    """One per running job. NOT shared between concurrent jobs: pausing or
    cancelling one job must never affect another, so callers should
    construct a fresh JobControl per StreamingLosslessModel /
    compress_model_to_disk invocation rather than reusing one instance.
    """

    def __init__(self, progress_callback: Optional[Callable[[dict], None]] = None):
        self._paused = threading.Event()
        self._paused.set()  # set = "go"; clear = "paused"
        self._cancelled = threading.Event()
        self.progress_callback = progress_callback

    def pause(self) -> None:
        self._paused.clear()

    def resume(self) -> None:
        self._paused.set()

    def cancel(self) -> None:
        self._cancelled.set()
        self._paused.set()  # unblock any pending wait so cancellation lands promptly

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def checkpoint(self) -> None:
        """Call at a natural pause boundary. Blocks while paused; raises
        JobCancelled if cancelled (including while blocked on a pause)."""
        self._paused.wait()
        if self._cancelled.is_set():
            raise JobCancelled()

    def report(self, **fields) -> None:
        """Structured progress, forwarded to progress_callback if one was
        given (e.g. the server pushes this over a WebSocket). A no-op
        without a callback, so engine code can call this unconditionally."""
        if self.progress_callback is not None:
            self.progress_callback(fields)
