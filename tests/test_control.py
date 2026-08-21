import threading
import time

import pytest

from afterimage.runtime.control import JobCancelled, JobControl


def test_checkpoint_does_not_block_when_not_paused():
    ctl = JobControl()
    t0 = time.perf_counter()
    ctl.checkpoint()
    assert time.perf_counter() - t0 < 0.1


def test_pause_blocks_checkpoint_until_resume():
    ctl = JobControl()
    ctl.pause()
    assert ctl.is_paused

    unblocked = threading.Event()

    def worker():
        ctl.checkpoint()
        unblocked.set()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    time.sleep(0.2)
    assert not unblocked.is_set(), "checkpoint() returned while still paused"

    ctl.resume()
    th.join(timeout=2)
    assert unblocked.is_set(), "checkpoint() never unblocked after resume()"
    assert not ctl.is_paused


def test_cancel_raises_from_checkpoint():
    ctl = JobControl()
    ctl.cancel()
    assert ctl.is_cancelled
    with pytest.raises(JobCancelled):
        ctl.checkpoint()


def test_cancel_unblocks_a_paused_checkpoint_instead_of_hanging_forever():
    """Cancelling while paused must not leave a waiting thread stuck
    forever -- cancel() also releases the pause gate so checkpoint() wakes
    up and raises, rather than waiting on a resume() that will never come.
    """
    ctl = JobControl()
    ctl.pause()

    raised = threading.Event()

    def worker():
        try:
            ctl.checkpoint()
        except JobCancelled:
            raised.set()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    time.sleep(0.1)
    ctl.cancel()
    th.join(timeout=2)
    assert raised.is_set(), "checkpoint() never raised JobCancelled after cancel() while paused"


def test_report_calls_progress_callback_with_kwargs():
    seen = []
    ctl = JobControl(progress_callback=seen.append)
    ctl.report(n=3, total=10)
    assert seen == [{"n": 3, "total": 10}]


def test_report_is_a_safe_noop_without_a_callback():
    ctl = JobControl()
    ctl.report(anything="fine")  # must not raise
