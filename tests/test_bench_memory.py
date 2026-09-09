from types import SimpleNamespace

import pytest

from afterimage.bench import memory


def test_smi_mib_converted_to_decimal_gb():
    report = memory.MemoryReport(None, 8192, 1024, None, 2)
    assert report.smi_delta_gb == pytest.approx(7168 * (1 << 20) / 1e9)


def test_missing_smi_baseline_is_not_a_measured_zero():
    report = memory.MemoryReport(None, 8192, None, None, 2)
    assert report.smi_delta_gb is None


def test_transient_smi_os_error_does_not_kill_sampling(monkeypatch):
    monkeypatch.setattr(memory.shutil, "which", lambda _name: "/bin/nvidia-smi")
    responses = iter([OSError(14, "Bad address"), SimpleNamespace(stdout="4096\n")])

    def run(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(memory.subprocess, "run", run)
    assert memory.nvidia_smi_used_mb() is None
    assert memory.nvidia_smi_used_mb() == 4096
