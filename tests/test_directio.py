import platform

import torch

from afterimage.runtime import directio
from afterimage.runtime.tiers import Tier, TieredStore


def test_raw_tensor_roundtrip_preserves_values(tmp_path):
    torch.manual_seed(0)
    t = torch.randn(37, 53)
    path = tmp_path / "w.bin"
    directio.write_tensor_raw(path, t)
    back, result = directio.read_tensor_raw(path)
    assert back.shape == t.shape
    assert torch.allclose(back, t, atol=1e-6)
    assert result.bytes_read >= t.numel() * 4


def test_direct_io_availability_matches_platform():
    avail = directio.direct_io_available()
    if platform.system() != "Linux":
        assert avail is False, "O_DIRECT must not claim availability off Linux"


def test_store_reports_io_mode_honestly(tmp_path):
    """A store that fell back to buffered reads must not be presentable as a
    storage measurement -- this is the guard that prevents the 7x
    cache-inflated numbers measured on the dev rig from being reported."""
    store = TieredStore(tmp_path / "nvme", direct_io=True)
    torch.manual_seed(1)
    store.write_nvme("w", torch.randn(16, 16))
    _ = store.get("w")

    report = store.io_mode_report()
    if platform.system() == "Linux" and directio.direct_io_available():
        assert "O_DIRECT" in report or "MIXED" in report
    else:
        # off Linux the read must fall back, and the store must SAY so
        assert "NOT a valid storage measurement" in report
        assert store.direct_io_effective is False


def test_buffered_store_never_claims_validity(tmp_path):
    store = TieredStore(tmp_path / "nvme", direct_io=False)
    torch.manual_seed(2)
    store.write_nvme("w", torch.randn(8, 8))
    _ = store.get("w")
    assert store.direct_io_effective is False
    assert "NOT a valid storage measurement" in store.io_mode_report()


def test_direct_io_store_roundtrip_matches_buffered_store(tmp_path):
    torch.manual_seed(3)
    W = torch.randn(24, 31)

    buffered = TieredStore(tmp_path / "buf", direct_io=False)
    buffered.write_nvme("w", W)
    got_buf = buffered.get("w")

    direct = TieredStore(tmp_path / "dir", direct_io=True)
    direct.write_nvme("w", W)
    got_dir = direct.get("w")

    assert torch.allclose(got_buf, got_dir, atol=1e-6)
    assert torch.allclose(got_dir, W, atol=1e-6)


def test_byte_accounting_agrees_between_modes(tmp_path):
    """GB/token is the project's primary metric, so the two storage paths
    must count bytes identically or the metric is not comparable across
    configurations."""
    torch.manual_seed(4)
    W = torch.randn(64, 64)

    buffered = TieredStore(tmp_path / "buf2", direct_io=False)
    buffered.write_nvme("w", W)
    buffered.reset_stats()
    _ = buffered.get("w")

    direct = TieredStore(tmp_path / "dir2", direct_io=True)
    direct.write_nvme("w", W)
    direct.reset_stats()
    _ = direct.get("w")

    assert buffered.stats[Tier.NVME].bytes_read == direct.stats[Tier.NVME].bytes_read
