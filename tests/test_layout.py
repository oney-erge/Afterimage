import torch

from afterimage.runtime.layout import BitPlaneLadder, read_ladder, write_ladder
from afterimage.runtime.tiers import Tier, TieredStore



import pytest

pytestmark = pytest.mark.archive  # Phase-0 subspace-activation-cache branch, killed -- see docs/archive/README.md
def test_reconstruction_error_shrinks_monotonically_with_more_planes():
    torch.manual_seed(0)
    W = torch.randn(20, 30)
    ladder = BitPlaneLadder(base_bits=2, resid_bits=2, n_residual_planes=4)
    codes, specs, scale = ladder.encode(W)

    errors = []
    for k in range(1, len(codes) + 1):
        recon = BitPlaneLadder.decode(codes, specs, scale, up_to_plane=k)
        err = (recon - W).norm().item() / W.norm().item()
        errors.append(err)

    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] * 1.001, f"error did not shrink at plane {i}: {errors}"
    assert errors[-1] < errors[0] * 0.5, f"full ladder not meaningfully better than base alone: {errors}"


def test_full_ladder_reconstruction_is_close_to_original():
    torch.manual_seed(1)
    W = torch.randn(16, 16)
    ladder = BitPlaneLadder(base_bits=3, resid_bits=3, n_residual_planes=3)
    codes, specs, scale = ladder.encode(W)
    recon = BitPlaneLadder.decode(codes, specs, scale)
    rel_err = (recon - W).norm().item() / W.norm().item()
    assert rel_err < 0.05, f"12-bit-equivalent ladder should be reasonably tight: {rel_err}"


def test_bits_for_ratio_monotone_in_rho():
    ladder = BitPlaneLadder(base_bits=2, resid_bits=2, n_residual_planes=4)
    n_full = ladder.bits_for_ratio(1.0)
    n_small = ladder.bits_for_ratio(0.05)
    n_tiny = ladder.bits_for_ratio(0.001)
    assert n_small <= n_full
    assert n_tiny <= n_small
    assert n_tiny >= 1


def test_partial_read_touches_fewer_bytes_than_full_read(tmp_path):
    torch.manual_seed(2)
    W = torch.randn(40, 40)
    store = TieredStore(tmp_path / "nvme")
    ladder = BitPlaneLadder(base_bits=2, resid_bits=2, n_residual_planes=3)
    specs = write_ladder(store, "W", W, ladder)

    store.reset_stats()
    _ = read_ladder(store, "W", specs, n_planes=1)
    bytes_partial = store.stats[Tier.NVME].bytes_read

    store.reset_stats()
    _ = read_ladder(store, "W", specs, n_planes=len(specs))
    bytes_full = store.stats[Tier.NVME].bytes_read

    assert bytes_partial < bytes_full, "escalation should read strictly fewer bytes for fewer planes"


def test_write_then_read_roundtrip_matches_direct_decode(tmp_path):
    torch.manual_seed(3)
    W = torch.randn(10, 12)
    store = TieredStore(tmp_path / "nvme")
    ladder = BitPlaneLadder(base_bits=2, resid_bits=2, n_residual_planes=2)
    specs = write_ladder(store, "W", W, ladder)

    recon_from_disk = read_ladder(store, "W", specs)

    codes, specs2, scale = ladder.encode(W)
    recon_direct = BitPlaneLadder.decode(codes, specs2, scale)

    assert torch.allclose(recon_from_disk, recon_direct, atol=1e-6)
