import torch
import torch.nn as nn

from afterimage.probe.closed_loop import calibrate_bases, open_loop_error
from afterimage.probe.hooks import ActivationCapture


class TinyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return x + self.proj(x)


def test_stacked_masked_excludes_pad_positions():
    torch.manual_seed(0)
    d = 8
    model = TinyBlock(d)
    x = torch.randn(3, 5, d)  # (batch, seq, d)
    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
    ])

    with ActivationCapture(model, layer_names=["proj"]) as cap:
        model(x)

    masked = cap.stacked_masked("proj", mask)
    assert masked.shape[0] == mask.sum().item() == 3 + 4 + 1

    unmasked = cap.stacked("proj")
    assert unmasked.shape[0] == 3 * 5


def test_stacked_masked_matches_manual_indexing():
    torch.manual_seed(1)
    d = 6
    model = TinyBlock(d)
    x = torch.randn(2, 4, d)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    with ActivationCapture(model, layer_names=["proj"]) as cap:
        model(x)

    got = cap.stacked_masked("proj", mask)
    expected = torch.cat([x[0, :2], x[1, :4]], dim=0)
    assert torch.allclose(got, expected)


def test_pad_contamination_distorts_the_variance_curve():
    """Demonstrates WHY the fix matters. The docstring's claim is specifically
    about VARIANCE concentration (what variance_rank_curve measures), not
    discrete matrix rank -- a first version of this test asserted a discrete-
    rank claim and was wrong (duplicated pad rows add a new linearly
    independent direction, so exact rank goes UP, not down). The real harm:
    with realistic short-real / long-pad sequences, many identical pad rows
    pile onto one direction while only a minority of rows are real content,
    so that ONE direction (which carries zero real information -- it is
    always the same pad-token embedding) dominates the variance curve and
    a naive rank-1 basis looks like it captures "almost everything.\""""
    from afterimage.probe.spectra import variance_rank_curve

    torch.manual_seed(2)
    d = 10
    model = TinyBlock(d)

    real_content = torch.randn(2, 2, d)  # 2 sequences, 2 real tokens each
    pad_value = torch.randn(d) * 20  # far from the real content, like a real pad embedding
    padded = torch.cat([real_content, pad_value.expand(2, 20, d)], dim=1)  # long pad tail
    mask = torch.cat([torch.ones(2, 2), torch.zeros(2, 20)], dim=1)

    with ActivationCapture(model, layer_names=["proj"]) as cap:
        model(padded)

    contaminated = cap.stacked("proj")
    clean = cap.stacked_masked("proj", mask)

    assert contaminated.shape[0] == 44  # 2 * 22
    assert clean.shape[0] == 4  # 2 * 2 real tokens only

    _, contaminated_curve = variance_rank_curve(contaminated, max_rank=1)
    _, clean_curve = variance_rank_curve(clean, max_rank=1)

    assert contaminated_curve[0] > 0.9, (
        f"expected the pad direction to dominate variance in the "
        f"contaminated set: rank-1 captured={contaminated_curve[0]:.3f}"
    )
    assert contaminated_curve[0] > clean_curve[0], (
        "pad contamination should make rank-1 look MORE sufficient than it "
        f"genuinely is: contaminated={contaminated_curve[0]:.3f} "
        f"clean={clean_curve[0]:.3f}"
    )


def test_calibrate_bases_with_mask_ignores_padding():
    torch.manual_seed(3)
    d, r = 12, 3
    model = TinyBlock(d)

    real = torch.randn(4, 3, d)
    q, _ = torch.linalg.qr(torch.randn(d, r))
    real = (torch.randn(4, 3, r) @ q.T)  # confine real tokens to a rank-r subspace
    pad_value = torch.randn(d) * 50  # deliberately huge, unrelated to the subspace
    x = torch.cat([real, pad_value.expand(4, 5, d)], dim=1)
    mask = torch.cat([torch.ones(4, 3), torch.zeros(4, 5)], dim=1)

    bases_masked = calibrate_bases(model, x, ["proj"], rank=r, attention_mask=mask)
    bases_unmasked = calibrate_bases(model, x, ["proj"], rank=r, attention_mask=None)

    eval_real = (torch.randn(2, 3, r) @ q.T)
    eval_mask = torch.ones(2, 3)
    err_masked = open_loop_error(model, bases_masked, eval_real, ["proj"], attention_mask=eval_mask)
    err_unmasked = open_loop_error(model, bases_unmasked, eval_real, ["proj"], attention_mask=eval_mask)

    assert err_masked["proj"] < 1e-3, f"masked basis should recover the true subspace: {err_masked}"
    assert err_masked["proj"] < err_unmasked["proj"], (
        "a basis contaminated by huge pad-token activations should be "
        "measurably worse at representing the real subspace"
    )
