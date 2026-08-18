import torch

from afterimage.probe.closed_loop import calibrate_bases, closed_loop_error, open_loop_error
from afterimage.probe.hooks import ActivationCapture


def test_fit_basis_works_with_half_precision_activations():
    """Regression test for a real crash: torch.linalg.svd does not implement
    half precision on either CUDA (cuSOLVER gesvdj) or CPU (LAPACK), and a
    real model loaded in fp16 for GPU memory produces fp16 activations.
    Reproduced here on CPU with an explicit .half() cast, which hits the
    same "not implemented for Half" class of error the CUDA run hit,
    without needing a GPU to test it."""
    from afterimage.probe.closed_loop import _fit_basis

    torch.manual_seed(0)
    X = torch.randn(50, 20).half()
    Q = _fit_basis(X, r=5)
    assert Q.dtype == torch.float16
    assert Q.shape == (20, 5)


def test_calibrate_and_closed_loop_error_work_end_to_end_with_half_precision():
    import torch.nn as nn

    class TinyBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.proj = nn.Linear(d, d, bias=False)

        def forward(self, x):
            return x + self.proj(x)

    torch.manual_seed(1)
    d = 16
    model = TinyBlock(d).half()
    calib_x = torch.randn(30, d).half()
    eval_x = torch.randn(10, d).half()

    bases = calibrate_bases(model, calib_x, ["proj"], rank=4)
    assert bases["proj"].dtype == torch.float16

    ol = open_loop_error(model, bases, eval_x, ["proj"])
    cl = closed_loop_error(model, bases, eval_x, ["proj"])

    assert 0.0 <= ol["proj"]
    assert 0.0 <= cl
