import pytest
import torch
import torch.nn as nn

from afterimage.probe.closed_loop import calibrate_bases, closed_loop_error, open_loop_error
from afterimage.probe.hooks import ActivationCapture


class Inner(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return x + self.proj(x)


class Wrapper(nn.Module):
    """Mirrors scripts/run_probe_real.py's LogitsOnly: nests the real model
    one level deeper under a named attribute, which shifts every submodule's
    dotted path by that attribute's name."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner_model = inner

    def forward(self, x):
        return self.inner_model(x)


def test_unprefixed_layer_name_silently_matches_nothing_on_a_wrapped_model():
    """Documents the exact failure mode caught in scripts/run_probe_real.py:
    passing the INNER model's own layer name to a wrapper's ActivationCapture
    finds nothing, and the dict key is simply never created."""
    torch.manual_seed(0)
    inner = Inner(8)
    wrapper = Wrapper(inner)
    x = torch.randn(3, 8)

    with ActivationCapture(wrapper, layer_names=["proj"]) as cap:
        wrapper(x)

    assert "proj" not in cap.captured
    with pytest.raises(KeyError):
        cap.stacked("proj")


def test_prefixed_layer_name_correctly_matches_on_a_wrapped_model():
    torch.manual_seed(0)
    inner = Inner(8)
    wrapper = Wrapper(inner)
    x = torch.randn(3, 8)

    with ActivationCapture(wrapper, layer_names=["inner_model.proj"]) as cap:
        wrapper(x)

    assert cap.stacked("inner_model.proj").shape == (3, 8)


def test_calibrate_bases_and_closed_loop_error_work_through_a_wrapper():
    """End-to-end version of the real bug: calibrate_bases / open_loop_error
    / closed_loop_error must be called with layer names resolved relative to
    whatever module object is actually passed in, not relative to some
    inner model it wraps."""
    torch.manual_seed(1)
    d, r = 10, 3
    inner = Inner(d)
    wrapper = Wrapper(inner)
    wrapped_layers = ["inner_model.proj"]

    calib_x = torch.randn(20, d)
    eval_x = torch.randn(10, d)

    bases = calibrate_bases(wrapper, calib_x, wrapped_layers, rank=r)
    ol = open_loop_error(wrapper, bases, eval_x, wrapped_layers)
    cl = closed_loop_error(wrapper, bases, eval_x, wrapped_layers)

    assert set(ol.keys()) == {"inner_model.proj"}
    assert 0.0 <= cl
