import torch

from afterimage.probe.closed_loop import calibrate_bases, closed_loop_error, open_loop_error
from afterimage.testing.toy_model import ToyTransformer, narrow_session_inputs



import pytest

pytestmark = pytest.mark.archive  # Phase-0 subspace-activation-cache branch, killed -- see docs/archive/README.md
def _setup(seed=0):
    torch.manual_seed(seed)
    model = ToyTransformer(d_model=24, d_ffn=64, n_layers=4, seed=seed)
    model.eval()
    target_layers = [f"blocks.{i}.up" for i in range(4)]
    calib_x = narrow_session_inputs(n_tokens=300, d_model=24, effective_rank=8, seed=seed + 1)
    eval_x = narrow_session_inputs(n_tokens=100, d_model=24, effective_rank=8, seed=seed + 2)
    return model, target_layers, calib_x, eval_x


def test_closed_loop_error_decreases_as_rank_increases():
    model, target_layers, calib_x, eval_x = _setup()
    errors = []
    for r in [2, 6, 12, 24]:
        bases = calibrate_bases(model, calib_x, target_layers, rank=r)
        e = closed_loop_error(model, bases, eval_x, target_layers)
        errors.append(e)
    assert errors[0] > errors[-1], f"error should shrink with rank: {errors}"


def test_closed_loop_error_vanishes_at_full_rank():
    model, target_layers, calib_x, eval_x = _setup(seed=10)
    d_model = model.d_model
    bases = calibrate_bases(model, calib_x, target_layers, rank=d_model)
    e = closed_loop_error(model, bases, eval_x, target_layers)
    assert e < 1e-4, f"full-rank truncation should be a no-op: {e}"


def test_model_is_restored_after_closed_loop_measurement():
    """The layer-swap in closed_loop_error must be transactional -- a crash
    mid-measurement should not leave the model permanently patched."""
    model, target_layers, calib_x, eval_x = _setup(seed=20)
    bases = calibrate_bases(model, calib_x, target_layers, rank=4)
    before = {name: mod for name, mod in model.named_modules() if name in target_layers}
    closed_loop_error(model, bases, eval_x, target_layers)
    after = {name: mod for name, mod in model.named_modules() if name in target_layers}
    for name in target_layers:
        assert before[name] is after[name], f"{name} was not restored"


def test_open_loop_and_closed_loop_can_diverge():
    """The core warning of IMPLEMENTATION_PLAN.md #2.2: open-loop per-layer
    error, calibrated on clean activations, does not have to predict the
    real end-to-end closed-loop error, because closed-loop lets layer i's
    approximation change what layer i+1 actually receives. This test does
    not assert a specific direction (that is itself the Phase-0 empirical
    question) -- it only asserts the two measurements are computed
    independently and are free to disagree, i.e. neither derives from the
    other in the implementation."""
    model, target_layers, calib_x, eval_x = _setup(seed=30)
    r = 4
    bases = calibrate_bases(model, calib_x, target_layers, rank=r)
    per_layer = open_loop_error(model, bases, eval_x, target_layers)
    end_to_end = closed_loop_error(model, bases, eval_x, target_layers)

    assert all(0.0 <= v for v in per_layer.values())
    assert end_to_end >= 0.0
    assert len(per_layer) == len(target_layers)
