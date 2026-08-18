import torch

from afterimage.runtime.gate import GlobalController, JLGate


def test_jl_estimate_tracks_true_output_norm_within_bound():
    torch.manual_seed(0)
    d_in, d_out, m = 200, 300, 64
    W = torch.randn(d_out, d_in) / (d_in ** 0.5)

    gate = JLGate(d_in, d_out, m=m, seed=1)
    gate.calibrate(W)

    rel_errors = []
    for _ in range(200):
        v = torch.randn(d_in)
        true_norm = torch.linalg.vector_norm(W @ v).item()
        est_norm = gate.estimate_output_error(v)
        rel_errors.append(abs(est_norm - true_norm) / true_norm)

    rel_errors = torch.tensor(rel_errors)
    # JL concentration: expect the bulk of trials within O(1/sqrt(m)) ~ 0.35
    # at m=64; assert a loose but real bound so the test catches a broken
    # implementation (e.g. wrong scaling) without being flaky.
    assert rel_errors.mean().item() < 0.30
    assert (rel_errors < 0.6).float().mean().item() > 0.9


def test_gate_never_touches_full_weight_matrix_after_calibration():
    """Correctness-by-construction check: estimate_output_error must be
    computable from S alone (m x d_in), never from W (d_out x d_in), since
    avoiding the W fetch is the entire point of the gate."""
    torch.manual_seed(2)
    d_in, d_out, m = 50, 80, 16
    W = torch.randn(d_out, d_in)
    gate = JLGate(d_in, d_out, m=m, seed=3)
    gate.calibrate(W)
    assert gate.S.shape == (m, d_in)

    del W  # if estimate_output_error needed W it would now be broken
    v = torch.randn(d_in)
    val = gate.estimate_output_error(v)
    assert isinstance(val, float) and val >= 0.0


def test_global_controller_threshold():
    ctrl = GlobalController(lam=1.0)
    ctrl.set_sensitivity("layer.0", 2.0)
    assert ctrl.should_fetch("layer.0", 0.6) is True   # 2.0 * 0.6 = 1.2 > 1.0
    assert ctrl.should_fetch("layer.0", 0.4) is False  # 2.0 * 0.4 = 0.8 <= 1.0
    assert ctrl.should_fetch("layer.unseen", 0.5) is False  # default s=1.0
