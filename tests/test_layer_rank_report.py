import torch

from afterimage.probe.spectra import (
    effective_rank,
    functional_error_curve,
    layer_rank_report,
    variance_rank_curve,
)


def test_layer_rank_report_matches_the_three_separate_calls():
    torch.manual_seed(0)
    N, d_in, d_out = 200, 64, 40
    X = torch.randn(N, d_in)
    W = torch.randn(d_out, d_in)
    ranks = [2, 5, 10, 20]

    report = layer_rank_report(X, W, ranks)

    _, var_curve = variance_rank_curve(X, max_rank=max(ranks))
    expected_var = [var_curve[r - 1] for r in ranks]
    expected_func = functional_error_curve(X, W, ranks)
    expected_eff_rank = effective_rank(X)

    for a, b in zip(report["variance_captured"], expected_var):
        assert abs(a - b) < 1e-5
    for a, b in zip(report["functional_error"], expected_func):
        assert abs(a - b) < 1e-5
    assert abs(report["effective_rank"] - expected_eff_rank) < 1e-3
    assert report["n_samples"] == N
    assert report["d_in"] == d_in
