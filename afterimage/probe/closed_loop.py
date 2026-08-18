"""Closed-loop replay (IMPLEMENTATION_PLAN.md #2.2 -- the single easiest way
to fool yourself in this project).

Approximating layer L's output changes layer L+1's INPUT, and therefore
layer L+1's own subspace decomposition. Measuring "would a rank-r basis have
captured this?" against clean, unperturbed activations answers a question
the runtime never actually asks, and produces an optimistic number the
runtime will not reproduce.

This module runs both measurements side by side against the same model and
inputs, so the gap between them is visible rather than assumed:

  - open_loop_error: truncate each targeted layer's input independently,
    using a basis fit on CLEAN activations, and measure only that layer's
    own output error -- no propagation.
  - closed_loop_error: install the truncation into every targeted layer
    SIMULTANEOUSLY and run one real forward pass, so layer L's error
    actually reaches layer L+1's input, is decomposed with L+1's own basis
    (which was calibrated on clean data), and the final output is compared
    end to end against the exact model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .hooks import ActivationCapture


class TruncatedLinear(nn.Module):
    """Replaces an nn.Linear's forward with: project the input onto a fixed
    rank-r basis Q, reconstruct, then apply the real weight -- i.e. simulates
    an Afterimage cache that ALWAYS hits, using whatever basis it was given.
    Wraps the original module so weight/bias are shared, not duplicated."""

    def __init__(self, linear: nn.Linear, basis: torch.Tensor):
        super().__init__()
        self.linear = linear
        self.register_buffer("Q", basis)  # (d_in, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x @ self.Q
        x_hat = c @ self.Q.T
        return nn.functional.linear(x_hat, self.linear.weight, self.linear.bias)


def _fit_basis(X: torch.Tensor, r: int) -> torch.Tensor:
    """Computes the SVD in float32 regardless of X's own dtype, then casts
    the resulting basis back to X's dtype. Neither cuSOLVER nor LAPACK
    implements SVD for half precision -- calling this with fp16 activations
    (the normal case for a real model loaded in fp16 for GPU memory, as
    scripts/run_probe_real.py does) previously raised
    'RuntimeError: "svd_cuda_gesvdj" not implemented for Half' the first
    time this code ran against an actual model; spectra.py's
    layer_rank_report already upcast for the same reason and so never hit
    this. Casting the basis back to X.dtype keeps every downstream matmul
    in TruncatedLinear/open_loop_error/closed_loop_error dtype-consistent
    with whatever activations they are actually called with."""
    Xf = X.float()
    Xc = Xf - Xf.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
    return Vt[:r].T.to(X.dtype)  # (d_in, r)


def calibrate_bases(model: nn.Module, calibration_x: torch.Tensor, target_layers: list[str],
                     rank: int, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """attention_mask, when given, excludes right-padded positions from the
    activations the basis is fit on (hooks.py's ActivationCapture.
    stacked_masked) -- required whenever calibration_x is a padded batch of
    variable-length real sequences, or the basis gets contaminated with
    pad-token activations. The toy-model tests use fixed-length synthetic
    inputs with nothing to mask, hence the default of None."""
    with ActivationCapture(model, layer_names=target_layers) as cap:
        model(calibration_x)
    if attention_mask is not None:
        return {name: _fit_basis(cap.stacked_masked(name, attention_mask), rank) for name in target_layers}
    return {name: _fit_basis(cap.stacked(name), rank) for name in target_layers}


def open_loop_error(model: nn.Module, bases: dict[str, torch.Tensor],
                     eval_x: torch.Tensor, target_layers: list[str],
                     attention_mask: torch.Tensor | None = None) -> dict[str, float]:
    """Per-layer error using CLEAN inputs to each targeted layer -- no
    propagation between layers, since every layer sees the unperturbed
    activation regardless of what happened upstream. See calibrate_bases for
    why attention_mask matters on padded real-sequence batches."""
    linears = {name: mod for name, mod in model.named_modules() if name in target_layers}
    with ActivationCapture(model, layer_names=target_layers) as cap:
        model(eval_x)

    errs = {}
    for name in target_layers:
        X = cap.stacked_masked(name, attention_mask) if attention_mask is not None else cap.stacked(name)
        W = linears[name].weight
        Q = bases[name]
        C = X @ Q
        X_hat = C @ Q.T
        y_exact = X @ W.T
        y_hat = X_hat @ W.T
        num = torch.linalg.vector_norm(y_exact - y_hat, dim=1)
        den = torch.linalg.vector_norm(y_exact, dim=1).clamp_min(1e-12)
        errs[name] = (num / den).mean().item()
    return errs


def closed_loop_error(model: nn.Module, bases: dict[str, torch.Tensor],
                       eval_x: torch.Tensor, target_layers: list[str]) -> float:
    """Installs truncation into every targeted layer at once and runs one
    real forward pass, letting errors from earlier layers reach later ones.
    Returns the relative error of the FINAL model output against the exact
    (untouched) model on the same input."""
    with torch.no_grad():
        y_exact = model(eval_x)

    originals = {}
    for name, mod in model.named_modules():
        if name in target_layers:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            originals[name] = (parent, attr, mod)

    try:
        for name, (parent, attr, mod) in originals.items():
            setattr(parent, attr, TruncatedLinear(mod, bases[name]))
        with torch.no_grad():
            y_closed = model(eval_x)
    finally:
        for name, (parent, attr, mod) in originals.items():
            setattr(parent, attr, mod)

    num = torch.linalg.vector_norm(y_closed - y_exact, dim=1)
    den = torch.linalg.vector_norm(y_exact, dim=1).clamp_min(1e-12)
    return (num / den).mean().item()
