"""Output-space error gate (HYPOTHESIS.md #3.1).

The naive gate -- threshold on ||x_perp|| / ||x|| in *input* space -- is wrong.
LLM hidden states are dominated by a few massive-activation dimensions that
carry almost no functional signal (arXiv:2508.16929, arXiv:2604.20682), so an
input-space novelty ratio reads "hit" exactly when it is functionally wrong.

The fix: estimate ||W x_perp|| -- the actual effect on the layer's OUTPUT --
without ever fetching W. This works because W is fixed (frozen pretrained
weight), so its Johnson-Lindenstrauss sketch S = G @ W, with G a random
projection into m << d_out dimensions, can be computed ONCE (offline, or on
the layer's first natural fetch) and kept resident permanently at negligible
size (m x d_in floats, ~0.3 MB at m=32 for a d_in=5376 layer).

At runtime: ||G (W x_perp)|| = ||(G W) x_perp|| = ||S x_perp||, computed with
an (m x d_in) matvec instead of the full (d_out x d_in) matvec -- and without
ever touching the (d_out x d_in) weight matrix itself. Standard JL
concentration (Halko/Martinsson/Tropp; Dasgupta-Gupta) gives
||S x_perp|| = ||W x_perp|| * (1 +/- O(1/sqrt(m))) with high probability, for
G with i.i.d. N(0, 1/m) entries.
"""
from __future__ import annotations

import torch


class JLGate:
    def __init__(self, d_in: int, d_out: int, m: int = 32,
                 dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu",
                 seed: int | None = None):
        self.d_in = d_in
        self.d_out = d_out
        self.m = m
        self.dtype = dtype
        self.device = torch.device(device)
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        g = torch.randn(m, d_out, generator=gen, dtype=torch.float64) / (m ** 0.5)
        self._G = g.to(dtype=dtype, device=self.device)
        self.S: torch.Tensor | None = None  # (m, d_in), set by calibrate()

    def calibrate(self, W: torch.Tensor) -> None:
        """Precompute S = G @ W once, while W happens to be resident (a
        natural miss, or offline model preparation). This is the only time
        the full weight matrix is touched for gating purposes."""
        assert W.shape == (self.d_out, self.d_in), (W.shape, (self.d_out, self.d_in))
        self.S = (self._G.to(dtype=torch.float64) @ W.to(dtype=torch.float64)).to(self.dtype)

    def estimate_output_error(self, x_perp: torch.Tensor) -> float:
        assert self.S is not None, "gate not calibrated -- call calibrate(W) first"
        return torch.linalg.vector_norm(self.S @ x_perp.to(self.dtype)).item()

    def estimate_output_error_batch(self, x_perp: torch.Tensor) -> torch.Tensor:
        """x_perp: (d_in, B). Returns (B,) of per-column error estimates."""
        assert self.S is not None, "gate not calibrated -- call calibrate(W) first"
        return torch.linalg.vector_norm(self.S @ x_perp.to(self.dtype), dim=0)

    @property
    def resident_bytes(self) -> int:
        if self.S is None:
            return 0
        return self.S.element_size() * self.S.nelement()


class GlobalController:
    """One knob (lambda) governing every layer's fetch decision, per
    HYPOTHESIS.md #3.1: fetch layer L iff s_L * ||W x_perp|| > lambda.

    s_L is a per-layer sensitivity (how much a unit of output error at this
    layer moves the final logits) obtained by calibration (IMPLEMENTATION_PLAN
    #10.1 / Phase 0), not guessed. Until calibrated, sensitivities default to
    1.0 (equal weighting) so the controller is usable but not yet optimal.
    """

    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.sensitivity: dict[str, float] = {}

    def set_sensitivity(self, layer_key: str, s: float) -> None:
        self.sensitivity[layer_key] = s

    def should_fetch(self, layer_key: str, estimated_output_error: float) -> bool:
        s = self.sensitivity.get(layer_key, 1.0)
        return (s * estimated_output_error) > self.lam
