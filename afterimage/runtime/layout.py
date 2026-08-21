"""ARCHIVED RESEARCH, not the production path -- see basis.py's module
docstring for why. The current lossless codec is
runtime/huffman_chunked.py + runtime/compressed_store.py.

Bit-plane residual ladder (HYPOTHESIS.md #3.7, #6).

A weight matrix is stored as a coarse base plane plus a sequence of residual
correction planes, each roughly halving/quartering the remaining
reconstruction error. On a miss where the novelty ratio rho = ||x_perp|| /
||x|| is small, only the first few planes need to be fetched -- the
relative error contributed by dropping the rest scales with rho, so fewer
bits suffice (b_eff = b_full - log2(1/rho)).

Per-output-row scaling (one scale factor per row of W) rather than one global
scale, since weight rows can have very different dynamic ranges and a single
global scale wastes precision on rows near the mean.

This is successive-approximation scalar quantization, the same structural
idea as RRQ (arXiv:2608.04048) and Any-Precision LLM: a base quantized value
plus quantized residual corrections against a shrinking range, so truncating
the ladder at any point gives a valid (if coarser) reconstruction.
"""
from __future__ import annotations

import dataclasses

import torch

from .tiers import TieredStore


@dataclasses.dataclass
class PlaneSpec:
    name: str
    bits: int
    step: float  # quantization step used for this plane (float, shared across the tensor)


class BitPlaneLadder:
    def __init__(self, base_bits: int = 2, resid_bits: int = 2, n_residual_planes: int = 3):
        self.base_bits = base_bits
        self.resid_bits = resid_bits
        self.n_residual_planes = n_residual_planes

    @property
    def full_precision_bits(self) -> int:
        return self.base_bits + self.resid_bits * self.n_residual_planes

    def encode(self, W: torch.Tensor) -> tuple[list[torch.Tensor], list[PlaneSpec], torch.Tensor]:
        """Returns (plane_codes, specs, per_row_scale). plane_codes[i] is an
        int8 tensor the same shape as W; specs[i].step is the float step size
        needed to reconstruct that plane's contribution."""
        scale = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        Wn = W / scale

        codes: list[torch.Tensor] = []
        specs: list[PlaneSpec] = []

        qmax0 = 2 ** (self.base_bits - 1) - 1
        step0 = 1.0 / qmax0
        code0 = torch.clamp(torch.round(Wn / step0), -qmax0, qmax0)
        codes.append(code0.to(torch.int8))
        specs.append(PlaneSpec("base", self.base_bits, step0))

        residual = Wn - code0 * step0
        # Round-to-nearest bounds |residual| by half the step just used, not
        # the full step -- using the full step here (an earlier version of
        # this code did) means each new plane's quantizer is sized for a
        # range twice as large as the residual actually occupies, so it
        # spends its levels on values that can't occur and the ladder barely
        # improves on the base plane alone (caught by
        # tests/test_layout.py::test_reconstruction_error_shrinks_
        # monotonically_with_more_planes during development).
        cur_bound = step0 / 2.0
        qmax_r = 2 ** (self.resid_bits - 1) - 1
        for p in range(self.n_residual_planes):
            step_r = cur_bound / qmax_r
            code_r = torch.clamp(torch.round(residual / step_r), -qmax_r, qmax_r)
            codes.append(code_r.to(torch.int8))
            specs.append(PlaneSpec(f"resid{p}", self.resid_bits, step_r))
            residual = residual - code_r * step_r
            cur_bound = step_r / 2.0

        return codes, specs, scale

    @staticmethod
    def decode(codes: list[torch.Tensor], specs: list[PlaneSpec], scale: torch.Tensor,
               up_to_plane: int | None = None) -> torch.Tensor:
        if up_to_plane is None:
            up_to_plane = len(codes)
        acc = torch.zeros_like(codes[0], dtype=torch.float32)
        for code, spec in zip(codes[:up_to_plane], specs[:up_to_plane]):
            acc = acc + code.to(torch.float32) * spec.step
        return acc * scale

    def bits_for_ratio(self, rho: float, min_planes: int = 1) -> int:
        """HYPOTHESIS.md #3.7: b_eff = b_full - log2(1/rho). Returns the
        number of planes (>= min_planes) whose cumulative bit budget covers
        b_eff. rho <= 0 or >= 1 both fall back to full precision (rho>=1
        means the miss carries no exploitable slack; rho<=0 is degenerate)."""
        if rho <= 0.0 or rho >= 1.0:
            return len(self.plane_bit_list())
        import math
        b_eff = self.full_precision_bits + math.log2(rho)
        cum = 0
        for i, b in enumerate(self.plane_bit_list()):
            cum += b
            if cum >= b_eff:
                return max(min_planes, i + 1)
        return len(self.plane_bit_list())

    def plane_bit_list(self) -> list[int]:
        return [self.base_bits] + [self.resid_bits] * self.n_residual_planes


def write_ladder(store: TieredStore, key: str, W: torch.Tensor, ladder: BitPlaneLadder) -> list[PlaneSpec]:
    codes, specs, scale = ladder.encode(W)
    for i, code in enumerate(codes):
        store.write_nvme(f"{key}.plane{i}", code)
    store.write_nvme(f"{key}.scale", scale)
    return specs


def read_ladder(store: TieredStore, key: str, specs: list[PlaneSpec], n_planes: int | None = None) -> torch.Tensor:
    """Reads only the first n_planes planes -- this is where precision
    escalation actually saves bytes: unread planes are never touched."""
    if n_planes is None:
        n_planes = len(specs)
    codes = [store.read_nvme_raw(f"{key}.plane{i}") for i in range(n_planes)]
    scale = store.read_nvme_raw(f"{key}.scale")
    return BitPlaneLadder.decode(codes, specs[:n_planes], scale, up_to_plane=n_planes)
