"""ARCHIVED RESEARCH, not the production path. Phase 0 (docs/PHASE0_RESULTS.md)
measured this subspace-cache idea against real hardware and got a NO-GO:
functional error 250-450x above threshold. The current engine is
runtime/streaming_engine.py (lossless compressed weight streaming) -- see
README.md and docs/MASTER_PLAN.md. Kept, not deleted: the code is correct
and tested (67 passing tests), and the negative result has standalone value.

Online orthonormal basis for the Afterimage cache.

HYPOTHESIS.md #3, #9.5: maintains U such that span(U) covers the activation
directions seen so far for one linear layer. Extended incrementally on every
miss at zero extra I/O cost (the weights are already in VRAM at that moment).

Uses modified Gram-Schmidt with a second reorthogonalization pass. Giraud &
Langou (2002) show one reorthogonalization pass is enough to reach machine
precision orthogonality for MGS in floating point; we do it unconditionally
rather than adaptively, since the extra pass is a handful of (dim x r) matvecs
against an r that stays small by construction.

Eviction is least-frequently-used, not least-recently-used: per
HYPOTHESIS.md #3.5, atom usage under a workload is expected to be heavy-tailed
(a few directions get hit constantly, most get hit once and never again), and
LFU is the correct policy under a heavy-tailed access distribution (Cormode &
Muthukrishnan; standard result in the caching literature, see
IMPLEMENTATION_PLAN.md #7.1).
"""
from __future__ import annotations

import torch


class OnlineBasis:
    def __init__(self, dim: int, max_rank: int, min_norm: float = 1e-8,
                 min_norm_ratio: float = 1e-5,
                 dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"):
        self.dim = dim
        self.max_rank = max_rank
        self.min_norm = min_norm
        self.min_norm_ratio = min_norm_ratio
        self.dtype = dtype
        self.device = torch.device(device)
        self.U = torch.zeros((dim, 0), dtype=dtype, device=self.device)
        self.usage: list[int] = []
        self.total_adds = 0
        self.total_evictions = 0

    @property
    def rank(self) -> int:
        return self.U.shape[1]

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Returns (coefficients c, residual x_perp). c is None if rank is 0."""
        if self.rank == 0:
            return None, x
        c = self.U.T @ x
        x_perp = x - self.U @ c
        return c, x_perp

    def project_batch(self, X: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        """X: (B, dim). Returns (C, X_perp) with C of shape (r, B) -- same
        column-major convention as project()'s single-vector c, stacked --
        and X_perp of shape (dim, B)."""
        Xt = X.T  # (dim, B)
        if self.rank == 0:
            return None, Xt
        C = self.U.T @ Xt  # (r, B)
        X_perp = Xt - self.U @ C
        return C, X_perp

    def bump(self, coeffs: torch.Tensor, threshold: float = 1e-3) -> None:
        """Credit usage to directions that materially contributed to this query."""
        if coeffs is None:
            return
        mags = coeffs.abs()
        active = (mags > threshold * mags.max().clamp_min(1e-12)).nonzero().flatten()
        for i in active.tolist():
            self.usage[i] += 1

    def add(self, x_perp: torch.Tensor, x_norm: float | None = None) -> tuple[bool, int | None]:
        """Install a new direction from a residual.

        x_norm, when given, is the norm of the original activation x that
        x_perp was computed from. Without it, "negligible residual" can only
        be judged against an absolute floor -- but in float32, repeated
        projection against a basis leaves a rounding-noise residual on the
        order of 1e-7 * ||x||, which is well above any reasonable absolute
        floor once ||x|| is large. Left unguarded, that noise gets installed
        as new "directions" indefinitely, polluting the basis with numerical
        garbage until it saturates at max_rank on pure floating-point error
        (caught by tests/test_basis.py::test_subspace_activations_saturate_
        basis_at_true_rank during development). The fix is a threshold
        relative to signal scale, not the input-space novelty gate rejected
        in HYPOTHESIS.md #3.1 -- this only decides whether a residual is real
        information or float noise, it never decides whether to fetch.

        Returns (added, evicted_index). added is False (evicted_index=None)
        if x_perp is numerically already inside span(U) -- nothing to install,
        the basis already covers this direction. evicted_index is set when
        the basis was at capacity and an existing column had to be dropped to
        make room; callers must drop the matching column of any parallel
        resident structure (e.g. sketch.py's M = W @ U) at that index.
        """
        u = x_perp.to(dtype=self.dtype)
        norm = torch.linalg.vector_norm(u)
        floor = self.min_norm
        if x_norm is not None:
            floor = max(floor, self.min_norm_ratio * x_norm)
        if norm < floor:
            return False, None
        u = u / norm

        if self.rank > 0:
            u = u - self.U @ (self.U.T @ u)
            u = u - self.U @ (self.U.T @ u)  # second MGS pass
            n2 = torch.linalg.vector_norm(u)
            if n2 < floor:
                return False, None
            u = u / n2

        evicted = None
        if self.rank >= self.max_rank:
            evicted = int(torch.tensor(self.usage).argmin().item())
            keep = [i for i in range(self.rank) if i != evicted]
            self.U = self.U[:, keep]
            self.usage = [self.usage[i] for i in keep]
            self.total_evictions += 1

        self.U = torch.cat([self.U, u.unsqueeze(1)], dim=1)
        self.usage.append(1)
        self.total_adds += 1
        return True, evicted

    def orthogonality_error(self) -> float:
        if self.rank == 0:
            return 0.0
        gram = self.U.T @ self.U
        eye = torch.eye(self.rank, dtype=self.dtype, device=self.device)
        return torch.linalg.matrix_norm(gram - eye, ord="fro").item()

    def rebuild(self) -> None:
        """Full re-orthonormalization via QR. Called if orthogonality_error()
        exceeds tolerance (IMPLEMENTATION_PLAN.md #10.1: monitor every ~100
        updates, rebuild above 1e-4)."""
        if self.rank == 0:
            return
        q, _ = torch.linalg.qr(self.U)
        self.U = q[:, : self.rank]
