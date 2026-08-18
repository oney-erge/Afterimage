"""The Afterimage cache for one linear layer (HYPOTHESIS.md #2-#3).

Ties together OnlineBasis (which directions have we seen), JLGate (would the
output change enough to matter if we didn't fetch), and TieredStore (where
the real weights live). This is the object whose correctness the project's
central unit test depends on: for activations confined to a rank-r subspace,
hits must reproduce W @ x to machine precision (IMPLEMENTATION_PLAN #10.1).
"""
from __future__ import annotations

import torch

from .basis import OnlineBasis
from .gate import GlobalController, JLGate
from .tiers import TieredStore


class AfterimageLayer:
    def __init__(self, key: str, d_in: int, d_out: int, max_rank: int,
                 store: TieredStore, gate: JLGate, controller: GlobalController,
                 fill_rank: int = 1, dtype: torch.dtype = torch.float32,
                 device: str | torch.device = "cpu"):
        self.key = key
        self.d_in = d_in
        self.d_out = d_out
        self.store = store
        self.gate = gate
        self.controller = controller
        self.fill_rank = fill_rank  # batched fills, HYPOTHESIS.md #3.5
        self.dtype = dtype
        self.device = torch.device(device)

        self.basis = OnlineBasis(d_in, max_rank, dtype=dtype, device=self.device)
        self.M = torch.zeros((d_out, 0), dtype=dtype, device=self.device)

        self.hits = 0
        self.misses = 0
        self.forced_calibration_misses = 0

    def forward(self, x: torch.Tensor, extra_directions: list[torch.Tensor] | None = None) -> torch.Tensor:
        """extra_directions: activation vectors predicted by the draft model
        for upcoming positions (HYPOTHESIS.md #3.5, batched fills). Installed
        alongside the real residual on a miss, at zero extra I/O cost since
        the weight matrix is already resident at that moment."""
        c, x_perp = self.basis.project(x)

        if self.gate.S is None:
            fetch = True
            self.forced_calibration_misses += 1
        else:
            err = self.gate.estimate_output_error(x_perp)
            fetch = self.controller.should_fetch(self.key, err)

        if not fetch:
            self.hits += 1
            self.basis.bump(c)
            return self.M @ c

        self.misses += 1
        W = self.store.get(self.key).to(dtype=self.dtype, device=self.device)
        if self.gate.S is None:
            self.gate.calibrate(W)
        y = W @ x

        x_norm = torch.linalg.vector_norm(x).item()
        to_install = [x_perp]
        if extra_directions:
            to_install.extend(extra_directions[: max(0, self.fill_rank - 1)])

        for direction in to_install:
            added, evicted = self.basis.add(direction, x_norm=x_norm)
            if evicted is not None:
                keep = [i for i in range(self.M.shape[1]) if i != evicted]
                self.M = self.M[:, keep]
            if added:
                new_u = self.basis.U[:, -1]
                m_new = (W @ new_u).unsqueeze(1)
                self.M = torch.cat([self.M, m_new], dim=1)

        return y

    def forward_batch(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, d_in), one row per position in a draft chain/tree being
        verified together. This is the actual amortization mechanism behind
        "verify k tokens in one sweep" (SpecExec/SpecOffload/SubSpec,
        LITERATURE.md #5-#7): each row's hit/miss decision is independent
        (cheap, uses only the resident gate) but if ANY row misses, W is
        fetched ONCE and reused for every missing row in a single batched
        matmul -- not fetched once per row. The basis is updated once per
        miss row afterward, in x-norm order arbitrary (row order), same as
        the single-x path.
        """
        B = X.shape[0]
        c_all, x_perp_all = self.basis.project_batch(X)
        if self.gate.S is None:
            miss_mask = torch.ones(B, dtype=torch.bool)
        else:
            errs = self.gate.estimate_output_error_batch(x_perp_all)
            miss_mask = torch.tensor(
                [self.controller.should_fetch(self.key, e) for e in errs.tolist()]
            )

        Y = torch.zeros(B, self.d_out, dtype=self.dtype, device=self.device)
        n_hit = int((~miss_mask).sum().item())
        n_miss = int(miss_mask.sum().item())
        self.hits += n_hit
        self.misses += n_miss

        if n_hit > 0:
            if c_all is None:
                Y[~miss_mask] = 0.0
            else:
                Y[~miss_mask] = (self.M @ c_all[:, ~miss_mask]).T
                for row_c in c_all[:, ~miss_mask].T:
                    self.basis.bump(row_c)

        if n_miss > 0:
            W = self.store.get(self.key).to(dtype=self.dtype, device=self.device)
            if self.gate.S is None:
                self.gate.calibrate(W)
                self.forced_calibration_misses += 1
            Xm = X[miss_mask]
            Y[miss_mask] = Xm @ W.T
            for row in torch.nonzero(miss_mask).flatten().tolist():
                x = X[row]
                x_perp = x_perp_all[:, row] if x_perp_all is not None else x
                added, evicted = self.basis.add(x_perp, x_norm=torch.linalg.vector_norm(x).item())
                if evicted is not None:
                    keep = [i for i in range(self.M.shape[1]) if i != evicted]
                    self.M = self.M[:, keep]
                if added:
                    new_u = self.basis.U[:, -1]
                    m_new = (W @ new_u).unsqueeze(1)
                    self.M = torch.cat([self.M, m_new], dim=1)

        return Y

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def resident_bytes(self) -> int:
        u_bytes = self.basis.U.element_size() * self.basis.U.nelement()
        m_bytes = self.M.element_size() * self.M.nelement()
        return u_bytes + m_bytes + self.gate.resident_bytes
