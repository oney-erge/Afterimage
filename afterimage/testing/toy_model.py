"""Synthetic transformer-like model for testing the probe and runtime without
a real LLM. This development environment has no CUDA and no `transformers`
package installed (IMPLEMENTATION_STATUS.md), so nothing here substitutes for
Phase 0's actual measurement on Gemma-3-27B or Qwen3-32B -- it only lets the
probe and cache machinery be exercised end-to-end against known ground truth
(a controllable, exactly-known effective rank), which is useful in its own
right: it validates that the probe correctly detects low rank when it is
truly present, and correctly detects HIGH rank / functional error when it
is not, before that code is ever pointed at a real model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyBlock(nn.Module):
    """Pre-norm MLP block with an additive residual connection -- the same
    error-damping structure HYPOTHESIS.md #3.4 relies on: approximation
    error introduced by a layer adds into the residual stream rather than
    compounding through a product of Jacobians."""

    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, d_ffn, bias=False)
        self.down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.up(self.norm(x)))
        return x + self.down(h)


class ToyTransformer(nn.Module):
    def __init__(self, d_model: int = 64, d_ffn: int = 256, n_layers: int = 6, seed: int | None = 0):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_layers = n_layers
        self.blocks = nn.ModuleList([ToyBlock(d_model, d_ffn) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x

    def linear_layers(self) -> dict[str, nn.Linear]:
        return {name: mod for name, mod in self.named_modules() if isinstance(mod, nn.Linear)}


def narrow_session_inputs(n_tokens: int, d_model: int, effective_rank: int, seed: int) -> torch.Tensor:
    """Activations confined to a fixed random rank-r subspace of R^d_model --
    the optimistic case Afterimage's mechanism needs (HYPOTHESIS.md #2)."""
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(d_model, effective_rank, generator=g)
    q, _ = torch.linalg.qr(basis)
    coeffs = torch.randn(n_tokens, effective_rank, generator=g)
    return coeffs @ q.T


def topic_switch_inputs(n_tokens: int, d_model: int, effective_rank: int, n_topics: int,
                         switch_every: int, seed: int) -> torch.Tensor:
    """Piecewise: a new random rank-r subspace every `switch_every` tokens.
    The adversarial workload from IMPLEMENTATION_PLAN.md #2.3 -- tests
    whether a single global basis or a clustered one (HYPOTHESIS.md #3.3)
    is needed."""
    g = torch.Generator().manual_seed(seed)
    topic_bases = []
    for _ in range(n_topics):
        basis = torch.randn(d_model, effective_rank, generator=g)
        q, _ = torch.linalg.qr(basis)
        topic_bases.append(q)

    rows = []
    for i in range(n_tokens):
        topic = (i // switch_every) % n_topics
        q = topic_bases[topic]
        coeffs = torch.randn(effective_rank, generator=g)
        rows.append(q @ coeffs)
    return torch.stack(rows, dim=0)


def full_rank_inputs(n_tokens: int, d_model: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_tokens, d_model, generator=g)
