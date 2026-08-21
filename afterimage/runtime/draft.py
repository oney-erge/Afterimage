"""ARCHIVED RESEARCH, not the production path -- see basis.py's module
docstring for why. The current engine's speculative decoding
(runtime/streaming_engine.py generate_speculative + runtime/verify.py) uses
a real small resident model as the draft, not this low-rank substitute.

Substitute draft model (IMPLEMENTATION_PLAN.md Phase 3.2, LITERATURE.md
#7).

SubSpec's key idea: instead of a separate small model, build the draft FROM
the target's own weights, so the draft's outputs are structurally close to
the target's and acceptance rate is high. SubSpec does this with data-free
low-bit (4-bit HQQ) quantized substitutes; this implementation uses low-rank
weight substitutes (truncated SVD of each weight matrix) for the same reason
SubSpec picked low-bit: it is cheap, data-free, and needs no training. The
underlying principle (build the cheap approximation from the target, not
from an unrelated small model) is identical; the compression mechanism is
different and simpler to implement and verify here.

This is a weight-space approximation and is a different mechanism from the
Afterimage activation-space cache (sketch.py) -- the two are meant to be
used together: the substitute draft proposes candidate tokens cheaply, and
the real (Afterimage-cached, offloaded) target verifies them in one batched
sweep (engine.py).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..testing.toy_lm import ToyLM


class LowRankSubstituteLinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        W = linear.weight.data
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        r = min(rank, S.shape[0])
        self.register_buffer("A", U[:, :r] * S[:r])  # (d_out, r)
        self.register_buffer("B", Vt[:r, :])  # (r, d_in)
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.B.T
        y = h @ self.A.T
        if self.bias is not None:
            y = y + self.bias
        return y

    @property
    def compression_ratio(self) -> float:
        full = self.A.shape[0] * self.B.shape[1]
        low_rank = self.A.numel() + self.B.numel()
        return full / low_rank


def build_substitute_draft(target: ToyLM, rank: int) -> ToyLM:
    """Returns a structural copy of `target` with every backbone Linear
    replaced by a low-rank substitute built from that layer's own weight.
    The embedding and head stay full-precision and shared in spirit with
    SubSpec's "GPU-resident layer sharing," though here they are duplicated
    (deepcopy) for simplicity rather than literally aliased."""
    draft = copy.deepcopy(target)
    for name, mod in list(draft.backbone.named_modules()):
        if isinstance(mod, nn.Linear):
            parent_name, _, attr = name.rpartition(".")
            parent = draft.backbone.get_submodule(parent_name) if parent_name else draft.backbone
            original = target.backbone.get_submodule(name)
            setattr(parent, attr, LowRankSubstituteLinear(original, rank))
    return draft


def propose_chain(draft: nn.Module, prefix: torch.Tensor, k: int, vocab_size: int,
                   temperature: float = 1.0, generator: torch.Generator | None = None
                   ) -> tuple[list[int], list[torch.Tensor]]:
    """Greedily extends `prefix` by k tokens using the draft model, sampling
    each next token from the draft's own distribution. Returns the proposed
    token ids and the draft probability distribution used at each step
    (needed by verify.speculative_sample_step)."""
    tokens: list[int] = []
    probs: list[torch.Tensor] = []
    seq = prefix.clone()
    for _ in range(k):
        p = draft.next_token_probs(seq.unsqueeze(0) if seq.dim() == 1 else seq, temperature=temperature)
        p = p.squeeze(0)
        tok = int(torch.multinomial(p, 1, generator=generator).item())
        tokens.append(tok)
        probs.append(p)
        seq = torch.cat([seq, torch.tensor([tok])])
    return tokens, probs
