"""A minimal language model on top of ToyTransformer: token embedding table
plus an output head over a small vocabulary. Exists only because no real
tokenizer/vocabulary is available in this environment (no `transformers`
package -- see IMPLEMENTATION_STATUS.md) and speculative decoding needs a
real notion of "next token distribution" to test against, not just a
regression output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .toy_model import ToyTransformer


class ToyLM(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 32, d_ffn: int = 96,
                 n_layers: int = 3, seed: int | None = 0):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.backbone = ToyTransformer(d_model=d_model, d_ffn=d_ffn, n_layers=n_layers, seed=None)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B,) or (B, T) of int64. Returns logits at the LAST
        position only, (B, vocab_size) -- this toy model has no positional
        structure or causal masking; it is a stand-in for "given a prefix,
        what is the distribution over the next token," which is all
        speculative decoding needs from it."""
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(1)
        x = self.embed(token_ids)  # (B, T, d)
        x = x.mean(dim=1)  # collapse sequence -- no causal attention in this toy stand-in
        h = self.backbone(x)
        return self.head(h)

    def next_token_probs(self, prefix: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.forward(prefix)
        return F.softmax(logits / max(temperature, 1e-6), dim=-1)

    def linear_layers(self) -> dict[str, nn.Linear]:
        out = {name: mod for name, mod in self.backbone.named_modules() if isinstance(mod, nn.Linear)}
        out["head"] = self.head
        return out
