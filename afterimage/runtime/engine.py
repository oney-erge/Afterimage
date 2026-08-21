"""ARCHIVED RESEARCH, not the production path -- see basis.py's module
docstring for why, and runtime/streaming_engine.py for the current engine
(unrelated despite the similar filename/role).

Decode loop: draft + batched offloaded verification + Afterimage cache
(IMPLEMENTATION_PLAN.md Phase 3-4), wired together and run against the toy
LM since no real tokenizer/model is available in this environment (see
IMPLEMENTATION_STATUS.md).

Each decode "sweep" verifies an entire draft chain of k+1 positions (k draft
tokens plus one bonus position) through the offloaded target in ONE pass per
layer (sketch.py's forward_batch), which is the actual mechanism by which
speculative verification amortizes weight I/O across multiple tokens
(LITERATURE.md #5). The Afterimage cache then additionally lets some of
those layer evaluations skip I/O entirely on a hit.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..testing.toy_lm import ToyLM
from .gate import GlobalController, JLGate
from .sketch import AfterimageLayer
from .tiers import Tier, TieredStore
from .verify import sample_categorical, speculative_sample_step


class OffloadedBlock:
    def __init__(self, norm: nn.LayerNorm, up: AfterimageLayer, down: AfterimageLayer):
        self.norm = norm
        self.up = up
        self.down = down

    def forward_batch(self, X: torch.Tensor) -> torch.Tensor:
        Xn = self.norm(X)
        H = F.gelu(self.up.forward_batch(Xn))
        D = self.down.forward_batch(H)
        return X + D


class OffloadedToyLM:
    """Same computation as ToyLM, but every backbone Linear is an
    AfterimageLayer backed by a TieredStore instead of a plain nn.Linear.
    The embedding and head are kept full-precision and resident, matching
    the "small shared layers stay resident" idea (LITERATURE.md #7)."""

    def __init__(self, embed: nn.Embedding, blocks: list[OffloadedBlock], head: nn.Linear):
        self.embed = embed
        self.blocks = blocks
        self.head = head

    def forward_batch_from_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            X = block.forward_batch(X)
        return self.head(X)


def build_offloaded_target(target: ToyLM, store: TieredStore, max_rank: int,
                            gate_m: int = 24, lam: float = 1e-4, seed: int = 0) -> OffloadedToyLM:
    blocks = []
    for i, block in enumerate(target.backbone.blocks):
        up_key, down_key = f"block{i}.up", f"block{i}.down"
        store.write_nvme(up_key, block.up.weight.data)
        store.write_nvme(down_key, block.down.weight.data)

        up_gate = JLGate(block.up.in_features, block.up.out_features, m=gate_m, seed=seed + i * 2)
        down_gate = JLGate(block.down.in_features, block.down.out_features, m=gate_m, seed=seed + i * 2 + 1)
        ctrl_up = GlobalController(lam=lam)
        ctrl_down = GlobalController(lam=lam)

        up_layer = AfterimageLayer(up_key, block.up.in_features, block.up.out_features,
                                    max_rank, store, up_gate, ctrl_up)
        down_layer = AfterimageLayer(down_key, block.down.in_features, block.down.out_features,
                                      max_rank, store, down_gate, ctrl_down)
        blocks.append(OffloadedBlock(block.norm, up_layer, down_layer))

    return OffloadedToyLM(target.embed, blocks, target.head)


def embed_chain_contexts(embed: nn.Embedding, prefix: torch.Tensor, draft_tokens: list[int]) -> torch.Tensor:
    """Returns (k+1, d_model): row i (i<k) is the embedding of the context
    that should predict draft_tokens[i]; row k is the context after the
    full accepted chain (used for the bonus token)."""
    rows = []
    seq = prefix
    rows.append(embed(seq).mean(dim=0))
    for t in draft_tokens:
        seq = torch.cat([seq, torch.tensor([t])])
        rows.append(embed(seq).mean(dim=0))
    return torch.stack(rows, dim=0)


@dataclasses.dataclass
class DecodeStats:
    tokens_generated: int = 0
    sweeps: int = 0
    bytes_read_nvme: int = 0
    bytes_read_vram: int = 0
    bytes_read_ram: int = 0

    @property
    def tokens_per_sweep(self) -> float:
        return self.tokens_generated / self.sweeps if self.sweeps else 0.0

    @property
    def gb_per_token(self) -> float:
        total = self.bytes_read_nvme + self.bytes_read_vram + self.bytes_read_ram
        return (total / 1e9) / self.tokens_generated if self.tokens_generated else 0.0


def run_decode(target: ToyLM, draft: nn.Module, store: TieredStore, offloaded: OffloadedToyLM,
                prefix: torch.Tensor, k: int, n_sweeps: int, temperature: float = 1.0,
                seed: int = 0) -> tuple[torch.Tensor, DecodeStats]:
    from .draft import propose_chain

    gen = torch.Generator().manual_seed(seed)
    stats = DecodeStats()
    seq = prefix.clone()

    b0 = {t: s.bytes_read for t, s in store.stats.items()}

    for _ in range(n_sweeps):
        draft_tokens, draft_probs = propose_chain(draft, seq, k, target.vocab_size,
                                                    temperature=temperature, generator=gen)
        contexts = embed_chain_contexts(offloaded.embed, seq, draft_tokens)  # (k+1, d)
        logits = offloaded.forward_batch_from_embeddings(contexts)
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        target_probs = list(probs[:k])
        bonus_probs = probs[k]

        tokens, _n_accepted = speculative_sample_step(draft_probs, target_probs, draft_tokens,
                                                        bonus_probs, gen)
        seq = torch.cat([seq, torch.tensor(tokens)])
        stats.tokens_generated += len(tokens)
        stats.sweeps += 1

    stats.bytes_read_nvme = store.stats[Tier.NVME].bytes_read - b0[Tier.NVME]
    stats.bytes_read_vram = store.stats[Tier.VRAM].bytes_read - b0[Tier.VRAM]
    stats.bytes_read_ram = store.stats[Tier.RAM].bytes_read - b0[Tier.RAM]
    return seq, stats
