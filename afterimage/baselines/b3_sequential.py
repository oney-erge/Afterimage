"""The AirLLM-equivalent control baseline (IMPLEMENTATION_PLAN.md #1, #5:
"our own controlled AirLLM-equivalent... an apples-to-apples control").

No residency, no cache, no speculation, no batching -- every layer's weight
is fetched fresh from NVMe for every single generated token, exactly the
"repeat the entire sweep for every single token" behavior diagrammed in
LITERATURE.md #2. This exists so that "Afterimage is faster than AirLLM" and
"Afterimage is faster than our own engine's floor" are measured separately;
comparing only against the real AirLLM confounds the method with engine and
kernel differences (different codebase, different tokenizer, different I/O
path).
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from ..runtime.tiers import Tier, TieredStore
from ..testing.toy_lm import ToyLM


@dataclasses.dataclass
class BlockMeta:
    norm: torch.nn.Module
    up_key: str
    down_key: str


def prepare_sequential_baseline(target: ToyLM, store: TieredStore) -> list[BlockMeta]:
    metas = []
    for i, block in enumerate(target.backbone.blocks):
        up_key, down_key = f"seq.block{i}.up", f"seq.block{i}.down"
        store.write_nvme(up_key, block.up.weight.data)
        store.write_nvme(down_key, block.down.weight.data)
        metas.append(BlockMeta(block.norm, up_key, down_key))
    return metas


def sequential_forward(store: TieredStore, metas: list[BlockMeta], embed: torch.nn.Embedding,
                        head: torch.nn.Linear, seq: torch.Tensor) -> torch.Tensor:
    x = embed(seq).mean(dim=0)
    for meta in metas:
        xn = meta.norm(x)
        Wup = store.get(meta.up_key)  # fresh NVMe fetch, every layer, every token
        h = F.gelu(Wup @ xn)
        Wdown = store.get(meta.down_key)
        x = x + Wdown @ h
    return head(x)


@dataclasses.dataclass
class SequentialStats:
    tokens_generated: int = 0
    bytes_read_nvme: int = 0

    @property
    def gb_per_token(self) -> float:
        return (self.bytes_read_nvme / 1e9) / self.tokens_generated if self.tokens_generated else 0.0


def run_sequential_baseline(target: ToyLM, store: TieredStore, metas: list[BlockMeta],
                             prefix: torch.Tensor, n_tokens: int, temperature: float = 1.0,
                             seed: int = 0) -> tuple[torch.Tensor, SequentialStats]:
    gen = torch.Generator().manual_seed(seed)
    seq = prefix.clone()
    stats = SequentialStats()
    b0 = store.stats[Tier.NVME].bytes_read

    for _ in range(n_tokens):
        logits = sequential_forward(store, metas, target.embed, target.head, seq)
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        tok = int(torch.multinomial(probs, 1, generator=gen).item())
        seq = torch.cat([seq, torch.tensor([tok])])
        stats.tokens_generated += 1

    stats.bytes_read_nvme = store.stats[Tier.NVME].bytes_read - b0
    return seq, stats
