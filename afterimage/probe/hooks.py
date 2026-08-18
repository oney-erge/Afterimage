"""Forward-hook activation capture (IMPLEMENTATION_PLAN.md Phase 0).

Captures the INPUT to each nn.Linear -- the activation vector x that would be
multiplied by W -- since that is what the cache's basis needs to span, not
the output.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActivationCapture:
    def __init__(self, model: nn.Module, layer_names: list[str] | None = None):
        self.model = model
        self.layer_names = layer_names
        self.captured: dict[str, list[torch.Tensor]] = {}
        self._handles = []

    def _matches(self, name: str, mod: nn.Module) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if self.layer_names is not None:
            return name in self.layer_names
        return True

    def attach(self) -> "ActivationCapture":
        for name, mod in self.model.named_modules():
            if self._matches(name, mod):
                self.captured.setdefault(name, [])

                def make_hook(layer_name):
                    def hook(module, inputs):
                        x = inputs[0]
                        self.captured[layer_name].append(x.detach().clone())
                    return hook

                handle = mod.register_forward_pre_hook(make_hook(name))
                self._handles.append(handle)
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self) -> None:
        for k in self.captured:
            self.captured[k] = []

    def stacked(self, layer_name: str) -> torch.Tensor:
        """All captured activations for one layer, stacked into (N, d_in)."""
        rows = self.captured[layer_name]
        flat = [r.reshape(-1, r.shape[-1]) for r in rows]
        return torch.cat(flat, dim=0)

    def stacked_masked(self, layer_name: str, attention_mask: torch.Tensor) -> torch.Tensor:
        """Like stacked(), but drops padded positions. Needed whenever
        captured tensors are (batch, seq, d_in) from a right-padded batch:
        an unmasked stack() would include the pad-token embedding's own
        activations, which are near-constant across the padded tail of every
        short sequence and artificially deflate any rank measurement run on
        the result (this was a real bug caught while wiring Phase 0 up to
        the first genuine transformer -- see scripts/run_probe_real.py).
        attention_mask must match the (batch, seq) shape of each captured
        tensor for this layer."""
        rows = self.captured[layer_name]
        mask = attention_mask.to(dtype=torch.bool)
        flat = [r[mask] for r in rows]
        return torch.cat(flat, dim=0)

    def __enter__(self):
        return self.attach()

    def __exit__(self, exc_type, exc, tb):
        self.detach()
