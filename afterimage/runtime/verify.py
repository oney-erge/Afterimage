"""Speculative sampling / verification (IMPLEMENTATION_PLAN.md Phase 3.3).

The standard algorithm (Leviathan et al. 2023; Chen et al. 2023): a draft
model proposes a chain of k tokens; the target model scores all k positions
in one batched sweep (the amortization SpecExec/SpecOffload/SubSpec all
exploit); each draft token is accepted with probability
min(1, p_target(x)/p_draft(x)); the first rejection is replaced by a sample
from the normalized residual (p_target - p_draft)_+, which is what makes the
whole procedure sample EXACTLY from the target's distribution regardless of
how bad the draft is -- correctness never depends on draft quality, only
speed does. If every draft token is accepted, an extra "bonus" token is drawn
from the target's distribution at the position after the chain, which the
verification sweep already computed for free.

This module implements a single linear CHAIN (one proposed continuation),
not a full branching tree (SpecExec-style multi-candidate trees). A tree
verifies more candidates per sweep and is the better production design, but
the chain case already exercises the entire correctness-critical machinery
(the accept/reject/resample step and the exact-distribution guarantee) and is
what this codebase's tests actually verify -- see IMPLEMENTATION_STATUS.md
for what's built versus what a production system would still need.
"""
from __future__ import annotations

import torch


def sample_categorical(p: torch.Tensor, generator: torch.Generator | None = None) -> int:
    return int(torch.multinomial(p, 1, generator=generator).item())


def temperature_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """softmax(logits / temperature), except temperature <= 0 returns a
    one-hot distribution at the argmax instead of dividing by zero.

    This is what makes speculative decoding provably reproduce PLAIN GREEDY
    decoding, not just itself: at temperature 0 both draft and target
    distributions are one-hot at their own argmax, so
    speculative_sample_step's accept/reject step collapses to "accept iff
    the draft's argmax equals the target's argmax, otherwise emit the
    target's argmax" -- exactly generate_greedy's rule, for ANY draft
    (any k, any quality). That gives every adaptive/self-draft arm a real
    correctness assertion (token-identical to generate_greedy) instead of
    only a distributional one -- see docs/ADAPTIVE_TEST_PLAN.md §3.
    """
    if temperature <= 0:
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs
    return torch.softmax(logits / temperature, dim=-1)


def speculative_sample_step(
    draft_probs: list[torch.Tensor],
    target_probs: list[torch.Tensor],
    draft_tokens: list[int],
    bonus_target_probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[list[int], int]:
    """draft_probs[i] / target_probs[i] are the draft/target distributions at
    chain position i, over the SAME vocabulary. bonus_target_probs is the
    target's distribution one position past the full chain (used only if
    every draft token is accepted).

    Returns (tokens, n_accepted_from_draft): tokens has length
    n_accepted_from_draft + 1 -- the accepted prefix of the draft, plus
    exactly one more token (a resample correcting the first rejection, or a
    free bonus token if nothing was rejected).
    """
    k = len(draft_tokens)
    accepted: list[int] = []
    for i in range(k):
        xi = draft_tokens[i]
        p_xi = target_probs[i][xi].item()
        q_xi = draft_probs[i][xi].item()
        accept_prob = min(1.0, p_xi / max(q_xi, 1e-12))
        # device= must match the generator's own device (torch.rand defaults
        # to CPU otherwise, which raises the instant generator is CUDA --
        # exactly the case when target_probs/draft_probs come from a GPU
        # model, as they do outside this module's own CPU-tensor tests).
        # sample_categorical's multinomial call a few lines below already
        # requires generator and probs to share a device, so this is not a
        # new constraint, just the same one applied consistently.
        rand_device = generator.device if generator is not None else None
        u = torch.rand(1, generator=generator, device=rand_device).item()
        if u <= accept_prob:
            accepted.append(xi)
            continue
        residual = (target_probs[i] - draft_probs[i]).clamp_min(0)
        total = residual.sum()
        if total.item() <= 1e-12:
            # Degenerate only when q dominates p everywhere despite this
            # token having just been rejected by the accept test above --
            # falls back to sampling directly from the target at this
            # position, which still preserves the target marginal.
            residual = target_probs[i]
        else:
            residual = residual / total
        accepted.append(sample_categorical(residual, generator))
        return accepted, i

    accepted.append(sample_categorical(bonus_target_probs, generator))
    return accepted, k
