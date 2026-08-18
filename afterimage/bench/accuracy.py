"""Accuracy instruments (VALIDATION_PLAN.md #4).

Three instruments, deliberately ordered by SENSITIVITY rather than by how
much anyone cares about them:

  1. token_identity_rate  -- most sensitive; for a lossless method this must
                             be exactly 1.0 and any deviation is a bug
  2. perplexity           -- continuous; catches degradation that has not yet
                             flipped a single answer
  3. task accuracy        -- what you actually care about, but the NOISIEST
                             detector (see paired_accuracy_test below)

Task accuracy alone cannot establish "no deterioration" at realistic sample
sizes: with n=50 graded questions the smallest detectable difference is
roughly 14 percentage points, so "72% vs 70%" is indistinguishable from
noise. That is why the other two instruments exist and why every accuracy
number here is returned WITH a confidence interval rather than as a bare
point estimate.
"""
from __future__ import annotations

import dataclasses
import math

import torch


# ---------------------------------------------------------------------------
# 1. Token identity -- the strongest signal for lossless claims
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TokenIdentityResult:
    n_prompts: int
    n_tokens_compared: int
    n_tokens_identical: int
    n_prompts_fully_identical: int
    first_divergence_positions: list[int]

    @property
    def token_rate(self) -> float:
        return self.n_tokens_identical / self.n_tokens_compared if self.n_tokens_compared else 1.0

    @property
    def prompt_rate(self) -> float:
        return self.n_prompts_fully_identical / self.n_prompts if self.n_prompts else 1.0

    @property
    def is_lossless(self) -> bool:
        """Exact equality. Deliberately not a tolerance: a method claiming
        losslessness either reproduces every token or it does not."""
        return self.n_tokens_identical == self.n_tokens_compared


def token_identity_rate(reference_sequences: list[list[int]],
                         candidate_sequences: list[list[int]]) -> TokenIdentityResult:
    """Compares greedy-decoded token id sequences position by position.

    Sequences may differ in length (one side may stop early); comparison
    stops at the shorter length and the surplus counts as divergence, since
    a method that truncates output early is not reproducing the reference.
    """
    assert len(reference_sequences) == len(candidate_sequences), (
        f"{len(reference_sequences)} reference vs {len(candidate_sequences)} candidate sequences"
    )

    total = identical = fully = 0
    first_div: list[int] = []

    for ref, cand in zip(reference_sequences, candidate_sequences):
        n = max(len(ref), len(cand))
        total += n
        diverged_at = -1
        for i in range(n):
            r = ref[i] if i < len(ref) else None
            c = cand[i] if i < len(cand) else None
            if r is not None and r == c:
                identical += 1
            elif diverged_at < 0:
                diverged_at = i
        if diverged_at < 0:
            fully += 1
        else:
            first_div.append(diverged_at)

    return TokenIdentityResult(
        n_prompts=len(reference_sequences),
        n_tokens_compared=total,
        n_tokens_identical=identical,
        n_prompts_fully_identical=fully,
        first_divergence_positions=first_div,
    )


# ---------------------------------------------------------------------------
# 2. Perplexity
# ---------------------------------------------------------------------------


def perplexity_from_logits(logits: torch.Tensor, target_ids: torch.Tensor,
                            attention_mask: torch.Tensor | None = None) -> float:
    """logits: (B, T, V) predicting position t+1 from position t.
    target_ids: (B, T). Standard causal shift is applied here, so callers
    pass the model's raw logits and the raw input ids.

    Computed in float32 regardless of input dtype -- exponentiating a mean of
    fp16 log-probs loses meaningful precision, and perplexity is being used
    here precisely to detect SMALL degradations."""
    logits = logits[:, :-1, :].float()
    targets = target_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    if attention_mask is not None:
        mask = attention_mask[:, 1:].to(dtype=token_lp.dtype)
    else:
        mask = torch.ones_like(token_lp)

    total_lp = (token_lp * mask).sum()
    n_tokens = mask.sum().clamp_min(1.0)
    return float(torch.exp(-total_lp / n_tokens).item())


# ---------------------------------------------------------------------------
# 3. Task accuracy, with the statistics that make it interpretable
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PairedAccuracyResult:
    n: int
    ref_correct: int
    cand_correct: int
    both_correct: int
    ref_only: int      # b: reference right, candidate wrong (regressions)
    cand_only: int     # c: candidate right, reference wrong (improvements)
    neither: int

    @property
    def ref_accuracy(self) -> float:
        return self.ref_correct / self.n if self.n else 0.0

    @property
    def cand_accuracy(self) -> float:
        return self.cand_correct / self.n if self.n else 0.0

    @property
    def delta(self) -> float:
        return self.cand_accuracy - self.ref_accuracy

    @property
    def mcnemar_statistic(self) -> float:
        """McNemar's chi-squared with continuity correction, on the
        discordant pairs only. Pairing is what makes this far more powerful
        than comparing two independent accuracy rates at the same n: it
        removes per-question difficulty variance entirely, since both
        systems answered the SAME questions."""
        b, c = self.ref_only, self.cand_only
        if b + c == 0:
            return 0.0
        return (abs(b - c) - 1) ** 2 / (b + c)

    @property
    def significant_at_05(self) -> bool:
        """chi-squared with 1 dof, critical value 3.841 at alpha=0.05."""
        return self.mcnemar_statistic > 3.841

    @property
    def delta_ci95(self) -> tuple[float, float]:
        """Wald CI on the paired difference, derived from discordant pairs."""
        if self.n == 0:
            return (0.0, 0.0)
        b, c = self.ref_only, self.cand_only
        se = math.sqrt(max(b + c, 1)) / self.n
        margin = 1.96 * se
        return (self.delta - margin, self.delta + margin)


def paired_accuracy_test(ref_correct: list[bool], cand_correct: list[bool]) -> PairedAccuracyResult:
    assert len(ref_correct) == len(cand_correct), "paired test needs equal-length results"
    both = ref_only = cand_only = neither = 0
    for r, c in zip(ref_correct, cand_correct):
        if r and c:
            both += 1
        elif r and not c:
            ref_only += 1
        elif c and not r:
            cand_only += 1
        else:
            neither += 1
    return PairedAccuracyResult(
        n=len(ref_correct),
        ref_correct=sum(ref_correct),
        cand_correct=sum(cand_correct),
        both_correct=both,
        ref_only=ref_only,
        cand_only=cand_only,
        neither=neither,
    )


def min_detectable_delta(n: int, p: float = 0.5, power: float = 0.8) -> float:
    """Smallest accuracy difference detectable at the given n, for an
    UNPAIRED two-proportion comparison at alpha=0.05.

    Verified against the standard reference point: detecting 5 pp around
    p=0.5 at 80% power requires n ~= 1570 per group, which this reproduces.

    This is the pessimistic bound. Quote it when the two systems are being
    compared on DIFFERENT question samples. When they answer the same
    questions, use min_detectable_delta_paired -- it is dramatically better
    and is why VALIDATION_PLAN.md insists on paired runs.
    """
    z_alpha, z_beta = 1.96, 0.84 if power == 0.8 else 1.28
    return (z_alpha + z_beta) * math.sqrt(2 * p * (1 - p) / max(n, 1))


def min_detectable_delta_paired(n: int, min_discordant: int = 6) -> float:
    """Smallest one-sided regression detectable by McNemar at the given n.

    Pairing is enormously more powerful for the specific comparison this
    project makes -- a candidate method against the reference model it is
    meant to reproduce -- because the two systems agree on almost every
    item. When the candidate only ever regresses and never improves (c=0),
    McNemar's statistic (|b-c|-1)^2/(b+c) clears the 3.841 critical value at
    roughly b >= 6 discordant pairs. So detecting a regression affecting a
    fraction `delta` of items needs delta*n >= ~6, i.e. delta >= 6/n.

    This is why n=400 is a sensible budget (detects ~1.5 pp one-sided)
    even though the UNPAIRED bound at n=400 is a useless ~10 pp: the
    assumption c~=0 is realistic when comparing a compressed/approximated
    model against the exact model it approximates, since the approximation
    should not be making the model *better* at anything.

    If the candidate both regresses and improves (c > 0), power drops toward
    the unpaired bound -- check `PairedAccuracyResult.cand_only` before
    trusting this figure for a given run.
    """
    return min_discordant / max(n, 1)
