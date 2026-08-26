
import pytest
import torch

from afterimage.bench.accuracy import (
    min_detectable_delta,
    min_detectable_delta_paired,
    paired_accuracy_test,
    perplexity_from_logits,
    token_identity_rate,
)


# --- token identity --------------------------------------------------------


def test_identical_sequences_are_lossless():
    ref = [[1, 2, 3], [4, 5, 6, 7]]
    result = token_identity_rate(ref, [list(s) for s in ref])
    assert result.is_lossless
    assert result.token_rate == 1.0
    assert result.prompt_rate == 1.0
    assert result.first_divergence_positions == []


def test_single_differing_token_breaks_losslessness():
    """The instrument must be absolute, not tolerant: one token out of many
    is still a failed losslessness claim."""
    ref = [[1, 2, 3, 4, 5]] * 10
    cand = [[1, 2, 3, 4, 5]] * 10
    cand[7] = [1, 2, 9, 4, 5]  # one token, one prompt

    result = token_identity_rate(ref, cand)
    assert not result.is_lossless
    assert result.n_tokens_identical == 49
    assert result.n_tokens_compared == 50
    assert result.prompt_rate == 0.9
    assert result.first_divergence_positions == [2]


def test_early_truncation_counts_as_divergence():
    """A method that stops generating early is not reproducing the reference,
    even though every token it DID emit matches."""
    ref = [[1, 2, 3, 4, 5]]
    cand = [[1, 2, 3]]
    result = token_identity_rate(ref, cand)
    assert not result.is_lossless
    assert result.n_tokens_compared == 5
    assert result.n_tokens_identical == 3


# --- perplexity ------------------------------------------------------------


def test_perplexity_is_low_for_confident_correct_predictions():
    torch.manual_seed(0)
    B, T, V = 2, 6, 50
    targets = torch.randint(0, V, (B, T))
    logits = torch.full((B, T, V), -10.0)
    # make the model confidently predict the correct NEXT token at each step
    for b in range(B):
        for t in range(T - 1):
            logits[b, t, targets[b, t + 1]] = 10.0
    ppl = perplexity_from_logits(logits, targets)
    assert ppl < 1.1, f"confident-correct should give ppl near 1.0, got {ppl}"


def test_perplexity_is_high_for_confident_wrong_predictions():
    torch.manual_seed(1)
    B, T, V = 2, 6, 50
    targets = torch.randint(0, V, (B, T))
    logits = torch.zeros((B, T, V))
    for b in range(B):
        for t in range(T - 1):
            logits[b, t, targets[b, t + 1]] = -20.0  # actively avoid the truth
    ppl = perplexity_from_logits(logits, targets)
    assert ppl > 50, f"confident-wrong should give high ppl, got {ppl}"


def test_perplexity_detects_small_degradation():
    """The point of using perplexity at all: it must move on a degradation
    too small to flip any argmax (so task accuracy would see nothing)."""
    torch.manual_seed(2)
    B, T, V = 4, 20, 100
    targets = torch.randint(0, V, (B, T))

    good = torch.randn(B, T, V) * 0.1
    for b in range(B):
        for t in range(T - 1):
            good[b, t, targets[b, t + 1]] = 5.0

    slightly_worse = good.clone()
    for b in range(B):
        for t in range(T - 1):
            slightly_worse[b, t, targets[b, t + 1]] = 4.0  # still the argmax

    assert good.argmax(-1).equal(slightly_worse.argmax(-1)), "setup must not flip any argmax"

    ppl_good = perplexity_from_logits(good, targets)
    ppl_worse = perplexity_from_logits(slightly_worse, targets)
    assert ppl_worse > ppl_good, (
        f"perplexity failed to detect degradation invisible to argmax: "
        f"{ppl_good:.4f} -> {ppl_worse:.4f}"
    )


def test_perplexity_respects_attention_mask():
    torch.manual_seed(3)
    B, T, V = 2, 8, 30
    targets = torch.randint(0, V, (B, T))
    logits = torch.randn(B, T, V)

    full_mask = torch.ones(B, T)
    half_mask = torch.ones(B, T)
    half_mask[:, 4:] = 0

    ppl_full = perplexity_from_logits(logits, targets, full_mask)
    ppl_half = perplexity_from_logits(logits, targets, half_mask)
    assert ppl_full != ppl_half, "masked positions must not contribute"


# --- paired accuracy -------------------------------------------------------


def test_paired_test_counts_discordant_pairs():
    ref = [True, True, False, False, True]
    cand = [True, False, True, False, True]
    r = paired_accuracy_test(ref, cand)
    assert r.n == 5
    assert r.both_correct == 2
    assert r.ref_only == 1     # index 1: ref right, cand wrong
    assert r.cand_only == 1    # index 2: cand right, ref wrong
    assert r.neither == 1
    assert abs(r.delta) < 1e-9  # 3/5 vs 3/5


def test_paired_test_detects_a_real_regression():
    ref = [True] * 100
    cand = [True] * 70 + [False] * 30  # 30pp regression, all discordant
    r = paired_accuracy_test(ref, cand)
    assert r.ref_only == 30
    assert r.cand_only == 0
    assert r.delta == pytest.approx(-0.30)
    assert r.significant_at_05, "a 30pp regression at n=100 must be significant"


def test_paired_test_detects_a_small_one_sided_regression():
    """The case VALIDATION_PLAN.md relies on: a candidate that only ever
    regresses (never improves) is detectable at small effect sizes, because
    every discordant pair points the same way. 2pp at n=400."""
    n = 400
    ref = [True] * n
    cand = [True] * (n - 8) + [False] * 8  # 2pp regression, one-sided
    r = paired_accuracy_test(ref, cand)
    assert r.cand_only == 0
    assert r.ref_only == 8
    assert r.significant_at_05, (
        "an 8-item one-sided regression at n=400 should be detectable by McNemar"
    )


def test_paired_test_does_not_cry_wolf_on_noise():
    """Equal numbers of regressions and improvements is noise, not a real
    difference, and McNemar must not report it as significant."""
    ref = [True] * 50 + [False] * 50
    cand = [True] * 45 + [False] * 5 + [True] * 5 + [False] * 45
    r = paired_accuracy_test(ref, cand)
    assert r.ref_only == r.cand_only
    assert not r.significant_at_05


def test_min_detectable_delta_matches_the_documented_noise_floor():
    """Guards the numbers quoted in VALIDATION_PLAN.md #4.1 so the doc and
    the code cannot drift apart. An earlier draft of that doc quoted figures
    roughly 2x too optimistic (14pp at n=50); this test caught it and the
    doc was corrected to match the real formula, not the reverse."""
    assert 0.26 < min_detectable_delta(50) < 0.30     # doc: ~28pp
    assert 0.18 < min_detectable_delta(100) < 0.22    # doc: ~20pp
    assert 0.09 < min_detectable_delta(400) < 0.11    # doc: ~10pp
    assert 0.05 < min_detectable_delta(1000) < 0.07   # doc: ~6pp
    assert min_detectable_delta(1000) < min_detectable_delta(50)


def test_min_detectable_delta_reproduces_the_standard_reference_point():
    """Detecting 5pp around p=0.5 at 80% power is a textbook n~=1570 per
    group. If this drifts, the formula is wrong."""
    assert min_detectable_delta(1570) == pytest.approx(0.05, abs=0.002)


def test_paired_bound_is_far_tighter_than_unpaired():
    """The justification for n=400 in VALIDATION_PLAN.md: paired one-sided
    detection is roughly an order of magnitude more sensitive than the
    unpaired bound at the same n."""
    assert min_detectable_delta_paired(400) < 0.02
    assert min_detectable_delta_paired(400) < min_detectable_delta(400) / 5
