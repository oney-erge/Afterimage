import torch

from afterimage.runtime.verify import sample_categorical, speculative_sample_step


def _empirical_first_token_dist(draft_probs, target_probs, bonus_probs, n_trials, vocab_size, seed):
    gen = torch.Generator().manual_seed(seed)
    counts = torch.zeros(vocab_size)
    for _ in range(n_trials):
        draft_tokens = [sample_categorical(p, gen) for p in draft_probs]
        tokens, _ = speculative_sample_step(draft_probs, target_probs, draft_tokens, bonus_probs, gen)
        counts[tokens[0]] += 1
    return counts / counts.sum()


def test_matches_target_distribution_when_draft_equals_target():
    torch.manual_seed(0)
    vocab = 8
    k = 3
    target = torch.softmax(torch.randn(k, vocab), dim=-1)
    draft = target.clone()
    bonus = torch.softmax(torch.randn(vocab), dim=-1)

    empirical = _empirical_first_token_dist(list(draft), list(target), bonus, n_trials=20000,
                                             vocab_size=vocab, seed=1)
    tvd = 0.5 * (empirical - target[0]).abs().sum().item()
    assert tvd < 0.03, f"total variation distance too high: {tvd}"


def test_matches_target_distribution_when_draft_disagrees_with_target():
    """The core correctness guarantee of speculative sampling: the SAMPLED
    output distribution matches the target exactly regardless of how wrong
    the draft is -- only the acceptance rate (and therefore speed) depends
    on draft quality, never correctness."""
    torch.manual_seed(2)
    vocab = 10
    k = 2
    target = torch.softmax(torch.randn(k, vocab) * 2.0, dim=-1)
    draft = torch.softmax(torch.randn(k, vocab) * 2.0, dim=-1)  # deliberately unrelated to target
    bonus = torch.softmax(torch.randn(vocab), dim=-1)

    empirical = _empirical_first_token_dist(list(draft), list(target), bonus, n_trials=40000,
                                             vocab_size=vocab, seed=3)
    tvd = 0.5 * (empirical - target[0]).abs().sum().item()
    assert tvd < 0.03, f"total variation distance too high with mismatched draft: {tvd}"


def test_full_acceptance_appends_bonus_token():
    target = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
    draft = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
    draft_tokens = [0]
    bonus = torch.tensor([0.0, 1.0, 0.0, 0.0])  # deterministic bonus token = 1
    gen = torch.Generator().manual_seed(0)
    tokens, n_accepted = speculative_sample_step(list(draft), list(target), draft_tokens, bonus, gen)
    assert n_accepted == 1
    assert tokens == [0, 1]


def test_certain_rejection_resamples_from_residual():
    # draft is CERTAIN of token 0, target puts zero mass on token 0 -- must
    # always reject and resample from target's remaining mass.
    draft = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.5, 0.3, 0.2]])
    draft_tokens = [0]
    bonus = torch.tensor([0.25, 0.25, 0.25, 0.25])
    gen = torch.Generator().manual_seed(0)
    tokens, n_accepted = speculative_sample_step(list(draft), list(target), draft_tokens, bonus, gen)
    assert n_accepted == 0
    assert tokens[0] != 0
