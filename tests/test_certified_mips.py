import math

import torch

from afterimage.runtime.certified_mips import MIPSIndex, certified_argmax


def _gamma(n, unit):
    v = n * unit
    return v / (1.0 - v) if v < 1.0 else float("inf")


def _reference_ordered(index, q64, gamma64, arithmetic_error, qmax):
    """The original per-block Python loop certified_argmax's bound pass used
    to run, kept here only as a reference to check the batched version
    against -- not something production code calls."""
    ordered = []
    for block in index.blocks:
        box_terms = torch.maximum(block.coord_min * q64, block.coord_max * q64)
        real_upper = float(box_terms.sum())
        real_upper += gamma64 * float(box_terms.abs().sum())
        real_upper = math.nextafter(real_upper, float("inf"))
        roundoff = arithmetic_error * qmax * block.max_l1
        ordered.append((real_upper + roundoff, block))
    return ordered


def test_certified_argmax_matches_full_projection():
    weights = torch.tensor([[10.0, 0.0], [9.0, 0.0], [-5.0, 1.0], [-4.0, 1.0]])
    query = torch.tensor([1.0, 0.0])
    index = MIPSIndex.build(weights, block_rows=2)
    assert index.nbytes >= weights.numel() * 8
    result = certified_argmax(query, weights, index)
    assert result.index == int(torch.mv(weights, query).argmax())
    assert result.certified


def test_inconclusive_numeric_bounds_use_full_fallback():
    weights = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    query = torch.tensor([1.0, 0.0])
    index = MIPSIndex.build(weights, block_rows=1)
    result = certified_argmax(query, weights, index, fallback=lambda: 1)
    assert result.index == 1
    assert not result.certified


def test_bf16_output_rounding_is_included_in_certificate():
    weights = torch.tensor([[1.0], [1.003]], dtype=torch.bfloat16)
    query = torch.tensor([1.0], dtype=torch.bfloat16)
    index = MIPSIndex.build(weights, block_rows=1)
    result = certified_argmax(query, weights, index, fallback=lambda: 0)
    assert result.index == 0
    assert not result.certified


def test_batched_block_bounds_match_the_original_per_block_loop():
    """certified_argmax used to compute each block's upper bound with a
    small Python loop (torch.maximum/.sum()/.abs().sum()/math.nextafter per
    block); it's now one batched pass over index.*_stack. Both must produce
    the exact same per-block bound and therefore the same sorted order --
    checked directly against a reimplementation of the original loop, not
    just indirectly through certified_argmax's final answer."""
    torch.manual_seed(0)
    for rows, dims, block_rows in [(1000, 64, 37), (300, 12, 300), (2048, 128, 128)]:
        weights = (torch.randn(rows, dims) * 0.02)
        index = MIPSIndex.build(weights, block_rows=block_rows)
        query = torch.randn(dims)

        q64 = query.to(torch.float64)
        gamma64 = _gamma(dims, 2.0 ** -53)
        gamma = _gamma(dims, 2.0 ** -24)
        arithmetic_error = gamma + 2.0 ** -24 * (1.0 + gamma)
        qmax = float(q64.abs().max())

        reference = sorted(_reference_ordered(index, q64, gamma64, arithmetic_error, qmax),
                           key=lambda item: item[0], reverse=True)

        box_terms_all = torch.maximum(index.coord_min_stack * q64, index.coord_max_stack * q64)
        real_upper_all = box_terms_all.sum(dim=1)
        real_upper_all = real_upper_all + gamma64 * box_terms_all.abs().sum(dim=1)
        real_upper_all = torch.nextafter(real_upper_all,
                                         torch.full_like(real_upper_all, float("inf")))
        roundoff_all = arithmetic_error * qmax * index.max_l1_stack
        batched = sorted(zip((real_upper_all + roundoff_all).tolist(), index.blocks),
                         key=lambda item: item[0], reverse=True)

        assert len(reference) == len(batched)
        for (ref_upper, ref_block), (new_upper, new_block) in zip(reference, batched):
            assert ref_upper == new_upper, (rows, dims, block_rows)
            assert ref_block.row_start == new_block.row_start


def test_certified_argmax_correct_with_and_without_pruning():
    """Exercises both branch-and-bound regimes the evaluation loop can take:
    a skewed weight matrix where most blocks get pruned outright, and a
    uniform one where none do -- the batched bound pass must still hand the
    (unchanged) evaluation loop the right winner in both."""
    torch.manual_seed(1)
    dims, block_rows = 64, 32

    skewed = torch.randn(512, dims) * 0.001
    skewed[100:105] += torch.randn(5, dims) * 5.0
    query = torch.ones(dims)
    index = MIPSIndex.build(skewed, block_rows=block_rows)
    result = certified_argmax(query, skewed, index)
    true_argmax = int(torch.mv(skewed.double(), query.double()).argmax())
    assert result.index == true_argmax
    assert result.blocks_pruned > 0

    uniform = torch.randn(512, dims) * 0.02
    index2 = MIPSIndex.build(uniform, block_rows=block_rows)
    result2 = certified_argmax(query, uniform, index2)
    true_argmax2 = int(torch.mv(uniform.double(), query.double()).argmax())
    assert result2.index == true_argmax2
