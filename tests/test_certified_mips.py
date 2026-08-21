import torch

from afterimage.runtime.certified_mips import MIPSIndex, certified_argmax


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
