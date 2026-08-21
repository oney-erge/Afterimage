import pytest
import torch

from afterimage.runtime.xor_reference import (
    audit_reference_candidates, decode_xor_reference, encode_xor_reference,
    load_xor_reference, save_xor_reference,
)


def test_xor_reference_round_trips_every_bf16_bit():
    base = torch.randn(16, 8, dtype=torch.bfloat16)
    target = base.clone()
    target[0, 0] = -target[0, 0]
    blob = encode_xor_reference(base, target)
    restored = decode_xor_reference(base, blob)
    assert torch.equal(restored.view(torch.int16), target.view(torch.int16))


def test_wrong_base_is_rejected():
    base = torch.zeros(4, dtype=torch.bfloat16)
    target = torch.ones(4, dtype=torch.bfloat16)
    blob = encode_xor_reference(base, target)
    with pytest.raises(ValueError, match="checksum"):
        decode_xor_reference(torch.ones(4, dtype=torch.bfloat16), blob)


def test_audit_finds_compatible_reference():
    tensors = {"a": torch.zeros(32, dtype=torch.bfloat16),
               "b": torch.zeros(32, dtype=torch.bfloat16)}
    assert audit_reference_candidates(tensors)["b"]["base"] == "a"


def test_audit_can_restrict_references_to_an_acyclic_base_set():
    tensors = {"base": torch.zeros(32, dtype=torch.bfloat16),
               "target": torch.ones(32, dtype=torch.bfloat16)}
    audit = audit_reference_candidates(tensors, base_keys=["base"])
    assert audit["base"] is None
    assert audit["target"]["base"] == "base"


def test_xor_reference_artifact_round_trip(tmp_path):
    base = torch.zeros(32, dtype=torch.float16)
    target = torch.arange(32, dtype=torch.float16)
    path = tmp_path / "expert.aixor"
    save_xor_reference(encode_xor_reference(base, target), path)
    restored = decode_xor_reference(base, load_xor_reference(path))
    assert torch.equal(restored.view(torch.int16), target.view(torch.int16))
