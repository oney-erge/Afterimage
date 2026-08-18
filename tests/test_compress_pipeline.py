"""End-to-end tests for the offline compression pipeline (streaming_engine.py
compress_model_to_disk): the raw binstore format (lever 1), parallel
compression (lever 4), and embedding row-gather storage (lever 2). Runs
against a small LOCAL safetensors fixture with huggingface_hub /
AutoConfig.from_pretrained monkeypatched out, so it needs no network access
and no GPU (decoding is checked via the CPU reference path).
"""
import types

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from afterimage.runtime.binstore import BinaryWeightReader
from afterimage.runtime.compressed_store import CompressedLayer, decompress_layer_cpu_reference
from afterimage.runtime.huffman_chunked import ChunkedEncoded
from afterimage.runtime.streaming_engine import compress_model_to_disk


def _fake_weights(seed: int, big_key: str = "model.embed_tokens.weight"):
    # Large enough that the Huffman LUT's fixed cost (up to 320 KB at the
    # max_bits=16 ceiling) is amortized away -- see
    # test_compressed_store.py::test_lut_fixed_cost_dominates_below_a_scale_
    # threshold for the same boundary documented at the codec level. A
    # fixture near that boundary made compression genuinely lose on a
    # synthetic tensor, which is real documented behavior, not a bug, but
    # the wrong scale to test "does compression shrink real-shaped weights"
    # at.
    g = torch.Generator().manual_seed(seed)
    return {
        big_key: (torch.randn(900, 900, generator=g) * 0.02).to(torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight":
            (torch.randn(1200, 900, generator=g) * 0.02).to(torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(300, dtype=torch.bfloat16),
    }


def _patch_hf(monkeypatch, snap_dir, tied: bool):
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda model_id, **kw: str(snap_dir))
    fake_cfg = types.SimpleNamespace(tie_word_embeddings=tied)
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained", lambda model_id, **kw: fake_cfg)


def _read_and_decode(store_dir, manifest, key):
    """CPU-only readback of one tensor, mirroring what
    StreamingLosslessModel._decode_tensor does but without needing CUDA."""
    meta = manifest["tensors"][key]
    with BinaryWeightReader(store_dir / "weights.bin") as reader:
        arrays = {name: reader.read(ref) for name, ref in meta["blobs"].items()}

    if meta.get("row_gather"):
        hidden = meta["hidden_size"]
        raw = arrays["raw"]
        return torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(meta["shape"][0], hidden)

    if not meta["compressed"]:
        out = torch.from_numpy(arrays["raw"])
        return out.to(torch.bfloat16)

    enc = ChunkedEncoded(
        packed=arrays["packed"], chunk_offsets=arrays["chunk_offsets"],
        chunk_nbytes=arrays["chunk_nbytes"], sym_lut=arrays["sym_lut"],
        len_lut=arrays["len_lut"], max_bits=int(meta["max_bits"]),
        chunk_size=int(meta["chunk_size"]), n_symbols=int(meta["n_symbols"]),
        shape=tuple(meta["shape"]),
    )
    layer = CompressedLayer(sign_mantissa=torch.from_numpy(arrays["sign_mantissa"]),
                            encoded=enc, shape=tuple(meta["shape"]))
    return decompress_layer_cpu_reference(layer)


@pytest.mark.parametrize("max_workers", [1, 4])
def test_untied_model_row_gathers_embedding_and_round_trips_bit_exact(
        tmp_path, monkeypatch, max_workers):
    snap = tmp_path / "snap"
    snap.mkdir()
    weights = _fake_weights(seed=0)
    save_file(weights, str(snap / "model.safetensors"))
    _patch_hf(monkeypatch, snap, tied=False)

    out_dir = tmp_path / "store"
    manifest = compress_model_to_disk("fake/untied", out_dir, chunk_size=64,
                                      max_workers=max_workers)

    embed_meta = manifest["tensors"]["model.embed_tokens.weight"]
    assert embed_meta["row_gather"] is True
    assert embed_meta["compressed"] is False

    recon_embed = _read_and_decode(out_dir, manifest, "model.embed_tokens.weight")
    assert torch.equal(recon_embed, weights["model.embed_tokens.weight"])

    recon_mlp = _read_and_decode(out_dir, manifest, "model.layers.0.mlp.down_proj.weight")
    assert torch.equal(recon_mlp, weights["model.layers.0.mlp.down_proj.weight"])
    assert manifest["tensors"]["model.layers.0.mlp.down_proj.weight"]["compressed"] is True

    recon_norm = _read_and_decode(out_dir, manifest, "model.layers.0.input_layernorm.weight")
    assert torch.equal(recon_norm, weights["model.layers.0.input_layernorm.weight"])


def test_tied_model_does_not_row_gather_embedding(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    weights = _fake_weights(seed=1)
    save_file(weights, str(snap / "model.safetensors"))
    _patch_hf(monkeypatch, snap, tied=True)

    out_dir = tmp_path / "store"
    manifest = compress_model_to_disk("fake/tied", out_dir, chunk_size=64, max_workers=2)

    embed_meta = manifest["tensors"]["model.embed_tokens.weight"]
    assert "row_gather" not in embed_meta or embed_meta["row_gather"] is False
    assert embed_meta["compressed"] is True, (
        "a tied model's embedding must go through the normal compression "
        "path -- lm_head aliases it and needs the full matrix regardless, "
        "so row-gather storage would gain nothing and only add a special "
        "case to _materialize_resident for no benefit")

    recon_embed = _read_and_decode(out_dir, manifest, "model.embed_tokens.weight")
    assert torch.equal(recon_embed, weights["model.embed_tokens.weight"])


def test_parallel_and_serial_compression_reconstruct_identical_weights(tmp_path, monkeypatch):
    """The property lever 4 must not break: which worker (or none) compressed
    a tensor must never change what comes back out. Compares reconstructed
    VALUES rather than raw file bytes, since imap_unordered may interleave
    tensors into weights.bin in a different order across runs -- offsets
    differing is fine, decoded content differing is not."""
    snap = tmp_path / "snap"
    snap.mkdir()
    weights = _fake_weights(seed=2)
    save_file(weights, str(snap / "model.safetensors"))
    _patch_hf(monkeypatch, snap, tied=False)

    out_serial = tmp_path / "store_serial"
    out_parallel = tmp_path / "store_parallel"
    man_serial = compress_model_to_disk("fake/x", out_serial, chunk_size=64, max_workers=1)
    man_parallel = compress_model_to_disk("fake/x", out_parallel, chunk_size=64, max_workers=4)

    assert set(man_serial["tensors"]) == set(man_parallel["tensors"])
    for key in weights:
        a = _read_and_decode(out_serial, man_serial, key)
        b = _read_and_decode(out_parallel, man_parallel, key)
        assert torch.equal(a, b), f"{key} differs between serial and parallel compression"
        assert torch.equal(a, weights[key])


def test_manifest_totals_match_written_file_size(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    weights = _fake_weights(seed=3)
    save_file(weights, str(snap / "model.safetensors"))
    _patch_hf(monkeypatch, snap, tied=False)

    out_dir = tmp_path / "store"
    manifest = compress_model_to_disk("fake/y", out_dir, chunk_size=64, max_workers=2)

    max_end = 0
    for meta in manifest["tensors"].values():
        for ref in meta["blobs"].values():
            max_end = max(max_end, ref["offset"] + ref["nbytes"])
    assert max_end == (out_dir / "weights.bin").stat().st_size
    assert manifest["total_orig_bytes"] > manifest["total_comp_bytes"]
