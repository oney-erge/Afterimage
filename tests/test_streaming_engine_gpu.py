"""End-to-end GPU tests for StreamingLosslessModel against a tiny synthetic
Qwen3 checkpoint (random weights, no network, no real download): proves the
new pipeline (raw binstore, row-gather, prefetch) reproduces a plain
in-memory forward pass bit-exactly, the same standard this project has held
every other piece of the engine to.

Requires CUDA + triton, like the rest of the GPU test suite; skips cleanly
otherwise.
"""
import types

import pytest
import torch
from safetensors.torch import save_file

try:
    import triton  # noqa: F401
    _HAS = torch.cuda.is_available()
except ImportError:
    _HAS = False

pytestmark = pytest.mark.skipif(not _HAS, reason="needs CUDA + triton")


def _build_tiny_model(tie: bool, seed: int):
    from transformers import Qwen3Config, AutoModelForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=64, tie_word_embeddings=tie)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    model.eval()
    return cfg, model


def _make_store(tmp_path, monkeypatch, model, cfg, model_id, tag):
    from afterimage.runtime.streaming_engine import compress_model_to_disk

    snap = tmp_path / ("snap_" + tag)
    snap.mkdir()
    # A real tied checkpoint never stores lm_head.weight separately -- it is
    # meant to alias embed_tokens.weight (see _materialize_resident's
    # re-tying logic). state_dict() returns both names pointing at the same
    # storage, which safetensors correctly refuses to save twice; dropping
    # the alias here reproduces what a real tied checkpoint on disk actually
    # looks like.
    sd = model.state_dict()
    if cfg.tie_word_embeddings:
        sd.pop("lm_head.weight", None)
    save_file(sd, str(snap / "model.safetensors"))

    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda mid, **kw: str(snap))
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained",
                        lambda mid, **kw: cfg)

    store_dir = tmp_path / ("store_" + tag)
    compress_model_to_disk(model_id, store_dir, chunk_size=32, max_workers=2)
    return store_dir


def test_untied_row_gather_and_prefetch_are_bit_exact_vs_reference(tmp_path, monkeypatch):
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=0)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/untied-engine", "untied")

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(42)
    ids = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    with torch.no_grad():
        ref_logits = ref_model_gpu(input_ids=ids).logits

    # monkeypatch must still be active: StreamingLosslessModel's own
    # __init__ calls AutoConfig.from_pretrained too.
    sm_pre = StreamingLosslessModel("fake/untied-engine", store_dir, device="cuda",
                                    prefetch=True)
    assert sm_pre.manifest["tensors"]["model.embed_tokens.weight"]["row_gather"] is True
    with torch.no_grad():
        logits_prefetch = sm_pre.forward_logits(ids)
    sm_pre.close()

    sm_no = StreamingLosslessModel("fake/untied-engine", store_dir, device="cuda",
                                   prefetch=False)
    with torch.no_grad():
        logits_no_prefetch = sm_no.forward_logits(ids)
    sm_no.close()

    assert torch.equal(logits_prefetch, ref_logits), (
        "row-gathered + prefetched engine diverged from a plain reference "
        "forward pass on the identical weights")
    assert torch.equal(logits_no_prefetch, ref_logits), (
        "row-gathered engine (no prefetch) diverged from reference")
    assert torch.equal(logits_prefetch, logits_no_prefetch), (
        "prefetch changed the output -- it must only change WHEN bytes are "
        "read, never WHAT is read")


def test_tied_model_still_works_with_row_gather_storage_present(tmp_path, monkeypatch):
    """Regression check: a tied model must take the OLD full-materialization
    path for embed_tokens (row_gather absent from its manifest entry) and
    still reproduce reference logits exactly, now that row-gather exists as
    a code path it must correctly avoid."""
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=True, seed=1)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/tied-engine", "tied")

    assert "row_gather" not in store_dir.joinpath("manifest.json").read_text() \
        or True  # cheap sanity; the real assertion is below via the manifest object

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    with torch.no_grad():
        ref_logits = ref_model_gpu(input_ids=ids).logits

    sm = StreamingLosslessModel("fake/tied-engine", store_dir, device="cuda", prefetch=True)
    assert not sm.manifest["tensors"]["model.embed_tokens.weight"].get("row_gather")
    with torch.no_grad():
        logits = sm.forward_logits(ids)
    sm.close()

    assert torch.equal(logits, ref_logits)


def test_generate_greedy_matches_reference_argmax_sequence(tmp_path, monkeypatch):
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=2)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/greedy-engine", "greedy")

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device="cuda")

    def ref_generate_greedy(seq, n):
        with torch.no_grad():
            for _ in range(n):
                logits = ref_model_gpu(input_ids=seq).logits
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                seq = torch.cat([seq, nxt], dim=1)
        return seq

    expected = ref_generate_greedy(ids.clone(), 4)

    sm = StreamingLosslessModel("fake/greedy-engine", store_dir, device="cuda", prefetch=True)
    with torch.no_grad():
        got = sm.generate_greedy(ids.clone(), max_new_tokens=4)
    sm.close()

    assert torch.equal(got, expected)
