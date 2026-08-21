"""Chunked lm_head projection -- lowers the VRAM floor, at the cost of
bit-exactness. OPT-IN, and NOT part of the lossless path.

Peak VRAM for any layer-streaming engine is bounded below by the largest
tensor it must hold at once. On a 14B that is lm_head (151936 x 5120 bf16 =
1.556 GB), which is also exactly where AirLLM sits -- both systems are
pinned to the same floor by the same tensor, so no budget below ~1.7 GB is
expressible for either. Computing logits in row blocks removes that floor,
because logits are a concatenation over output rows with no interaction
between blocks:

    logits[..., a:b] = x @ W[a:b].T

WHAT THAT COSTS, measured rather than assumed. The identity above is exact
in real arithmetic and the decompressed WEIGHTS are still bit-exact, but the
matmul is not: cuBLAS picks a different kernel and split-K reduction
strategy per output shape, so a blocked product accumulates in a different
order than one full product and bf16 rounding diverges. Measured at real 14B
dimensions (hidden=5120): up to 2.0 absolute logit deviation. Forcing fp32
accumulation via allow_bf16_reduced_precision_reduction=False was tested and
does NOT fix it (2.0 -> 1.0).

At the tiny dimensions this test file can afford (hidden=64) it happens to
come out bit-identical, which is precisely why that must NOT be asserted as
a guarantee here -- doing so would encode a false promise that only holds
because the test model is small. test_blocked_matmul_is_not_bit_exact_at_
production_shape below pins the real behaviour instead.
"""
import json

import pytest
import torch

try:
    import triton  # noqa: F401
    _HAS = torch.cuda.is_available()
except ImportError:
    _HAS = False

pytestmark = pytest.mark.skipif(not _HAS, reason="needs CUDA + triton")

from test_streaming_engine_gpu import _build_tiny_model, _make_store


def _build_big_head_model(seed: int):
    """A model whose lm_head is large enough to be entropy-coded.

    The default tiny model's head is 97x32 = 3104 elements, under
    compress_model_to_disk's 65536-element threshold, so it is stored RAW
    and the chunked path (which decodes Huffman chunk ranges) correctly
    declines to engage. 2048x64 = 131072 clears the threshold, and
    64 cols / 32 chunk_size divides evenly, so row boundaries land on chunk
    boundaries -- the same alignment the real 14B has (5120/1024 = 5).
    """
    return _build_tiny_model(tie=False, seed=seed, intermediate_size=64,
                             vocab_size=2048, hidden_size=64)


def test_blocked_matmul_is_not_bit_exact_at_production_shape():
    """The measured reason this feature is opt-in and marked lossy.

    This is a characterization test: it documents real, surprising hardware
    behaviour that the tiny-model tests cannot reveal. If a future PyTorch
    or cuBLAS makes blocked and unblocked products agree bit-for-bit, this
    test failing is GOOD NEWS -- it means lm_head_slice_rows could then be
    promoted to the lossless path. Do not "fix" it by loosening the
    assertion; re-derive whether the feature can become lossless.
    """
    torch.manual_seed(0)
    seq, hidden, vocab = 8, 5120, 9496  # real 14B hidden size
    x = torch.randn(1, seq, hidden, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(vocab, hidden, dtype=torch.bfloat16, device="cuda")

    full = torch.nn.functional.linear(x, W)
    blocked = torch.cat(
        [torch.nn.functional.linear(x, W[r:min(r + 2374, vocab)])
         for r in range(0, vocab, 2374)], dim=-1)

    assert not torch.equal(blocked, full), (
        "blocked and unblocked matmul now agree bit-for-bit at production "
        "shape -- see this test's docstring: lm_head_slice_rows may be "
        "promotable to the lossless path, re-check before changing this")
    # Close, just not identical -- the deviation is bf16 rounding, not a bug.
    assert torch.allclose(blocked.float(), full.float(), atol=4.0)


def test_chunked_head_config_declares_itself_lossy():
    from afterimage.runtime.config import EngineConfig

    assert EngineConfig().is_lossless
    cfg = EngineConfig(lm_head_slice_rows=8192)
    assert not cfg.is_lossless
    assert "NOT bit-exact" in cfg.describe()
    assert "lm_head_slice_rows" in cfg.describe()


def test_chunked_lm_head_matches_whole_head_closely(tmp_path, monkeypatch):
    """End-to-end: the chunked path must use the same weights in the same
    projection, so logits agree to within bf16 blocking noise and the
    decoded weights themselves stay exact. Bit-equality is deliberately NOT
    asserted -- see the module docstring."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_big_head_model(seed=71)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/chunked-head", "chunked_head")

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(31)
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    with torch.no_grad():
        ref_logits = ref_model_gpu(input_ids=ids).logits

    whole = StreamingLosslessModel("fake/chunked-head", store_dir, device="cuda",
                                   config=EngineConfig(io_prefetch_depth=1))
    with torch.no_grad():
        logits_whole = whole.forward_logits(ids)
    whole.close()
    # The lossless path is still exactly that.
    assert torch.equal(logits_whole, ref_logits)

    # 7 deliberately does NOT divide 2048 -- the last block is short and row
    # boundaries land mid-chunk, exercising the general covering-chunk-range
    # path rather than only the aligned fast case.
    for step in (2048, 512, 128, 7):
        sm = StreamingLosslessModel(
            "fake/chunked-head", store_dir, device="cuda",
            config=EngineConfig(io_prefetch_depth=1, lm_head_slice_rows=step))
        assert sm._chunked_head_rows(sm.config) == step
        with torch.no_grad():
            logits_chunked = sm.forward_logits(ids)
        # lm_head must never be materialized as a parameter at all
        assert sm.model.lm_head.weight.device.type == "meta"
        sm.close()
        torch.testing.assert_close(logits_chunked, ref_logits,
                                   rtol=0.05, atol=0.5,
                                   msg="chunked lm_head (step=%d) diverged beyond "
                                       "bf16 blocking noise" % step)


def test_chunked_lm_head_makes_smaller_budgets_feasible(tmp_path, monkeypatch):
    """The actual point of the feature: a budget the planner refuses when
    lm_head must be materialized whole becomes reachable when it never is."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel
    from afterimage.runtime.vram_planner import (
        DEFAULT_ACTIVATION_SLACK_BYTES, plan_from_manifest)

    cfg, ref_model = _build_big_head_model(seed=72)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/chunked-floor", "chunked_floor")
    man = json.loads((store_dir / "manifest.json").read_text())
    head = man["tensors"]["lm_head.weight"]
    head_bytes, head_rows = head["orig_bytes"], head["shape"][0]

    # Room for the activation slack and a quarter of the head, but not the
    # whole head.
    budget_gb = (DEFAULT_ACTIVATION_SLACK_BYTES + head_bytes // 4) / 1e9

    whole = plan_from_manifest(man, vram_budget_gb=budget_gb, decode_slice_elems=64)
    assert not whole.feasible, "test needs a budget whole-head planning refuses"

    chunked = plan_from_manifest(man, vram_budget_gb=budget_gb, decode_slice_elems=64,
                                 stream_only={"lm_head.weight": head_bytes // 8})
    assert chunked.feasible, "chunked head should reach this budget: " + chunked.reason
    assert "lm_head.weight" in chunked.disk_keys
    assert "lm_head.weight" not in chunked.vram_keys

    ref_gpu = ref_model.to("cuda")
    torch.manual_seed(5)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device="cuda")
    with torch.no_grad():
        ref_logits = ref_gpu(input_ids=ids).logits

    sm = StreamingLosslessModel(
        "fake/chunked-floor", store_dir, device="cuda",
        config=EngineConfig(io_prefetch_depth=1, vram_budget_gb=budget_gb,
                            decode_slice_elems=64,
                            lm_head_slice_rows=head_rows // 8))
    with torch.no_grad():
        logits = sm.forward_logits(ids)
    sm.close()
    torch.testing.assert_close(logits, ref_logits, rtol=0.05, atol=0.5)


def test_chunked_lm_head_declines_where_it_cannot_apply(tmp_path, monkeypatch):
    """Must be a no-op where it does not apply: a tied model has no separate
    lm_head tensor, and a small head is stored raw (no chunk ranges to
    decode). Both fall back to the normal path rather than crashing -- and
    the fallback stays bit-exact, since nothing was blocked."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg_tied, tied_model = _build_tiny_model(tie=True, seed=73)
    tied_store = _make_store(tmp_path, monkeypatch, tied_model, cfg_tied,
                             "fake/chunked-tied", "chunked_tied")
    ref_tied = tied_model.to("cuda")
    ids = torch.randint(0, cfg_tied.vocab_size, (1, 4), device="cuda")
    with torch.no_grad():
        ref_logits = ref_tied(input_ids=ids).logits

    sm = StreamingLosslessModel("fake/chunked-tied", tied_store, device="cuda",
                                config=EngineConfig(io_prefetch_depth=1,
                                                    lm_head_slice_rows=8))
    assert sm._chunked_head_rows(sm.config) == 0
    with torch.no_grad():
        logits = sm.forward_logits(ids)
    sm.close()
    assert torch.equal(logits, ref_logits)

    cfg_raw, raw_model = _build_tiny_model(tie=False, seed=74)  # 97x32 -> raw
    raw_store = _make_store(tmp_path, monkeypatch, raw_model, cfg_raw,
                            "fake/chunked-raw", "chunked_raw")
    sm2 = StreamingLosslessModel("fake/chunked-raw", raw_store, device="cuda",
                                 config=EngineConfig(io_prefetch_depth=1,
                                                     lm_head_slice_rows=8))
    assert sm2._chunked_head_rows(sm2.config) == 0
    sm2.close()


def test_decompress_rows_matches_full_decode():
    """The WEIGHTS side is exact and stays exact: block decoding must
    reassemble byte-for-byte what whole-tensor decoding produces. This is
    the part of the feature that IS lossless -- only the matmul blocking
    above is not."""
    from afterimage.runtime.compressed_store import (
        compress_layer, decompress_layer_gpu, decompress_rows_gpu)

    torch.manual_seed(17)
    W = torch.randn(300, 64, dtype=torch.float32).to(torch.bfloat16)
    layer = compress_layer(W, chunk_size=32)

    full = decompress_layer_gpu(layer, device="cuda")
    assert torch.equal(full, W.to("cuda"))

    for step in (300, 128, 64, 7, 1):
        parts = [decompress_rows_gpu(layer, r0, min(r0 + step, 300), device="cuda")
                 for r0 in range(0, 300, step)]
        assert torch.equal(torch.cat(parts, dim=0), full), "step=%d" % step

    assert decompress_rows_gpu(layer, 5, 5, device="cuda").shape == (0, 64)
