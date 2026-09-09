import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("slice_elems", [97, 1 << 20])
def test_reused_tables_are_bit_exact_and_tensor_local(monkeypatch, slice_elems):
    from afterimage.runtime import gpu_decode_v2 as gd
    from afterimage.runtime.compressed_store import compress_layer, decompress_layer_gpu

    original = gd.prepare_decode_tables
    seen = []

    def prepare(enc, device="cuda"):
        seen.append(id(enc))
        return original(enc, device)

    monkeypatch.setattr(gd, "prepare_decode_tables", prepare)
    # Two different exponent distributions ensure a stale codebook from the
    # first tensor cannot silently decode the second.
    for seed, scale in [(1, .01), (2, 10.)]:
        torch.manual_seed(seed)
        weight = (torch.randn(31, 59)*scale).to(torch.bfloat16)
        layer = compress_layer(weight, chunk_size=31)
        before = len(seen)
        output = decompress_layer_gpu(
            layer, max_slice_elems=slice_elems, reuse_decode_tables=True)
        assert torch.equal(output.cpu().view(torch.int16), weight.view(torch.int16))
        assert len(seen)-before == 1


def test_reused_tables_work_on_nondefault_stream_and_special_bit_patterns():
    from afterimage.runtime.compressed_store import compress_layer, decompress_layer_gpu

    bits = torch.tensor([0, -32768, 32640, -128, 32705, -63, 1, 127, 128,
                         16256, -16512, 32767], dtype=torch.int16).reshape(3, 4)
    layer = compress_layer(bits.view(torch.bfloat16), chunk_size=5)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output = decompress_layer_gpu(layer, max_slice_elems=7, reuse_decode_tables=True)
    stream.synchronize()
    assert torch.equal(output.cpu().view(torch.int16), bits)


@pytest.mark.parametrize("value", [0, -1, 3])
def test_decoder_rejects_invalid_block_width_before_allocating(value):
    from afterimage.runtime.gpu_decode_v2 import decode_gpu_v2

    with pytest.raises(ValueError, match="power of 2"):
        decode_gpu_v2(None, block_chunks=value)
