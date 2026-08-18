"""Full weight reconstruction: sign + mantissa (stored raw) + exponent
(entropy-coded) -> the exact original bf16 tensor (LOSSLESS_ENGINE.md Phase B).

entropy.py and gpu_decode_v2.py operate on the exponent field alone, since
that's the only field with exploitable structure (mantissa entropy measured
at 6.97/7 bits -- see docs/LOSSLESS_ENGINE.md #2). This module is the piece
that was still missing: taking a REAL weight tensor apart into its three
fields, entropy-coding only the compressible one, and putting the exact
original value back together from the pieces. Nothing here is new
information-theoretically; it's the plumbing that turns "we can compress
one field" into "we can losslessly compress a real tensor."

bf16 bit layout (MSB to LSB): [sign:1][exponent:8][mantissa:7].
sign_mantissa keeps sign and mantissa, zeroing the exponent's 8 bits, via
mask 0x807F (bit 15, and bits 6-0). Reconstruction ORs the decoded exponent
(shifted left 7, landing exactly in bits 7-14) back in -- the two pieces
never overlap by construction, so OR is exact, not approximate.
"""
from __future__ import annotations

import dataclasses

import torch

from .huffman_chunked import ChunkedEncoded, encode_chunked


@dataclasses.dataclass
class CompressedLayer:
    sign_mantissa: torch.Tensor  # uint8, packed (sign<<7)|mantissa, CPU
    encoded: ChunkedEncoded      # entropy-coded exponent field
    shape: tuple[int, ...]

    @property
    def compressed_bytes(self) -> int:
        return (self.sign_mantissa.numel()  # 1 byte/weight -- see compress_layer's note
                + self.encoded.packed.nbytes
                + self.encoded.sym_lut.nbytes + self.encoded.len_lut.nbytes)

    @property
    def original_bytes(self) -> int:
        return self.sign_mantissa.numel() * 2  # bf16, 2 bytes/weight


EXPONENT_SHIFT = 7


def compress_layer(W: torch.Tensor, chunk_size: int = 1024, max_bits: int = 16) -> CompressedLayer:
    """sign_mantissa is packed into ONE byte per weight, not stored as
    int16. sign (bit 15) and mantissa (bits 6-0) are 8 bits of real content
    but are NOT contiguous in the original 16-bit word -- the 8-bit
    exponent sits between them (bits 14-7) -- so they must be extracted and
    repacked as (sign << 7) | mantissa, not just masked in place. An
    earlier version masked sign+mantissa in place and stored the result as
    int16 (2 bytes/weight): that wastes a full extra byte per weight
    encoding nothing, since the masked-out exponent bits were simply zero,
    not informative padding. At 2 bytes/weight for sign_mantissa ALONE, the
    "compressed" size could never beat the original 2-bytes/weight bf16
    tensor no matter how well the exponent compressed -- caught by
    test_compression_reduces_size_at_real_layer_scale measuring a REAL
    27 MB layer's "compressed" form at 32.5 MB, larger than the input.

    sign_mantissa is stored FLAT (not W's original shape), matching the
    flat exponent stream encode_chunked always produces; the original
    shape is restored by a single .reshape() at the end of decompression.
    """
    assert W.dtype == torch.bfloat16, f"expected bfloat16, got {W.dtype}"
    bits = W.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exponent = (bits >> EXPONENT_SHIFT) & 0xFF
    sign = (bits >> 15) & 1
    mantissa = bits & 0x7F
    sign_mantissa = ((sign << 7) | mantissa).to(torch.uint8).cpu().flatten()

    encoded = encode_chunked(exponent.flatten(), chunk_size=chunk_size, max_bits=max_bits)
    return CompressedLayer(sign_mantissa=sign_mantissa, encoded=encoded, shape=tuple(W.shape))


def _recombine(exponent: torch.Tensor, sign_mantissa: torch.Tensor,
               shape: tuple) -> torch.Tensor:
    """Rebuild bf16 bits from the three fields, entirely in int16.

    Deliberately avoids an int32 intermediate. The obvious formulation
    promotes everything to int32 first, which quadruples the working set
    relative to the uint8 inputs -- on a 778M-weight embedding that is a
    3.1 GB allocation for a tensor whose final form is 1.56 GB, and it is
    what blew the 4 GB VRAM cap even after the decoder's own scratch was
    already fixed to uint8.

    The sign bit needs care in int16: bit 15 IS the sign bit, so ORing it in
    would overflow. In two's complement, setting bit 15 of a value below
    32768 is exactly subtracting 32768, which is what the where() does.
    """
    e16 = exponent.to(torch.int16)
    sm16 = sign_mantissa.to(torch.int16)
    bits = (e16 << EXPONENT_SHIFT) | (sm16 & 0x7F)
    negative = sign_mantissa >= 128
    bits = torch.where(negative, bits - 32768, bits)
    return bits.view(torch.bfloat16).reshape(shape)


def _slice_encoded(enc: ChunkedEncoded, c0: int, c1: int) -> ChunkedEncoded:
    """A view of chunks [c0, c1) as a standalone ChunkedEncoded.

    Possible only because chunks are independently decodable by
    construction (huffman_chunked.py): each has its own byte-aligned
    bitstream and its own offset, so any contiguous run of them decodes
    without reference to the rest.
    """
    byte_start = int(enc.chunk_offsets[c0])
    nbytes = enc.chunk_nbytes[c0:c1]
    byte_end = int(enc.chunk_offsets[c1 - 1]) + int(nbytes[-1])
    # +8 slack: the reader refills ahead of the last symbol (see
    # encode_chunked's note on why the packed buffer carries a tail).
    byte_end = min(byte_end + 8, enc.packed.shape[0])

    return ChunkedEncoded(
        packed=enc.packed[byte_start:byte_end],
        chunk_offsets=(enc.chunk_offsets[c0:c1] - byte_start),
        chunk_nbytes=nbytes,
        sym_lut=enc.sym_lut,
        len_lut=enc.len_lut,
        max_bits=enc.max_bits,
        chunk_size=enc.chunk_size,
        n_symbols=(c1 - c0) * enc.chunk_size,
        shape=((c1 - c0) * enc.chunk_size,),
    )


def decompress_layer_gpu(layer: CompressedLayer, device: str = "cuda",
                         max_slice_elems: int = 1 << 25) -> torch.Tensor:
    """Reconstructs the EXACT original bf16 tensor.

    Slicing happens at the DECODE level, not just the recombine. An earlier
    version decoded the whole stream first and only sliced the bit math
    afterwards -- which bounded nothing, because the full uint8 exponent
    (778 MB on a 151936x5120 embedding) plus the full sign/mantissa plus
    the full output were all live before the first slice was taken. That
    cost five consecutive OOM failures at 4 GB and 6 GB caps while looking
    like it should fit.

    Decoding chunk ranges is what actually bounds peak memory, and it is
    available for free: chunks were already made independently decodable so
    the GPU could work on them in parallel. The output tensor is still
    allocated in full (it is the layer's real weight and has to exist), so
    peak is output + one slice of scratch, not output + a full second copy.
    """
    from .gpu_decode_v2 import decode_gpu_v2

    enc = layer.encoded
    n = enc.n_symbols

    if n <= max_slice_elems:
        exponent = decode_gpu_v2(enc, device=device)
        sm = layer.sign_mantissa.to(device=device)
        return _recombine(exponent[:n], sm[:n], layer.shape)

    chunks_per_slice = max(1, max_slice_elems // enc.chunk_size)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)

    for c0 in range(0, enc.n_chunks, chunks_per_slice):
        c1 = min(c0 + chunks_per_slice, enc.n_chunks)
        sub = _slice_encoded(enc, c0, c1)
        exp = decode_gpu_v2(sub, device=device)

        s0 = c0 * enc.chunk_size
        s1 = min(c1 * enc.chunk_size, n)
        take = s1 - s0
        sm = layer.sign_mantissa[s0:s1].to(device=device)
        out[s0:s1] = _recombine(exp[:take], sm, (take,))
        del exp, sm

    return out.reshape(layer.shape)


def decompress_layer_cpu_reference(layer: CompressedLayer) -> torch.Tensor:
    """CPU oracle using the CPU reference decoder, independent of any GPU
    code -- what test_compressed_store.py checks decompress_layer_gpu
    against, so a GPU-specific bug can't hide behind a shared bug in the
    bit-recombination math."""
    from .huffman_chunked import decode_chunked_cpu_reference

    exponent = torch.from_numpy(decode_chunked_cpu_reference(layer.encoded)).to(torch.int32)
    return _recombine(exponent, layer.sign_mantissa, layer.shape)
