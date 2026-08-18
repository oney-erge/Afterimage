"""Engine configuration.

The default is strictly lossless, and that is deliberate: the comparison this
project is built around is against AirLLM, which does not quantize either.
Turning quantization on makes the head-to-head measure two different things,
so it is opt-in and never silent.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class EngineConfig:
    """
    quantize
        None   -- strictly lossless (DEFAULT). Output is bit-identical to the
                  original bf16 model: verified on real models by comparing
                  every weight, the forward-pass logits (max abs diff 0.0),
                  and the generated token ids.
        "q8"   -- OPT-IN LOSSY. Group-wise 8-bit quantization applied before
                  entropy coding. Measured on real layers: ~2.0x compression
                  at 0.55% mean relative output error, versus ~1.46x at 0.00%
                  for lossless. It compresses better than lossless does, and
                  0.55% is small enough that most deployments would not
                  notice -- but it is NOT lossless, so any run using it must
                  not be reported as such.

    chunk_size
        Symbols per independently-decodable chunk. Larger amortizes per-chunk
        overhead; smaller gives the GPU more parallel work. 1024 measured
        best on this hardware.

    block_chunks
        Chunks decoded per Triton program. Must be a power of 2. 32 (the
        warp width) measured fastest -- see gpu_decode_v2.decode_gpu_v2.

    max_bits
        Ceiling on Huffman code length, which bounds the decode LUT to
        2**max_bits entries. Only an upper bound; the real table is usually
        smaller.
    """

    quantize: str | None = None
    chunk_size: int = 1024
    block_chunks: int = 32
    max_bits: int = 16

    def __post_init__(self) -> None:
        if self.quantize not in (None, "q8"):
            raise ValueError(
                "quantize must be None (lossless) or 'q8', got %r" % (self.quantize,))
        if self.block_chunks & (self.block_chunks - 1) != 0:
            raise ValueError(
                "block_chunks must be a power of 2 (Triton tl.arange), got %d"
                % self.block_chunks)
        if not (1 <= self.max_bits <= 16):
            raise ValueError("max_bits must be in [1, 16], got %d" % self.max_bits)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1, got %d" % self.chunk_size)

    @property
    def is_lossless(self) -> bool:
        return self.quantize is None

    def describe(self) -> str:
        if self.is_lossless:
            return "LOSSLESS (bit-exact output)"
        return "LOSSY: quantize=%s -- output is NOT bit-exact" % self.quantize
