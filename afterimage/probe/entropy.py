"""Lossless-compression headroom of real model weights.

Measures the Shannon entropy of the floating-point *fields* of a weight
tensor, which is the information-theoretic floor for any lossless coder that
treats those fields as symbols. This is the DFloat11 mechanism
(arXiv:2504.11651) measured directly rather than assumed.

Why the exponent is the compressible part
-----------------------------------------
An IEEE float splits into sign | exponent | mantissa:

    bf16 :  1 |  8 | 7
    fp16 :  1 |  5 | 10

Trained weights are concentrated near zero with a roughly bell-shaped
magnitude distribution, so the *exponent* takes only a handful of distinct
values with a highly skewed distribution -- low entropy, far below the 8 bits
bf16 allocates to it. The sign is ~1 fair bit and the mantissa is very close
to uniform, so neither compresses meaningfully. Entropy-coding the exponent
alone is therefore where essentially all lossless gain comes from, and it is
bit-exact by construction: nothing is discarded, the field is just spelled
with a shorter code.

bf16 vs fp16 -- measured, and NOT what a naive reading predicts
---------------------------------------------------------------
An earlier version of this docstring asserted "bf16 compresses much better
than fp16," reasoning that bf16 wastes ~5.4 bits on its 8-bit exponent while
fp16 wastes only ~2.4 on its 5-bit one. The audit of a real checkpoint
contradicted that: Qwen2.5-1.5B compresses to 66.1% as bf16 and 66.2% as
fp16 -- effectively identical.

The reason is that modern checkpoints are *natively bf16*. Converting one to
fp16 widens the mantissa from 7 bits to 10 and zero-pads the low 3, adding no
information. Entropy coding then recovers those 3 padding bits, so fp16's
larger mantissa waste exactly offsets its smaller exponent waste. The claim
does hold when the source is genuinely fp32 (measured: bf16 65.7% vs fp16
84.4%), which is why the synthetic test agreed with it and the real model did
not.

Practical consequence: for a native-bf16 checkpoint, the dtype you *serve* in
does not change the compression ratio, so pick it for numerics, not for size.
"""
from __future__ import annotations

import dataclasses

import torch


def _entropy_bits(counts: torch.Tensor) -> float:
    """Shannon entropy in bits of an empirical symbol distribution."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0].double() / total.double()
    return float(-(p * torch.log2(p)).sum().item())


@dataclasses.dataclass
class EntropyReport:
    n_weights: int
    dtype: str
    sign_bits: int
    exponent_bits: int
    mantissa_bits: int
    exponent_entropy: float
    sign_entropy: float
    mantissa_entropy: float
    n_distinct_exponents: int

    @property
    def original_bits_per_weight(self) -> int:
        return self.sign_bits + self.exponent_bits + self.mantissa_bits

    @property
    def compressed_bits_per_weight(self) -> float:
        """Entropy floor if each field is coded independently at its entropy.

        Independent per-field coding is a slight over-estimate of the floor
        (a joint coder could exploit any correlation between fields), so this
        is a conservative, achievable target rather than a hard bound.
        """
        return self.sign_entropy + self.exponent_entropy + self.mantissa_entropy

    @property
    def compression_ratio(self) -> float:
        return self.original_bits_per_weight / max(self.compressed_bits_per_weight, 1e-9)

    @property
    def size_fraction(self) -> float:
        """Compressed size as a fraction of original. DFloat11 reports ~0.70
        for bf16 models."""
        return self.compressed_bits_per_weight / self.original_bits_per_weight


def analyze_tensor(W: torch.Tensor) -> EntropyReport:
    """Field-wise entropy of a float16/bfloat16 tensor.

    Works on the raw bit pattern via a bitwise view, so it measures the
    actual stored representation rather than a re-derived approximation.
    """
    if W.dtype == torch.float16:
        sign_bits, exp_bits, man_bits = 1, 5, 10
        bits = W.detach().cpu().contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    elif W.dtype == torch.bfloat16:
        sign_bits, exp_bits, man_bits = 1, 8, 7
        bits = W.detach().cpu().contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    else:
        raise TypeError(f"expected float16 or bfloat16, got {W.dtype}")

    flat = bits.flatten()
    man_mask = (1 << man_bits) - 1
    exp_mask = (1 << exp_bits) - 1

    mantissa = flat & man_mask
    exponent = (flat >> man_bits) & exp_mask
    sign = (flat >> (man_bits + exp_bits)) & 1

    exp_counts = torch.bincount(exponent, minlength=1 << exp_bits)
    sign_counts = torch.bincount(sign, minlength=2)
    man_counts = torch.bincount(mantissa, minlength=1 << man_bits)

    return EntropyReport(
        n_weights=flat.numel(),
        dtype=str(W.dtype),
        sign_bits=sign_bits,
        exponent_bits=exp_bits,
        mantissa_bits=man_bits,
        exponent_entropy=_entropy_bits(exp_counts),
        sign_entropy=_entropy_bits(sign_counts),
        mantissa_entropy=_entropy_bits(man_counts),
        n_distinct_exponents=int((exp_counts > 0).sum().item()),
    )


def compressed_bytes(W: torch.Tensor) -> int:
    """Bytes this tensor would occupy at its measured entropy floor."""
    rep = analyze_tensor(W)
    return int(rep.n_weights * rep.compressed_bits_per_weight / 8)
