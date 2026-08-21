"""Certified greedy argmax via exact maximum-inner-product bounds.

This method is intentionally limited to greedy decoding.  Sampling needs the
entire probability distribution and therefore uses the normal full head.
Every inconclusive search falls back to the caller's full projection.
"""
from __future__ import annotations

import dataclasses
import math

import torch


@dataclasses.dataclass(frozen=True)
class MIPSBlock:
    row_start: int
    row_end: int
    center: torch.Tensor
    radius: float
    max_l1: float
    coord_min: torch.Tensor
    coord_max: torch.Tensor


@dataclasses.dataclass
class MIPSIndex:
    blocks: list[MIPSBlock]
    rows: int
    dims: int
    block_rows: int
    weights64: torch.Tensor = dataclasses.field(repr=False)

    @property
    def nbytes(self) -> int:
        tensor_bytes = self.weights64.numel() * self.weights64.element_size()
        for block in self.blocks:
            for tensor in (block.center, block.coord_min, block.coord_max):
                tensor_bytes += tensor.numel() * tensor.element_size()
        return tensor_bytes

    @classmethod
    def build(cls, weights: torch.Tensor, block_rows: int = 256) -> "MIPSIndex":
        if weights.ndim != 2:
            raise ValueError("MIPS weights must be a matrix")
        cpu = weights.detach().to(device="cpu", dtype=torch.float64)
        blocks = []
        for start in range(0, cpu.shape[0], block_rows):
            end = min(cpu.shape[0], start + block_rows)
            part = cpu[start:end]
            center = part.mean(dim=0)
            radius = float(torch.linalg.vector_norm(part - center, dim=1).max())
            max_l1 = float(part.abs().sum(dim=1).max())
            blocks.append(MIPSBlock(start, end, center, radius, max_l1,
                                   part.min(dim=0).values,
                                   part.max(dim=0).values))
        return cls(blocks, int(cpu.shape[0]), int(cpu.shape[1]), block_rows, cpu)


@dataclasses.dataclass(frozen=True)
class CertifiedArgmaxResult:
    index: int
    certified: bool
    rows_evaluated: int
    blocks_pruned: int
    certificate_margin: float


def _gamma(n: int, unit_roundoff: float) -> float:
    value = n * unit_roundoff
    return value / (1.0 - value) if value < 1.0 else float("inf")


def certified_argmax(query: torch.Tensor, weights: torch.Tensor, index: MIPSIndex,
                     *, fp32_accumulation: bool = True,
                     fallback=None) -> CertifiedArgmaxResult:
    """Return an argmax, proving it when interval bounds separate the winner.

    Bounds cover a real-valued dot product plus the standard sequential FP32
    accumulation error bound. GPU kernels may use a different reduction tree;
    that only tightens the usual ``gamma_n`` worst case. If reduced-precision
    accumulation is enabled, use ``fp32_accumulation=False``; the wider BF16
    bound will usually force the safe full fallback.
    """
    q = query.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    if q.numel() != index.dims or weights.shape != (index.rows, index.dims):
        raise ValueError("query/weight shape does not match MIPS index")
    w_cpu = index.weights64
    qmax = float(q.abs().max()) if q.numel() else 0.0
    unit = 2.0 ** -24 if fp32_accumulation else 2.0 ** -8
    gamma = _gamma(index.dims, unit)
    gamma64 = _gamma(index.dims, 2.0 ** -53)
    output_unit = {
        torch.bfloat16: 2.0 ** -8,
        torch.float16: 2.0 ** -11,
        torch.float32: 2.0 ** -24,
        torch.float64: 2.0 ** -53,
    }.get(weights.dtype)
    if output_unit is None:
        raise ValueError("unsupported MIPS projection dtype %s" % weights.dtype)
    # The matmul may accumulate in FP32 and still round its returned logits
    # to BF16/FP16. Both stages must be enclosed before claiming the argmax.
    arithmetic_error = gamma + output_unit * (1.0 + gamma)

    ordered = []
    for block in index.blocks:
        # A coordinate box is looser than a center/radius ball but admits a
        # transparent outward-rounded bound. Each row lies inside the box,
        # so selecting the favorable endpoint for every query coordinate is
        # an upper bound on every real dot product in the block.
        box_terms = torch.maximum(block.coord_min * q, block.coord_max * q)
        real_upper = float(box_terms.sum())
        real_upper += gamma64 * float(box_terms.abs().sum())
        real_upper = math.nextafter(real_upper, float("inf"))
        roundoff = arithmetic_error * qmax * block.max_l1
        ordered.append((real_upper + roundoff, block))
    ordered.sort(key=lambda item: item[0], reverse=True)

    best_index = -1
    best_lower = -float("inf")
    best_upper = -float("inf")
    unevaluated_upper = -float("inf")
    rows_evaluated = 0
    blocks_pruned = 0
    for block_upper, block in ordered:
        if block_upper < best_lower:
            blocks_pruned += 1
            unevaluated_upper = max(unevaluated_upper, block_upper)
            continue
        part = w_cpu[block.row_start:block.row_end]
        scores = part @ q
        errors = (arithmetic_error + gamma64) * (part.abs() @ q.abs())
        lowers = scores - errors
        uppers = scores + errors
        local = int(torch.argmax(lowers))
        if float(lowers[local]) > best_lower:
            best_index = block.row_start + local
            best_lower = float(lowers[local])
            best_upper = float(uppers[local])
        rows_evaluated += block.row_end - block.row_start

    # Competing evaluated rows need to be bounded too, not only pruned blocks.
    all_scores = w_cpu @ q if rows_evaluated == index.rows else None
    if all_scores is not None:
        all_errors = (arithmetic_error + gamma64) * (w_cpu.abs() @ q.abs())
        competitor = torch.cat((all_scores[:best_index] + all_errors[:best_index],
                                all_scores[best_index + 1:] + all_errors[best_index + 1:]))
        competitor_upper = float(competitor.max()) if competitor.numel() else -float("inf")
    else:
        # Re-evaluate only visited blocks to find the strongest non-winner
        # upper bound. This does not touch pruned rows.
        competitor_upper = unevaluated_upper
        for block_upper, block in ordered:
            if block_upper < best_lower:
                continue
            part = w_cpu[block.row_start:block.row_end]
            scores = part @ q
            errors = (arithmetic_error + gamma64) * (part.abs() @ q.abs())
            for offset, upper in enumerate((scores + errors).tolist()):
                if block.row_start + offset != best_index:
                    competitor_upper = max(competitor_upper, float(upper))
    margin = best_lower - competitor_upper
    if best_index >= 0 and margin > 0:
        return CertifiedArgmaxResult(best_index, True, rows_evaluated,
                                     blocks_pruned, margin)

    if fallback is None:
        # Use the original dtype/device for the same full projection shape a
        # normal greedy head would execute.
        fallback_scores = torch.mv(weights, query.reshape(-1).to(weights.device,
                                                                  dtype=weights.dtype))
        fallback_index = int(torch.argmax(fallback_scores))
    else:
        fallback_index = int(fallback())
    return CertifiedArgmaxResult(fallback_index, False, rows_evaluated,
                                 blocks_pruned, margin)
