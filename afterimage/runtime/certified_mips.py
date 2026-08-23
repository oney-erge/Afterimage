"""Certified greedy argmax via exact maximum-inner-product bounds.

This method is intentionally limited to greedy decoding.  Sampling needs the
entire probability distribution and therefore uses the normal full head.
Every inconclusive search falls back to the caller's full projection.
"""
from __future__ import annotations

import dataclasses

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
    # Per-block coord_min/coord_max/max_l1, stacked across ALL blocks at
    # build time so certified_argmax can compute every block's upper bound
    # in one batched pass instead of one small tensor op per block. Measured
    # at ~0.15s of Python/dispatch overhead across 64 blocks on a
    # 16384-row slice (195 separate .abs() calls), almost all outside the
    # actual matmul work -- see docs/RESULTS_LOG.md speed audit. Values are
    # identical to iterating index.blocks one at a time; only the batching
    # changed, verified bit-for-bit against the per-block computation
    # (tests/test_certified_mips.py).
    coord_min_stack: torch.Tensor = dataclasses.field(repr=False, default=None)
    coord_max_stack: torch.Tensor = dataclasses.field(repr=False, default=None)
    max_l1_stack: torch.Tensor = dataclasses.field(repr=False, default=None)

    @property
    def nbytes(self) -> int:
        tensor_bytes = self.weights64.numel() * self.weights64.element_size()
        for block in self.blocks:
            for tensor in (block.center, block.coord_min, block.coord_max):
                tensor_bytes += tensor.numel() * tensor.element_size()
        for stacked in (self.coord_min_stack, self.coord_max_stack, self.max_l1_stack):
            if stacked is not None:
                tensor_bytes += stacked.numel() * stacked.element_size()
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
        if blocks:
            coord_min_stack = torch.stack([b.coord_min for b in blocks])
            coord_max_stack = torch.stack([b.coord_max for b in blocks])
            max_l1_stack = torch.tensor([b.max_l1 for b in blocks], dtype=torch.float64)
        else:
            coord_min_stack = torch.empty(0, cpu.shape[1], dtype=torch.float64)
            coord_max_stack = torch.empty(0, cpu.shape[1], dtype=torch.float64)
            max_l1_stack = torch.empty(0, dtype=torch.float64)
        return cls(blocks, int(cpu.shape[0]), int(cpu.shape[1]), block_rows, cpu,
                   coord_min_stack, coord_max_stack, max_l1_stack)


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
    # q never changes across the block loop below; q.abs() was being
    # recomputed once per block (64 times on a 16384-row/256-block-row
    # index) for a result that's identical every time.
    q_abs = q.abs()
    qmax = float(q_abs.max()) if q.numel() else 0.0
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

    # A coordinate box is looser than a center/radius ball but admits a
    # transparent outward-rounded bound. Each row lies inside the box, so
    # selecting the favorable endpoint for every query coordinate is an
    # upper bound on every real dot product in the block. Batched over all
    # blocks at once (index.*_stack, built in MIPSIndex.build) instead of
    # recomputing it with several small tensor ops per block -- the per-block
    # Python loop this replaced was almost pure dispatch overhead relative
    # to the actual evaluation work below. torch.nextafter matches
    # math.nextafter bit-for-bit for float64 (verified), so this is the same
    # outward rounding, just batched.
    if index.blocks:
        box_terms_all = torch.maximum(index.coord_min_stack * q, index.coord_max_stack * q)
        real_upper_all = box_terms_all.sum(dim=1)
        real_upper_all = real_upper_all + gamma64 * box_terms_all.abs().sum(dim=1)
        real_upper_all = torch.nextafter(real_upper_all,
                                         torch.full_like(real_upper_all, float("inf")))
        roundoff_all = arithmetic_error * qmax * index.max_l1_stack
        upper_all = (real_upper_all + roundoff_all).tolist()
        ordered = sorted(zip(upper_all, index.blocks), key=lambda item: item[0], reverse=True)
    else:
        ordered = []

    best_index = -1
    best_lower = -float("inf")
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
        errors = (arithmetic_error + gamma64) * (part.abs() @ q_abs)
        lowers = scores - errors
        local = int(torch.argmax(lowers))
        if float(lowers[local]) > best_lower:
            best_index = block.row_start + local
            best_lower = float(lowers[local])
        rows_evaluated += block.row_end - block.row_start

    # Competing evaluated rows need to be bounded too, not only pruned blocks.
    all_scores = w_cpu @ q if rows_evaluated == index.rows else None
    if all_scores is not None:
        all_errors = (arithmetic_error + gamma64) * (w_cpu.abs() @ q_abs)
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
            errors = (arithmetic_error + gamma64) * (part.abs() @ q_abs)
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
