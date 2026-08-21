# Implementation Status

Written after building the codebase, as a factual record of what is real,
what is verified, and what is not. Read this before trusting any claim made
elsewhere in this repository.

**UPDATE 2026-08-17: Phase 0 has now been run for real, on real GPU hardware,
against a real model — and the result is NO-GO.** See
[docs/PHASE0_RESULTS.md](docs/PHASE0_RESULTS.md) for the complete data. The
subspace cache does not have enough exploitable structure in a real
transformer (Qwen2.5-1.5B) to meet the success threshold, by roughly two to
three orders of magnitude. Everything below this notice describes the
codebase that was built to reach that answer; the codebase itself is sound
(67 tests passing, several real bugs caught during the real-model run and
fixed), but the research bet it was built to test did not pay off.

**Environment used for the real run:** WSL2 Ubuntu-24.04 on the same Windows
machine, with CUDA PyTorch 2.6.0+cu124, `transformers`, and `accelerate`
installed in a venv — not the CPU-only environment described below, which
was this repository's original state before that setup work. See
[docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md) Stage A for exactly how that
environment was brought up and verified (CUDA passthrough, O_DIRECT storage,
page-cache control).

**Everything below this line describes the ORIGINAL development environment
this codebase was built against, and is retained for the historical record of
what was validated before real hardware was available:**

**Original development environment:** Windows, Python 3.12, PyTorch 2.11
**CPU-only** (`torch.cuda.is_available() == False`), no `transformers`
package installed, no downloaded model weights, no NVMe rig matching
`configs/hardware.yaml`. Every number in this document below and every test
in `tests/` as originally written was measured on synthetic data or a small
synthetic model defined in `afterimage/testing/` — the real-model tests
(`test_masked_capture.py`, `test_wrapped_model_capture.py`, `test_fp16_basis.py`,
`test_workloads.py`, `test_layer_rank_report.py`) were added afterward,
specifically because running against a real model surfaced bugs the
synthetic tests could not have caught.

## What is real, implemented, and passing tests (67/67, `pytest tests/`)

| Component | File | What's verified |
|---|---|---|
| Online orthonormal basis | `runtime/basis.py` | Modified Gram-Schmidt + reorthogonalization; orthogonality (`‖UᵀU-I‖`) stays bounded over 500+ updates with eviction; LFU eviction; a **real float32 numerical bug** (rounding noise treated as new information, causing unbounded basis growth) was caught by test and fixed with relative-norm filtering |
| JL output-space gate | `runtime/gate.py` | Estimates `‖Wx⊥‖` from a precomputed `S=GW` sketch without ever touching `W`; concentration verified empirically against ground truth over 200 trials |
| Afterimage cache (single + batched) | `runtime/sketch.py` | **Central correctness claim verified to <1e-4 relative error**: for activations confined to a learned subspace, cache hits reproduce `Wx` almost exactly; batched verification fetches `W` exactly once per sweep regardless of batch size (**not** once per row) |
| Bit-plane precision ladder | `runtime/layout.py` | Reconstruction error shrinks monotonically with more planes; a **real quantization bug** (residual bound miscalculated, planes 2+ contributed nothing) was caught by test and fixed |
| Tiered storage + byte accounting | `runtime/tiers.py` | Real disk I/O (not simulated) for the NVMe tier; every read/write counted |
| Async double-buffered streaming | `runtime/streamer.py` | Real background-thread prefetch measurably overlaps with simulated compute (wall-clock verified) |
| Static residency planner | `runtime/resident.py` | Greedy knapsack respects budget and priority |
| Speculative sampling | `runtime/verify.py` | **The exact-distribution guarantee is empirically verified**, including the adversarial case where the draft and target distributions are unrelated (total variation distance < 0.03 over 40,000 trials) |
| Substitute draft (low-rank, SubSpec-style) | `runtime/draft.py` | Builds a cheap draft from the target's own weights via truncated SVD |
| Forward-hook activation capture | `probe/hooks.py` | Captures per-layer linear-layer inputs |
| Variance vs. functional rank curves | `probe/spectra.py` | **Directly demonstrates the rogue-dimension gap** HYPOTHESIS.md #3.1 predicts: a synthetic construction where 4 high-variance dimensions carry zero functional weight shows variance-based rank criteria reporting ~100% captured while functional error stays >50% |
| Closed-loop replay | `probe/closed_loop.py` | Transactional model patching (verified restored on every path); demonstrated to diverge from naive open-loop measurement on the toy transformer |
| Decode engine (draft+verify+cache) | `runtime/engine.py` | Full pipeline runs end to end; demonstrated `GB/token` improvement over the sequential baseline (3.9x-4.6x on the toy model, batching-dominated -- see below) |
| Sequential (AirLLM-equivalent) baseline | `baselines/b3_sequential.py` | Real no-cache, no-batch, fetch-every-layer-every-token control |
| Page-cache control | `bench/cachectl.py` | `free_ram_bytes()` verified working via real Windows API call; `drop_caches()` correctly reports unavailable on non-Linux rather than silently no-opping |
| Run-matrix harness | `bench/harness.py` | N-repeat aggregation, median/IQR, instability flagging, interleaved scheduling |
| Report formatting | `bench/report.py` | Table generation with instability markers |

Run it yourself:

```
pip install -e .
pytest tests/ -v
python scripts/run_probe_demo.py
python scripts/run_engine_demo.py
```

## What the demo scripts actually show (real output, reproduced here)

`run_engine_demo.py`, on the toy LM, showed the Afterimage cache's hit rate
at **0%** in its first run (unstructured random embedding table -- no
subspace exists to learn) and the entire 3.9x GB/token improvement came from
speculative batching alone. A second run with a deliberately low-rank
embedding table got the first layer's hit rate to 90%, but **it collapsed to
0% by the second layer** -- GELU and LayerNorm expand rank as depth
increases, even from a low-rank input. This is a small, honest, toy-scale
instance of exactly the risk HYPOTHESIS.md #6.2 raises from the published
literature (residual streams measuring ~90% effective rank in real models).
It was not cherry-picked or hidden; it is the actual output of the actual
code, printed in full.

**This is evidence about the toy model's specific nonlinearities, not
evidence about real transformers.** It is consistent with, and does not
resolve, the central open question the whole project depends on -- which
requires Phase 0 run against a real model (see below).

## What is NOT implemented

- **CUDA execution.** `tiers.py` has a VRAM/RAM device-selection path for
  CUDA but it has never run on a GPU. `streamer.py`'s overlap is
  thread-based, not `torch.cuda.Stream`-based -- functionally analogous, not
  the same mechanism, and the real PCIe/pinned-buffer staging path
  (IMPLEMENTATION_PLAN.md #5) is unbuilt.
- **Tree-based speculative verification.** `runtime/verify.py` implements a
  single linear draft *chain*, not SpecExec-style branching trees with tree
  attention masks. The chain case exercises every correctness-critical
  property (the exact-distribution guarantee) but a production system needs
  the tree to match SpecExec/SubSpec's reported acceptance-length numbers.
- **Clustered/piecewise subspaces** (HYPOTHESIS.md #3.3). Only a single
  global basis per layer is implemented. The `run_probe_demo.py`
  topic-switching output is printed specifically to flag where clustering
  would matter; it is not built.
- **Real model integration.** `baselines/b0_hf_offload.py`,
  `b1_airllm.py`, `b2_llamacpp.py` are real integration code written against
  each tool's documented API, but **none have been executed** -- this
  environment has no `transformers`, no `airllm` package, and no llama.cpp
  binary. Treat them as a starting point to validate on the real rig, not as
  tested components.
- **GPUDirect Storage.** Deliberately not attempted --
  LITERATURE.md #10 establishes it doesn't exist on consumer GeForce cards,
  so the staging-through-pinned-host-RAM path is the only one worth building
  regardless of platform.
- **Real checkpoint dimensions.** `configs/models.yaml` values are copied
  from public model cards, marked `UNVERIFIED`, and have not been checked
  against an actual downloaded checkpoint config.
- **Sensitivity calibration.** `gate.py`'s `GlobalController` defaults every
  layer's sensitivity `s_ℓ` to 1.0 (equal weighting). The plan's water-filling
  argument for *why* a single λ is optimal depends on real per-layer
  sensitivities, which requires calibration against a real model's logits
  (IMPLEMENTATION_PLAN.md Phase 0/4) -- not done here.

## Real bugs found by running against an actual model (not synthetic tests)

Every one of these was invisible to the toy-model test suite and only
surfaced when Phase 0 actually ran against Qwen2.5-1.5B-Instruct on CUDA.
Each is now fixed and covered by a dedicated regression test:

1. **Padding contamination.** Right-padded batches fed the pad token's own
   (near-constant) activation into rank measurements, artificially inflating
   variance-captured curves. Fixed by `hooks.py::stacked_masked` and
   attention-mask-aware `calibrate_bases`/`open_loop_error`. See
   `tests/test_masked_capture.py`.
2. **Infinite loop in a workload generator.** `probe/workloads.py`'s
   `topic_switch_prompts()` had a modular-arithmetic bug that, once the two
   smaller text pools were exhausted, permanently skipped the largest pool's
   index — the `while len(out) < total` loop never terminated. Found via a
   30+ minute real hang, not by inspection; the toy-model paths never called
   this function. See `tests/test_workloads.py`.
3. **Module-path mismatch through a wrapper.** `scripts/run_probe_real.py`'s
   `LogitsOnly` wrapper nests the real HF model one level deeper
   (`self.hf_model`), which shifts every submodule's dotted path. Passing
   unprefixed layer names silently matched nothing --
   `ActivationCapture.attach()` never created the corresponding dict entry,
   so the first read crashed with `KeyError` rather than failing loudly at
   the point of the actual mistake. See `tests/test_wrapped_model_capture.py`.
4. **fp16 SVD unsupported on both CUDA and CPU.** `closed_loop.py::_fit_basis`
   called `torch.linalg.svd` directly on activations, which are fp16 for any
   real model loaded at half precision for GPU memory. Neither cuSOLVER nor
   LAPACK implements SVD for half precision.
   `spectra.py::layer_rank_report` already upcast to float32 for the same
   reason and so never hit this. See `tests/test_fp16_basis.py`.
5. **Redundant SVD recomputation.** `functional_error_curve` recomputed a
   full SVD of the same matrix once per rank tested (7x redundant work per
   layer); `variance_rank_curve` and `effective_rank` each did their own
   independent SVD of the same activations on top of that (3x total). Not a
   correctness bug, but real enough on wide real-model layers (`d_in=8960`)
   to be worth fixing before it masked a genuine hang during debugging.
   Consolidated into `spectra.py::layer_rank_report`, one shared SVD per
   layer. See `tests/test_layer_rank_report.py`.

None of these would have been caught without actually running the code
against a real model -- which is itself evidence for why the plan's own
ordering (measure on real hardware before building the multi-week runtime)
was the right call.

## What this does and does not prove about the hypothesis

**Proves, from the synthetic test suite:** the mathematical mechanism in
HYPOTHESIS.md is internally consistent and correctly implementable -- the
exact-reproduction claim for subspace-confined activations, the JL-gate's
ability to estimate output error without fetching the weight, the batched
fetch-once amortization, and the speculative-sampling exact-distribution
guarantee are not just claimed, they are demonstrated against adversarial
synthetic tests designed to catch implementation errors (and did catch two
real ones, described earlier in this document).

**Now settled, from the real measurement:** real transformer activations do
NOT have low enough within-session functional rank for this mechanism to
matter at practical scale. **See
[docs/PHASE0_RESULTS.md](docs/PHASE0_RESULTS.md) for the full data.**
Functional error at rank 256 (2.9% of a real layer's width) was 25-45%
against a success threshold of <0.1% -- roughly 250-450x too high. End-to-end
closed-loop error with 6 layers truncated simultaneously was 60-96% even at
rank 128. This is not a marginal miss; it is a clear NO-GO, on real hardware,
against a real (if small, 1.5B) model. The toy-model engine demo's own
hit-rate collapse across layers, described below, turned out to be an
accurate small-scale preview of this result, not an artifact of the toy
model's specific nonlinearities.

## What should happen next

Per HYPOTHESIS.md #6 and EXECUTION_PLAN.md §6.1, both written *before* this
measurement ran: a failed gate means stop, not scale up and re-check. The
subspace cache is not a viable research direction on this evidence, and nothing
suggests a larger model would close a 250-450x gap. **Ship the fallback**:
residency + tree speculation (SubSpec/SpecExec-class methods,
IMPLEMENTATION_PLAN.md Phases 2-3), which was never dependent on this cache
working and remains a fully viable 27B-on-8GB product built from published,
reproducible methods.
