# FAQ

Short answers for first-time users. For the numbers behind any of these, see
the linked document.

## Why not just quantize the model?

Quantization changes the weights, which changes the model's outputs to some
degree. Afterimage exists for the opposite case: you want the original BF16
checkpoint's behavior, not an approximation of it, and are willing to trade
speed for that. If you don't need the original weights, a quantized runtime
(GGUF/llama.cpp, AWQ, GPTQ, and similar) is usually faster and simpler, and
is often the better choice.

## How is this different from AirLLM?

Both stream a model through the GPU a layer at a time so it fits in less
VRAM than the checkpoint needs. Afterimage adds a lossless compressed store
(so fewer bytes move per token), optional VRAM/RAM residency budgets, and
speculative decoding with a draft model. The [measured comparison](ALL_HYPOTHESES_AND_BASELINES.md)
shows neither engine wins everywhere: AirLLM wins at the exact lowest-VRAM
floor, Afterimage wins with residency and speculation turned on. See
[HOW_IT_WORKS.md](HOW_IT_WORKS.md) for the method-by-method contrast.

## Which models actually work?

Dense Qwen, Llama, and Mistral are the release-verified families. A shared
adapter now handles decoder traversal, and Qwen packed-MoE plus Qwen3-VL paths
are implemented as experimental capabilities. Other Hub models remain
downloadable even when Afterimage does not yet have an execution adapter.
Full evidence table:
[CONFIGURATION.md](CONFIGURATION.md#supported-model-architectures).

## Which hardware is actually validated?

The primary reference hardware is an NVIDIA RTX 3080 Laptop GPU (8 GB) under
WSL2/CUDA -- that's what every number in the [Results](../README.md#results)
table comes from. Native Linux + NVIDIA should behave the same. AMD/ROCm is
implemented but hasn't been run on real AMD hardware by this project. Native
Windows CUDA is less tested than the WSL2 path. macOS runs CPU-only (no CUDA
decode kernels).

## How slow is it, really?

Disk-bound: seconds per token, not tokens per second. On the reference 14B
model, the measured operating points are 9.150 s/token with speculation,
17.360 s/token with exact residency and no speculation, and 32.514 s/token
at the exact minimum-memory floor. `afterimage quickstart` proves the
pipeline works on your machine with a small model, but its timing doesn't
predict 14B performance -- your disk, CPU, and GPU all matter. See
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) for what to expect while a run is
in progress.

## What do "lossless" and "exact" actually mean here?

Several distinct, separately labeled things, not one blanket claim:

- The compressed **store** reconstructs the original BF16 weight bytes
  exactly -- that's the lossless compression codec.
- The default **execution** path is reference-execution equivalent: the same
  math as running the original checkpoint directly.
- **Speculative decoding at temperature zero** is provably greedy-token
  exact: the accepted tokens are identical to what the target model alone
  would produce.
- **`--quantize q8`** (compression time) and **`--lm-head-slice-rows`**
  (run time) are both opt-in and both **not** bit-exact -- they trade some
  accuracy for a smaller store or lower VRAM floor. Neither is a default.

## What should I expect for disk space, download, and compression time?

`afterimage compress MODEL --dry-run` estimates download size, compressed
store size, and peak disk usage (both the download and the store exist on
disk at once mid-pass) before you commit to anything. In the web UI, Get adds
compatibility inspection, persistent progress, pause, resume, cancel, and
store verification around that process. Reference: a 14B model was about 30
minutes to download and 6 minutes to compress on a 16-core machine; both vary
a lot with your connection and CPU.

## Do I need a draft model?

No. Without `--draft-model`, Afterimage runs exact streaming at the
measured 17.360 s/token (with a VRAM budget) or 32.514 s/token (minimum
memory). A draft model (`--draft-model`, e.g. `Qwen/Qwen3-0.6B`) enables
speculative decoding, the largest lossless speedup measured (9.150 s/token,
2.93x AirLLM 3.2.0) -- worth adding if you have the extra ~1.3 GB of VRAM for it.
