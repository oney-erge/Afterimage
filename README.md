<p align="center">
  <img src="docs/assets/afterimage-logo.svg" width="112" alt="Afterimage logo">
</p>

<h1 align="center">Afterimage</h1>

<p align="center"><strong>Run BF16 language models that are larger than your GPU, without quantizing their weights.</strong></p>

<p align="center">
  <a href="https://github.com/oney-erge/Afterimage/actions/workflows/ci.yml"><img src="https://github.com/oney-erge/Afterimage/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="Python 3.10+"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#results">Results</a> ·
  <a href="docs/USAGE.md">Usage</a> ·
  <a href="docs/h65/README.md">H6.5 paper</a> ·
  <a href="docs/README.md">Documentation</a>
</p>

Afterimage stores model weights losslessly and streams them through limited GPU
memory. In the reference setup below, it runs Qwen3-14B's 29.5 GB BF16
checkpoint on an 8 GB GPU. The normal exact modes keep the original weights;
they do not quantize them or silently change the requested execution mode.

This is for research, evaluation, and offline generation when the original
checkpoint matters more than interactive speed. A streamed model remains
disk-bound, so expect seconds per token. If the model fits in GPU memory, or a
quantized runtime meets your needs, an in-memory or quantized engine will
usually be faster.

## Quick start

```bash
git clone https://github.com/oney-erge/Afterimage.git
cd Afterimage
./run.sh       # Linux; macOS: ./run.command
```

On Windows, double-click `run.bat`, or use `./run.sh` inside WSL2. The launcher
sets up its managed environment, starts the server after its health check, and
starts the web UI at **http://127.0.0.1:8420**. First run installs dependencies;
later runs reuse them. Open that address in a browser; press Ctrl-C to stop.

The small `afterimage quickstart` model is available separately and does not
automatically download a large checkpoint. Use `doctor`, `repair`, `docker`,
`logs`, or `stop` after the launcher name when needed. The full launcher,
Docker, and pip instructions are in [Usage](docs/USAGE.md).

## Run a model

Activate the launcher's environment in a second terminal with
`source .venv/bin/activate` (Linux/macOS/WSL2) or
`.venv\Scripts\Activate.ps1` (Windows PowerShell). Then compress a supported
checkpoint and set the GPU-memory budget explicitly:

```bash
afterimage compress Qwen/Qwen3-14B
afterimage run Qwen/Qwen3-14B \
  "Explain why the sky appears blue in two sentences." \
  --vram-budget-gb 4 --stats
```

`afterimage compress Qwen/Qwen3-14B --dry-run` estimates download size,
compressed-store size, and peak disk use before downloading. A real 14B model
can take substantial time to download and prepare; the reference run took about
30 minutes to download and 6 minutes to compress, but hardware and network
speed matter.

For a speculative-decoding configuration, add a small draft model:

```bash
afterimage run Qwen/Qwen3-14B \
  "Write a short Python function that checks whether a number is prime." \
  --vram-budget-gb 4 --draft-model Qwen/Qwen3-0.6B --spec-k 8 --stats
```

At temperature zero, this keeps the target model's greedy tokens. At nonzero
temperature, it samples from the target distribution; the draft model changes
efficiency, not that distribution. See [Usage](docs/USAGE.md) for the CLI,
Python API, server, model-store, and configuration reference.

## Results

Qwen3-14B (29.536 GB BF16), RTX 3080 Laptop GPU (8 GB), WSL2/CUDA, cold page
cache, four prompt families with four forced greedy tokens:

<p align="center">
  <img src="docs/assets/results-chart.svg" width="600" alt="Peak VRAM versus seconds per token for the six configurations below">
</p>

| Configuration | Peak VRAM | Seconds/token | vs AirLLM | Exactness |
|---|---:|---:|---:|---|
| **Afterimage + fixed speculation** | 3.813 GB | **9.150** | **2.93x** | Greedy-token exact at T=0 |
| Hugging Face Accelerate GPU/CPU/disk | 3.800 GB | 14.318 | 1.87x | Same BF16 checkpoint and token IDs |
| Afterimage exact + 4 GB residency | 3.934 GB | 17.360 | 1.55x | Reference-execution equivalent |
| **AirLLM 3.2.0** | **1.583 GB** | 26.828 | 1.00x | Same BF16 checkpoint and token IDs |
| Afterimage chunked output head | **0.901 GB** | 29.606 | 0.91x | Approximate BF16 matmul |
| Afterimage exact minimum-memory | 1.723 GB | 32.514 | 0.83x | Reference-execution equivalent |

The practical reading is simple: AirLLM is faster at the exact lowest-memory
point, Accelerate is faster near 4 GB without speculation, and this fixed
speculative configuration is fastest in this measured suite. These are
single-machine research results, not universal performance claims. The full
methods, raw JSON, controls, and cross-model results are linked in
[Results](docs/RESULTS_LOG.md) and the [evidence record](docs/ALL_HYPOTHESES_AND_BASELINES.md).

## Compatibility and limits

- Dense Qwen, Llama, and Mistral are the release-verified model families.
  Qwen packed-MoE and Qwen3-VL are implemented but experimental. See
  [supported architectures](docs/CONFIGURATION.md#supported-model-architectures).
- macOS is CPU-only. AMD/ROCm has not been verified on real AMD hardware, and
  WSL2 is the better-tested NVIDIA route than native Windows CUDA.
- The optional `--quantize q8` and `--lm-head-slice-rows` paths are explicitly
  approximate. They are not the default exact path.
- Exact low-memory generation is slow because most weights are read or
  reconstructed for each target pass.

## Research and documentation

Afterimage also contains an opt-in research workspace. Its current and
historical candidate results live in the [research record](docs/ALL_HYPOTHESES_AND_BASELINES.md);
they do not replace the stable runtime defaults.

The H6.5 work is a separate, schedule-aware placement method for large models
under limited memory, with confirmatory live evidence on Llama-3.3-70B. Start
with the [H6.5 paper guide](docs/h65/README.md),
which links its frozen measurements and reproduction material.

[Documentation home](docs/README.md) points to setup, usage, configuration,
FAQ, troubleshooting, architecture, and reproduction material. If you use the
software or H6.5, cite the entries in [CITATION.cff](CITATION.cff).

Apache-2.0. Built by [Oney Erge](https://oneyerge.com).
