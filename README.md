<p align="center">
  <img src="docs/assets/afterimage-logo.png" width="160" alt="Afterimage logo">
</p>

<h1 align="center">Afterimage</h1>

<p align="center"><strong>Run BF16 models too big for your GPU, at full precision.</strong></p>

<p align="center">
  <a href="https://github.com/oney-erge/Afterimage/actions/workflows/ci.yml"><img src="https://github.com/oney-erge/Afterimage/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="Python 3.10+"></a>
  <a href="docs/ALL_HYPOTHESES_AND_BASELINES.md"><img src="https://img.shields.io/badge/evidence-H0--H18-4fd1a5" alt="H0-H18 evidence"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="docs/USAGE.md">Usage</a> ·
  <a href="docs/FAQ.md">FAQ</a> ·
  <a href="docs/TROUBLESHOOTING.md">Troubleshooting</a> ·
  <a href="#results">Results</a>
</p>

Afterimage compresses your model's weights losslessly and streams them through
your GPU a layer at a time. A 29.5 GB model runs on an 8 GB card, bit-for-bit
identical to the original. No quantization, no accuracy loss.

Turn on speculative decoding with a small draft model and it's 3x faster than
streaming alone, and faster than Hugging Face's own Accelerate offload at the
same memory. Every number below is a real run on real hardware, not a
projection. The tradeoff: this is still disk-bound, so it runs seconds per
token, not tokens per second.

It comes with a CLI, a web UI, an OpenAI-compatible server, and a Python API.
There's also an opt-in research lab where nineteen speedup ideas get tested
against named controls and reported honestly, wins and losses both.

## Is this for you?

**Good fit:**
- The original BF16 checkpoint doesn't fit in your GPU's VRAM, and you want
  to run it anyway instead of switching to a smaller or quantized model.
- Keeping the original weights matters -- research, evaluation, or anywhere
  quantization's accuracy loss is unacceptable.
- Seconds-per-token latency is fine for your use case: batch jobs, offline
  generation, evaluation runs. Not interactive chat.

**Probably not a good fit:**
- The model already fits in your GPU's memory. A normal in-memory engine
  (vLLM, TensorRT-LLM, plain Transformers) will just be faster.
- A quantized runtime already does what you need. Quantization trades some
  accuracy for speed and memory that Afterimage deliberately doesn't take;
  if you don't need the original weights, a quantized engine is usually the
  better choice.
- You need chat-speed responses. Streamed generation runs several seconds
  per token, not tokens per second.

**Worth knowing before you install:**
- Only Llama-family checkpoint layouts work: a top-level `model.layers` list
  of decoder blocks, `model.embed_tokens`, and `model.lm_head`. Qwen, Llama,
  and Mistral are the tested family; GPT-2/Falcon/MPT/BLOOM-style layouts do
  not work. See [CONFIGURATION.md](docs/CONFIGURATION.md#supported-model-architectures).
- macOS runs CPU-only today (no CUDA, so the GPU decode kernels don't run).
- AMD/ROCm is implemented but hasn't been run on real AMD hardware by this
  project -- treat it as untested, not verified.
- Native Windows CUDA is less tested here than the WSL2 path; WSL2 +
  `install.sh` is the better-verified route on Windows with an NVIDIA GPU.

## Quick start

```bash
git clone https://github.com/oney-erge/Afterimage.git
cd Afterimage
./run.sh       # Linux
./run.command  # macOS
```

On Windows, double-click `run.bat`. First run installs a pinned `uv`, a
managed Python, and the matching Torch build, then opens the server after its
health check succeeds. Later runs reuse the environment. Use `doctor`,
`repair`, `docker`, `logs`, or `stop` after the launcher name for the same
operational interface used by the other projects. The small model quickstart
is available separately as `afterimage quickstart` and is not an automatic
multi-gigabyte download.

Setup checks disk space before model tooling is installed, prevents concurrent
environment changes, retries temporary network failures up to three times, and
records failures in `.setup/install.log`.

Use `.\run.ps1` from PowerShell.

The script finishes by starting the server and stays running in that
terminal -- open **http://127.0.0.1:8420** in a browser for the web UI.
Press Ctrl-C to stop it; run the launcher again any time.

To use the CLI from another terminal while the server (or without it) is
running, use the virtual environment `./start` created:

```bash
.venv/bin/afterimage compress Qwen/Qwen3-14B      # macOS/Linux/WSL2
.venv\Scripts\afterimage.exe compress Qwen/Qwen3-14B   # Windows
```

or `source .venv/bin/activate` (macOS/Linux/WSL2) / `.venv\Scripts\Activate.ps1`
(Windows) first, then just `afterimage ...`.

### What to expect before you download a large model

The quickstart above proves the pipeline works on your machine; its
seconds-per-token number is for a 0.6B model and isn't a predictor of how a
14B model will perform, since it varies with your disk, CPU, and GPU.

For the reference benchmark below (Qwen3-14B, RTX 3080 Laptop GPU, 8 GB
VRAM), the measured operating points are:

- **9.150 s/token** with fixed speculative decoding, at 3.813 GB VRAM.
- **17.360 s/token** with exact residency and no speculation, at 3.934 GB VRAM.
- **32.514 s/token** at the exact minimum-memory floor, at 1.723 GB VRAM.

Downloading and compressing a 14B model was about 30 minutes of download and
6 minutes of compression on a 16-core reference machine; both vary a lot with
your connection speed and CPU.

```bash
afterimage compress Qwen/Qwen3-14B --dry-run
```

`--dry-run` estimates download size, compressed-store size, and peak disk
usage before you commit to the download -- it does not check whether the
model's architecture is supported. That check happens when a store is
actually built or run.

### Run a real model

```bash
afterimage compress Qwen/Qwen3-14B
afterimage run Qwen/Qwen3-14B \
  "Explain why the sky appears blue in two sentences." \
  --vram-budget-gb 4 --stats
```

Already have a compressed store published somewhere? `afterimage pull
Qwen/Qwen3-14B --store-repo someone/that-store` fetches it directly and
skips compression entirely -- see [USAGE.md](docs/USAGE.md).

Prefer Docker?

```bash
docker compose up
```

Docker GPU execution requires the NVIDIA Container Toolkit.

### Add speculative decoding

```bash
afterimage run Qwen/Qwen3-14B \
  "Write a short Python function that checks whether a number is prime." \
  --vram-budget-gb 4 \
  --draft-model Qwen/Qwen3-0.6B \
  --spec-k 8 --stats
```

At temperature zero, speculative decoding commits the same greedy tokens as the
target. At nonzero temperature it samples from the target distribution; the
draft changes efficiency, not the target distribution.

## Why Afterimage

- **Run the original checkpoint.** The measured 29.5 GB BF16 model runs on an
  8 GB GPU without weight quantization.
- **Choose the tradeoff explicitly.** Set a VRAM and optional host-RAM budget;
  infeasible exact plans fail before generation instead of silently degrading.
- **Store and stream fewer bytes.** The Qwen3-14B store is 20.328 GB, a 1.453x
  lossless reduction from 29.536 GB.
- **Use spare memory productively.** Residency eliminates repeated reads;
  speculation amortizes one streamed target pass over several committed tokens.
- **Keep contracts visible.** Exact, greedy-token-exact, distribution-exact,
  and approximate modes are labeled in configuration and benchmark output.
- **Use it as a service.** The package exposes one-shot generation, job control,
  an OpenAI-compatible endpoint, a web UI, and machine-readable experiment runs.

## Results

Qwen3-14B (29.536 GB BF16), RTX 3080 Laptop GPU (8 GB), WSL2/CUDA, cold page
cache, four prompt families × four forced greedy tokens:

| Configuration | Peak VRAM | Seconds/token | vs AirLLM | Exactness |
|---|---:|---:|---:|---|
| **Afterimage + fixed speculation** | 3.813 GB | **9.150** | **3.15x** | Greedy-token exact at T=0 |
| Hugging Face Accelerate GPU/CPU/disk | 3.800 GB | 14.318 | 2.02x | Same BF16 checkpoint and token IDs |
| Afterimage exact + 4 GB residency | 3.934 GB | 17.360 | 1.66x | Reference-execution equivalent |
| **AirLLM 3.1.0** | **1.583 GB** | 28.861 | 1.00x | Same BF16 checkpoint and token IDs |
| Afterimage chunked output head | **0.901 GB** | 29.606 | 0.97x | **Approximate** BF16 matmul |
| Afterimage exact minimum-memory | 1.723 GB | 32.514 | 0.89x | Reference-execution equivalent |

The honest reading:

- At the exact low-memory floor, **AirLLM wins**.
- At about 4 GB without speculation, **Hugging Face Accelerate wins**.
- With fixed speculation, **Afterimage wins this suite**: 1.56x Accelerate and
  3.15x AirLLM at 3.813 GB.
- The 0.901 GB Afterimage point is not lossless execution. Blocking the output
  head changes BF16 reduction order even when the final token happens to agree.

See [all H0-H18 hypotheses, controls, results, external comparisons, and the
ranked conclusion](docs/ALL_HYPOTHESES_AND_BASELINES.md). The raw JSON is in
[`results/`](results/).

The broader [cross-family and scale campaign](docs/CROSS_MODEL_BENCHMARK_2026-08-22.md)
adds Phi-4 Mini 3.8B and Mistral Small 24B. It finds a real Pareto boundary:
Accelerate is fastest on both new checkpoints; AirLLM owns the exact low-VRAM
24B point; and Afterimage certified MIPS reaches 27.539 s/token at 2.915 GB,
**1.66x AirLLM** but 9.52% behind Accelerate. Across Phi, Qwen, and Mistral,
the lossless store remains stable at 1.45–1.49x compression.

## Limits

- Benchmarks are single-machine research evidence, not universal performance
  claims. Effects below about 10% should be treated cautiously without paired
  confirmation.
- Exact low-VRAM generation is slow: a streamed 14B target reads or reconstructs
  most weights for every target pass.
- WSL2 limits available pinned system memory. H9's full 14B test needs native
  Linux or another host that can genuinely pin at least 1.6 GB.
- When the full model fits in GPU memory, use an in-memory engine such as vLLM,
  TensorRT-LLM, or Transformers instead.
- macOS runs CPU-only today; there's no CUDA, so the GPU decode kernels don't
  run. A streamed 14B model on CPU is slow enough to be a demo, not a daily
  tool. Apple Silicon's unified memory often means the model you want fits
  directly anyway, which is the case Afterimage doesn't need to solve.
- AMD/ROCm is untested on real hardware by this project, and native Windows
  CUDA is less validated than the WSL2 path.
- Checkpoint layout is a real, permanent limitation, not a bug: only the
  Llama-family structure (`model.layers`, `model.embed_tokens`, `model.lm_head`)
  is supported. See [CONFIGURATION.md](docs/CONFIGURATION.md#supported-model-architectures).
- Two opt-in paths are not lossless, and are labeled as such wherever they
  appear: `afterimage compress --quantize q8` trades bit-exactness for a
  smaller store, and `--lm-head-slice-rows` trades bit-exactness for a lower
  VRAM floor by changing matmul reduction order. Neither is the default.

## Python API

```python
from pathlib import Path

from transformers import AutoTokenizer

from afterimage.runtime.config import EngineConfig
from afterimage.runtime.streaming_engine import (
    StreamingLosslessModel,
    load_draft_model,
)

config = EngineConfig(
    vram_budget_gb=4.0,
    ram_budget_gb=8.0,
    draft_mode="model",
    spec_k=8,
)

model = StreamingLosslessModel(
    "Qwen/Qwen3-14B",
    store_dir=Path.home() / ".afterimage/stores/Qwen__Qwen3-14B",
    device="cuda",
    config=config,
)
draft = load_draft_model("Qwen/Qwen3-0.6B", device="cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
input_ids = tokenizer("Explain entropy in one paragraph.", return_tensors="pt").input_ids.cuda()
tokens, policy = model.generate_adaptive(
    input_ids,
    max_new_tokens=64,
    draft_model=draft,
)
```

## Server

```bash
pip install -e ".[server]"
afterimage serve --host 127.0.0.1 --port 8420
```

The server exposes `/v1/chat/completions`, compression job controls,
pause/resume/cancel, budget feasibility, runtime statistics, and the H0-H18
Experiment Lab.

## Research lab

Research methods are opt-in configurations and never replace the stable runtime
defaults automatically.

```bash
afterimage research experiments --json
afterimage research test-plan h9-ram-overlay-head --json
afterimage research pin-preflight --gigabytes 0.35 --json
afterimage research profile-trace TRACE.json --out PROFILE.json
afterimage research optimize-residency TRACE.json \
  --manifest STORE/manifest.json --vram-budget-gb 4 --out PLAN.json
```

Current research summary:

- H9 is the strongest positive mechanism screen, but only at 0.6B on this WSL2
  host because the 14B head exceeds its pinned-memory ceiling.
- H1 is the strongest positive live 14B candidate, at +1.61% versus control.
- H6 predicts a 38.56% preparation reduction and now has a scalable 441-tensor
  planner, but still needs held-out live execution.
- H16 and H17 regressed. H18's exact KV rollback passed its mechanism gate but
  stopped for L2 futility (-0.59% paired median; 90% interval -4.62% to +1.09%).
- The remaining candidates are below gate, action-identical, or contradicted.

No H1-H18 candidate has L3 confirmatory superiority evidence. Fixed speculation
is a stable core configuration, not one of the failed adaptive candidates, and
is the web UI's default profile.

## Documentation

| Document | Purpose |
|---|---|
| [All hypotheses and baselines](docs/ALL_HYPOTHESES_AND_BASELINES.md) | **Controlling results table, rankings, AirLLM/Accelerate comparisons, novelty assessment** |
| [Usage](docs/USAGE.md) | Install-to-serving walkthrough: CLI, server, Python API |
| [FAQ](docs/FAQ.md) | Short answers to common first-time questions |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common failures and fixes |
| [Architecture](docs/ARCHITECTURE.md) | Runtime, storage, memory-tier, speculation, and evidence diagrams |
| [Configuration](docs/CONFIGURATION.md) | Stable profiles and advanced flags |
| [How it works](docs/HOW_IT_WORKS.md) | Implementation walkthrough and AirLLM contrast |
| [Research methods](docs/RESEARCH_METHODS.md) | H0-H18 definitions, evidence levels, controls, and kill gates |
| [Hypothesis lineage](docs/HYPOTHESIS_LINEAGE.md) | Literature source and novelty boundary for each idea |
| [Literature](docs/LITERATURE.md) | Survey of running models larger than VRAM, and where this sits in it |
| [Cross-model benchmark](docs/CROSS_MODEL_BENCHMARK_2026-08-22.md) | Phi-4 Mini, Qwen3, and Mistral Small across families and scale |
| [Results log](docs/RESULTS_LOG.md) | Chronological corrections and raw-run interpretation |
| [Contributing](CONTRIBUTING.md) | Development and verification workflow |

Apache-2.0. Contributions and reproducible counter-results are welcome.
