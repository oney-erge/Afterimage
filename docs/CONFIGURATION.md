# Configuration

`EngineConfig` has about 40 fields, but only six of them matter for an
ordinary run. The rest belong to the opt-in H0-H18 research layer (see
[RESEARCH_METHODS.md](RESEARCH_METHODS.md)), default to the engine's
original behaviour, and never need touching unless you're running an
experiment. This page covers the six, plus supported models and the
CLI-to-API mapping.

## The six knobs that matter

| Knob | CLI flag | API field | Default | What it does |
|---|---|---|---|---|
| VRAM budget | `--vram-budget-gb` | `vram_budget_gb` | none (minimum-memory) | Spends spare VRAM on residency. Refused up front if infeasible, never silently approximated. Measured: 1.66x at 4 GB on a 14B model with an 8 GB card. |
| RAM budget | `--ram-budget-gb` | `ram_budget_gb` | none | A second, pinned-host-RAM tier below VRAM and above disk. Needs a VRAM budget set first (the planner fills VRAM before RAM). |
| Draft model | `--draft-model` | `draft_model` | none (no speculation) | A small resident model (e.g. `Qwen/Qwen3-0.6B`) that enables speculative decoding, the largest lossless speedup measured (3.15x). Must share the target's tokenizer and vocabulary. |
| Draft chain length | `--spec-k` | `spec_k` | 8 | How many tokens the draft model proposes per sweep, when a draft model is set. |
| Chunked output head | `--lm-head-slice-rows` | `lm_head_slice_rows` | 0 (whole head, exact) | Above 0, computes logits in row blocks instead of materializing the whole 1.5+ GB output head. Lowers the VRAM floor by about 43%, but is **not bit-exact** (see [HOW_IT_WORKS.md's Method 3](HOW_IT_WORKS.md#method-3--chunked-output-head-approximate)). |
| Quantization | `--quantize` (compress only) | n/a | none (lossless) | `q8` trades bit-exactness for a smaller store. Set at compression time, not per-run. |

## Profiles: the measured operating points, as presets

`--profile {min-memory,balanced,fast}` (CLI) applies one of the README's
benchmark rows directly:

| Profile | vram_budget_gb | draft_model | Measured |
|---|---|---|---|
| `min-memory` | none | none | exact, lowest VRAM, slowest (0.89x AirLLM on the reference hardware) |
| `balanced` | 4.0 | none | exact, 1.66x |
| `fast` | 4.0 | `Qwen/Qwen3-0.6B` | exact at T=0, 3.15x, the largest lossless win |

An explicit flag always overrides the profile's value for that field.
`--auto` picks a profile from detected VRAM instead of asking you to choose,
and prints its reasoning before running. There's no server-side equivalent
of `--auto` yet. Pass `vram_budget_gb`/`draft_model` explicitly in API
requests, or check `/api/plan` for feasibility first.

## Everything else: the research layer

The other roughly 34 fields (placement policy, prefetch policy, storage
read policy, representation policy, expert codec, critical-path profiles,
replay plans, tracing) are the H0-H18 research mechanisms. None of them
has passed its gate yet, and none of them changes behaviour unless you set
it explicitly. See [RESEARCH_METHODS.md](RESEARCH_METHODS.md) and
[HYPOTHESIS_LINEAGE.md](HYPOTHESIS_LINEAGE.md) for what each one claims and
what actually happened when it was tested. `afterimage run --help` shows
them under "advanced / research"; `afterimage research --help` lists the
dedicated subcommands for building profiles, traces and plans.

## Supported model architectures

Model discovery, storage format, execution compatibility, and expected
performance are separate properties. Model size never disables Get. The web
pipeline downloads a snapshot first, inspects its config on the meta device,
and only marks the model Ready after an executable adapter and store checksum
verification both succeed.

`ModelAdapter` currently resolves the ordinary causal-language layout and the
Qwen3-VL `model.language_model` layout. This removes decoder traversal from
hard-coded engine paths while keeping the evidence labels conservative:

| Family | Execution status |
|---|---|
| Qwen2/Qwen3 dense, Llama, Mistral | **Verified family.** These are the release-evidenced dense layouts. |
| Gemma-family text, Phi-3, OLMo, StableLM, Cohere, Persimmon | **Expected.** Their decoder structure matches the causal adapter, but this project has not release-verified every checkpoint. |
| Qwen3-MoE packed experts | **Experimental.** Preparation splits packed expert tensors losslessly, and runtime routing loads only selected experts. CPU unit tests cover slice reconstruction and selected-expert numerical equivalence; a real large checkpoint has not passed the GPU release suite yet. |
| Qwen3-VL dense and Qwen3-VL-MoE | **Experimental.** AutoProcessor image inputs, the resident vision front end, the adapted language decoder, and multimodal chat are wired. A real Qwen3-VL checkpoint has not passed the end-to-end GPU release suite yet. |
| Other layouts | **Download only until adapted.** They remain discoverable and downloadable. Afterimage reports that execution is not yet verified instead of conflating architecture with model size. |

The CLI `compress --dry-run` remains a size estimator. The web Get lifecycle is
the compatibility-aware path: Remote, Downloading, Downloaded, Preparing,
Verifying, then Ready.

**Apple Silicon:** no CUDA means no GPU decode kernels, so Afterimage runs
CPU-only on a Mac today. A streamed 14B model on CPU is slow enough to be a
demo rather than a daily tool. Worth knowing too: unified memory often means
the model you want already fits without streaming at all, in which case a
normal (non-streaming) runtime will simply be faster. Afterimage is for the
case where the model doesn't fit, which unified memory frequently avoids.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AFTERIMAGE_STORE_ROOT` | `~/.afterimage/stores` | Where compressed stores live. The Docker image sets this to `/data/stores` (see `Dockerfile`). |
| `HF_HOME` | HuggingFace's default | Model download cache. |

## Server-side API fields

`POST /v1/chat/completions` accepts the standard OpenAI fields
(`model`, `messages`, `max_tokens`, `temperature`, `stream`) plus
Afterimage's own: `vram_cap_gb`, `vram_budget_gb`, `ram_budget_gb`,
`draft_model`, `spec_k`, `lm_head_slice_rows`. The response's `usage`
object carries an `afterimage` block with `seconds_per_token`,
`peak_vram_gb`, `bytes_read_gb`, `prefetch_hit_rate`, and, when speculation
is active, `spec_sweeps`/`spec_accepted_tokens`. `GET /api/stats` returns
the same block for the most recently completed generation on the currently
loaded model.

See also: [USAGE.md](USAGE.md) for the install-to-serving walkthrough,
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common failures.
