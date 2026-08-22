# Configuration

`EngineConfig` has ~40 fields. Six of them are what an ordinary run needs;
the rest belong to the opt-in H0-H15 research layer (see
[RESEARCH_METHODS.md](RESEARCH_METHODS.md)) and default to this engine's
original, pre-research behaviour. This page covers the six, plus the
supported-model and CLI-vs-API mapping.

## The six knobs that matter

| Knob | CLI flag | API field | Default | What it does |
|---|---|---|---|---|
| VRAM budget | `--vram-budget-gb` | `vram_budget_gb` | none (minimum-memory) | Spends spare VRAM on residency. Refused up front if infeasible, never silently approximated. Measured: 1.66x at 4 GB on a 14B model with an 8 GB card. |
| RAM budget | `--ram-budget-gb` | `ram_budget_gb` | none | A second, pinned-host-RAM tier below VRAM and above disk. Needs a VRAM budget set first (the planner fills VRAM before RAM). |
| Draft model | `--draft-model` | `draft_model` | none (no speculation) | A small resident model (e.g. `Qwen/Qwen3-0.6B`) that enables speculative decoding — the largest lossless speedup measured (3.15x). Must share the target's tokenizer/vocabulary. |
| Draft chain length | `--spec-k` | `spec_k` | 8 | How many tokens the draft model proposes per sweep, when a draft model is set. |
| Chunked output head | `--lm-head-slice-rows` | `lm_head_slice_rows` | 0 (whole head, exact) | >0 computes logits in row blocks instead of materializing the whole 1.5+ GB output head. Lowers the VRAM floor by ~43%, but is **not bit-exact** — see the README's "why the chunked head is not exact" section. |
| Quantization | `--quantize` (compress only) | — | none (lossless) | `q8` trades bit-exactness for a smaller store. Set at compression time, not per-run. |

## Profiles — the measured operating points, as presets

`--profile {min-memory,balanced,fast}` (CLI) applies one of the README's
benchmark rows directly:

| Profile | vram_budget_gb | draft_model | Measured |
|---|---|---|---|
| `min-memory` | none | none | exact, lowest VRAM, slowest (0.89x AirLLM on the reference hardware) |
| `balanced` | 4.0 | none | exact, 1.66x |
| `fast` | 4.0 | `Qwen/Qwen3-0.6B` | exact at T=0, 3.15x — the largest lossless win |

An explicit flag always overrides the profile's value for that field.
`--auto` picks a profile from detected VRAM instead of asking you to choose;
it prints its reasoning before running. There is no server-side equivalent
of `--auto` yet — pass `vram_budget_gb`/`draft_model` explicitly in API
requests, or use `/api/plan` to check feasibility first.

## Everything else: the research layer

The other ~34 fields — placement policy, prefetch policy, storage read
policy, representation policy, expert codec, critical-path profiles, replay
plans, tracing — are the H0-H15 research mechanisms. None of them has
passed its gate yet (see [RESEARCH_METHODS.md](RESEARCH_METHODS.md) and
[HYPOTHESIS_LINEAGE.md](HYPOTHESIS_LINEAGE.md)), and none of them changes
behaviour unless explicitly set. `afterimage run --help` shows them under
"advanced / research"; `afterimage research --help` lists the dedicated
subcommands for building profiles, traces and plans.

## Supported model architectures

This engine hard-codes a Llama-family layout: a top-level `model.layers`
list of decoder blocks, `model.embed_tokens`, and `model.lm_head`.

| Family | Status |
|---|---|
| Qwen (Qwen2, Qwen3) | works — the primary tested family |
| Llama, Llama 2/3 | works |
| Mistral | works (same layout family) |
| GPT-2, Falcon, MPT and other non-Llama layouts | **not supported** — construction fails with a clear error naming the architecture, rather than a silent wrong result |

If you're not sure, `afterimage compress MODEL --dry-run` and
`afterimage run` both fail fast and name the problem before doing real work.

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
`peak_vram_gb`, `bytes_read_gb`, `prefetch_hit_rate`, and — when
speculation is active — `spec_sweeps`/`spec_accepted_tokens`. `GET
/api/stats` returns the same block for the most recently completed
generation on the currently loaded model.

See also: [USAGE.md](USAGE.md) for the install-to-serving walkthrough,
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common failures.
