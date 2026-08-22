# Usage

## Install

```bash
./install.sh          # Linux / WSL2 -- detects GPU, sets up a venv, runs the quickstart
```
```powershell
.\install.ps1          # Windows (native CUDA)
```

Both scripts detect your GPU vendor, install the matching torch build,
editable-install the package, run `afterimage doctor`, then run
`afterimage quickstart` before launching the server — so you find out in a
few minutes whether the install actually works, rather than after an hour
and ~50 GB spent on a 14B-class model.

By hand:
```bash
pip install -e ".[gpu,server]"   # NVIDIA; drop "gpu" for CPU-only or AMD
afterimage doctor
```

## Quickstart (a few minutes, ~2 GB)

```bash
afterimage quickstart
```

Compresses and runs Qwen3-0.6B end to end and prints the measured s/token,
peak VRAM, and I/O/decode split on your actual hardware. It is not a
benchmark of the engine's real capability — the point is to prove the
pipeline works here, before committing to a real model.

## A real model

```bash
afterimage compress Qwen/Qwen3-14B --dry-run   # estimate download/store size and disk headroom first
afterimage compress Qwen/Qwen3-14B             # ~30 min download, ~6 min compress on 16 cores
afterimage run Qwen/Qwen3-14B "The capital of France is" --auto
```

`--auto` detects your VRAM and picks `--profile min-memory|balanced|fast`
for you, printing what it chose and why. To pick explicitly:

```bash
afterimage run Qwen/Qwen3-14B "..." --profile fast     # 3.15x, needs a draft model + ~4 GB VRAM
afterimage run Qwen/Qwen3-14B "..." --profile balanced  # 1.66x, ~4 GB VRAM, no speculation
afterimage run Qwen/Qwen3-14B "..." --profile min-memory  # lowest VRAM, slowest, fully exact
```

Or set the underlying knobs directly — see
[CONFIGURATION.md](CONFIGURATION.md):

```bash
afterimage run Qwen/Qwen3-14B "..." \
  --vram-budget-gb 4 \
  --draft-model Qwen/Qwen3-0.6B --spec-k 8 \
  --stats
```

`--stats` prints peak VRAM, I/O/decode/compute time, and prefetch counters
after generation.

## Serving

```bash
afterimage serve                    # OpenAI-compatible API + web UI on :8420
```

```bash
docker compose up                   # needs the NVIDIA Container Toolkit
```

The web UI's "Try it" panel has the same dial as the CLI: a profile picker,
a VRAM slider with live feasibility checking against `/api/plan`, a draft
model field, and a live seconds-per-token readout once generation starts.

The API is OpenAI-compatible (`/v1/chat/completions`, `/v1/models`) plus
Afterimage's own fields for the dial — see
[CONFIGURATION.md](CONFIGURATION.md#server-side-api-fields). Operational
endpoints: `GET /health`, `GET /api/version`, `GET /api/stats`.

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8420/v1", api_key="unused")
resp = client.chat.completions.create(
    model="Qwen/Qwen3-14B",
    messages=[{"role": "user", "content": "The capital of France is"}],
    max_tokens=32,
    extra_body={"vram_budget_gb": 4, "draft_model": "Qwen/Qwen3-0.6B"},
)
print(resp.choices[0].message.content)
print(resp.usage.model_extra["afterimage"])  # seconds_per_token, peak_vram_gb, ...
```

## Programmatic use

```python
from afterimage.runtime.config import EngineConfig
from afterimage.runtime.streaming_engine import StreamingLosslessModel, load_draft_model

cfg = EngineConfig(vram_budget_gb=4.0, draft_mode="model", spec_k=8)
sm = StreamingLosslessModel("Qwen/Qwen3-14B", store_dir, device="cuda", config=cfg)
draft = load_draft_model("Qwen/Qwen3-0.6B", device="cuda")
seq, _policy = sm.generate_adaptive(input_ids, max_new_tokens=64, draft_model=draft,
                                    temperature=0.0)  # T=0: provably token-identical to greedy
```

## The research layer

Everything under `afterimage research …` (`experiments`, `test-plan`,
`pin-preflight`, `profile-trace`, `optimize-residency`) is opt-in and does
not affect `compress`/`run`/`serve`. See
[RESEARCH_METHODS.md](RESEARCH_METHODS.md) and
[HYPOTHESIS_LINEAGE.md](HYPOTHESIS_LINEAGE.md).

## Next steps

- [CONFIGURATION.md](CONFIGURATION.md) — every knob, what it costs, supported architectures
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — common failures and fixes
- [HOW_IT_WORKS.md](HOW_IT_WORKS.md) — the mechanism, method by method, next to AirLLM
