# Usage

## Get started

```bash
git clone https://github.com/oney-erge/Afterimage.git
cd Afterimage
./run.sh
```

That's it. First run, it detects your GPU, installs the matching Torch build,
creates a `.venv`, and starts the server. It does not download a model without
your request. Every run after that reuses the current environment. The
launcher finishes by starting the server and stays running in that
terminal -- open **http://127.0.0.1:8420** in a browser for the web UI.
Run `./run.sh` again any time.

On Windows, double-click `run.bat`. On macOS, double-click
`run.command`. Use `run.ps1` from PowerShell. Every root launcher also accepts
`doctor`, `repair`, `docker`, `logs`, and `stop`. The same script runs either
way, and it always tells you whether it just installed something or is
just starting.

When you are ready for the separate small-model check, run `afterimage
quickstart`. It downloads Qwen3-0.6B and measures the pipeline on your machine.

To use the CLI from another terminal instead of (or alongside) the
server, use the virtual environment the launcher created directly
(`.venv/bin/afterimage ...` on macOS/Linux/WSL2,
`.venv\Scripts\afterimage.exe ...` on Windows), or activate it first
(`source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1`) and just
run `afterimage ...`.

By hand, if you'd rather:
```bash
pip install -e ".[gpu,server]"   # NVIDIA; drop "gpu" for CPU-only or AMD
afterimage doctor
afterimage quickstart
```

## Quickstart (a few minutes, ~2 GB)

```bash
afterimage quickstart
```

Compresses and runs Qwen3-0.6B end to end and prints the measured
seconds-per-token, peak VRAM, and I/O/decode split on your actual
hardware. It's not a benchmark of what the engine can really do, just
proof the pipeline works here before you point it at something big.

## A real model

```bash
afterimage compress Qwen/Qwen3-14B --dry-run   # estimate download/store/disk size first (does not check architecture)
afterimage compress Qwen/Qwen3-14B             # reference: ~30 min download, ~6 min compress on 16 cores; varies a lot by connection and CPU
afterimage run Qwen/Qwen3-14B "The capital of France is" --auto
```

If someone has already published a compressed store for the model you
want, skip the download-and-compress step entirely:

```bash
afterimage pull Qwen/Qwen3-14B --store-repo someone/afterimage-qwen3-14b
afterimage run Qwen/Qwen3-14B "The capital of France is" --auto
```

This project doesn't host any stores itself yet, so `--store-repo` has to
point at a real one -- a compressed store is just `manifest.json` +
`weights.bin` (see `compress_model_to_disk`), so nothing stops anyone
publishing one. `pull` checksums the download against the manifest by
default (`--no-verify` to skip); `afterimage verify MODEL` re-runs that
check any time, against a pulled or locally-compressed store alike.

Output streams token by token as it's generated (a 14B model at these
profiles measures 9.150-32.514 s/token depending on the profile, so it
matters), applies the model's chat template, and stops at the model's own
end-of-turn token instead of running to `--max-new-tokens` regardless.
`--raw` skips the template for a base/completion model; `--think` allows
the model's native reasoning trace instead of suppressing it (real
generated tokens, at full per-token price); `--no-stream` prints the whole
answer at once instead. `afterimage doctor` also benchmarks your disk's
real read speed and estimates the minimum-memory profile's s/token from
it, so the README's reference-hardware numbers aren't the only data point
you have for your own disk.

`--auto` detects your VRAM and picks `--profile min-memory|balanced|fast`
for you, printing what it chose and why. To pick explicitly:

```bash
afterimage run Qwen/Qwen3-14B "..." --profile fast     # 3.15x, needs a draft model + ~4 GB VRAM
afterimage run Qwen/Qwen3-14B "..." --profile balanced  # 1.66x, ~4 GB VRAM, no speculation
afterimage run Qwen/Qwen3-14B "..." --profile min-memory  # lowest VRAM, slowest, fully exact
```

Or set the underlying knobs directly. See [CONFIGURATION.md](CONFIGURATION.md):

```bash
afterimage run Qwen/Qwen3-14B "..." \
  --vram-budget-gb 4 --max-context 8192 \
  --draft-model Qwen/Qwen3-0.6B --spec-k 8 \
  --stats
```

`--max-context` reserves VRAM for that many tokens of KV cache up front,
so a long conversation gets refused before it starts rather than running
out of memory partway through one -- worth setting whenever you're doing
more than a single short exchange.

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
Afterimage's own fields for the dial, documented in
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

- [CONFIGURATION.md](CONFIGURATION.md): every knob, what it costs, supported architectures
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): common failures and fixes
- [HOW_IT_WORKS.md](HOW_IT_WORKS.md): the mechanism, method by method, next to AirLLM
