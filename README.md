# Afterimage

**Run models larger than your GPU's VRAM, bit-exact — with a memory/speed
dial AirLLM doesn't have.**

Afterimage entropy-codes bf16 model weights (lossless, ~1.45x — the
information-theoretic ceiling for bf16 is ~1.51x) and streams them
layer-by-layer from a compressed on-disk store, the same algorithm AirLLM
uses. It adds three things on top: spending spare VRAM on residency,
speculative decoding, and an optional non-lossless mode that shrinks the
VRAM floor by 45%.

> Status: a working engine, measured against AirLLM on real hardware,
> including where it does **not** win. See
> [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for the full picture — this
> README is the short version.

---

## Measured, not projected

Qwen3-14B (29.5 GB bf16), RTX 3080 Laptop (8 GB VRAM), WSL2/CUDA, cold page
cache, identical prompt, **both systems on the same peak-VRAM counter**:

| Configuration | VRAM | s/token | Lossless | vs AirLLM |
|---|---|---|---|---|
| **AirLLM** (reference) | 1.57 GB | 27.4 | yes | 1.00x |
| Afterimage, compression only | 1.68 GB | 28.0 | yes | 0.98x — **parity** |
| Afterimage, + residency | 3.89 GB | 15.2 | yes | 1.80x |
| Afterimage, + speculative decoding | 3.78 GB | **2.2** | yes | **12.5x** |
| Afterimage, chunked head (opt-in, lossy) | **0.86 GB** | 26.4 | no | −45% VRAM, parity speed |
| Afterimage, chunked head + speculation (lossy) | 2.05 GB | 3.3 | no | 8.3x |

**The honest headline: at matched VRAM, lossless, we are not faster than
AirLLM — we're at parity.** Our real advantage is a dial AirLLM doesn't
have: spend more VRAM for up to 12.5x, or use 45% *less* VRAM at the same
speed if bit-exactness can be relaxed. Full derivation, method-by-method,
and the run-to-run noise band: **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)**.

---

## How speculative decoding gets 12.5x, in plain terms

Normally, producing **one word** means reading the **entire 20 GB model**
off disk. That's the whole cost — one trip to the library for one word.

Speculative decoding: a small, fast model (Qwen3-0.6B, resident in 1.3 GB
of VRAM) guesses the next several words cheaply. The big model then reads
itself **once** and checks *all* of those guesses in that single pass —
same one trip, but now it can confirm many words instead of one.

The check is exact: each guess is accepted with probability
`min(1, target_prob / draft_prob)`; the moment one is rejected, the correct
word is sampled from what's left over. This is provably the same output
distribution the big model would have produced alone — **the small model's
guesses can only change speed, never correctness.** A bad guess costs one
wasted word; it can never produce a wrong one.

That's the whole mechanism. No new compression, no approximation of the
big model's math — just fewer full-model reads per word actually spoken.

---

## Install & run

```bash
./install.sh          # Linux / WSL2 — detects GPU, sets up venv or launches server
```
```powershell
.\install.ps1          # Windows (native CUDA)
```

Or by hand:
```bash
pip install -e ".[gpu,server]"
afterimage doctor
afterimage compress Qwen/Qwen3-14B         # one-time, ~6 min on 16 cores
afterimage run Qwen/Qwen3-14B "The capital of France is"
afterimage serve                            # FastAPI + web UI on :8420
```

```bash
docker compose up      # needs the NVIDIA Container Toolkit
```

`afterimage serve` exposes an OpenAI-compatible `/v1/chat/completions`,
plus native job control (`/api/compress`, `/api/jobs/{id}`, pause/resume/
cancel, `/api/plan` for budget feasibility) and a minimal web UI at `/`.

---

## Controlling residency and speed

```python
from afterimage.runtime.config import EngineConfig
from afterimage.runtime.streaming_engine import StreamingLosslessModel, load_draft_model

cfg = EngineConfig(
    vram_budget_gb=2.0,          # the dial. Refused up front if infeasible, not approximated.
    ram_budget_gb=8.0,           # pinned host RAM as a second, faster-than-disk tier
    draft_mode="model",          # "none" | "model" | "self" -- speculative decoding
    spec_k=8,                    # draft chain length
    lm_head_slice_rows=0,        # >0 shrinks the VRAM floor ~1.4GB -- NOT lossless, see below
)
sm = StreamingLosslessModel("Qwen/Qwen3-14B", store_dir, device="cuda", config=cfg)
draft = load_draft_model("Qwen/Qwen3-0.6B", device="cuda")
seq, _policy = sm.generate_adaptive(input_ids, max_new_tokens=64, draft_model=draft)
```

Every weight is used once per token under plain streaming, so "importance"
can't rank residency — what differs is **bus traffic avoided per byte of
VRAM spent** (`compressed_bytes / uncompressed_bytes`). A poorly-compressing
tensor costs the most in re-streaming, so it's the best candidate to keep
resident. `vram_planner.py` fills VRAM, then RAM, greedily by this ranking.

`EngineConfig.is_lossless` is `False` whenever `quantize="q8"` or
`lm_head_slice_rows > 0` is set, and `describe()` says so explicitly. No
run using either may be reported as bit-exact.

---

## What's built

Lossless bf16 compression at 96% of the Shannon ceiling; GPU Huffman decode
via a Triton kernel; three-tier VRAM/RAM/disk residency planning; row-gathered
embeddings; a KV cache verified bit-exact; speculative decoding (small draft
model and self-drafting, both via `generate_adaptive`); an optional chunked
`lm_head` that removes the VRAM floor at the cost of bit-exactness; pause/
resume/cancel job control; an OpenAI-compatible server; Docker + installers
for NVIDIA, AMD (untested on real hardware), and CPU fallback. 267 tests.

**Tried and correctly killed, not hidden:** self-drafting with an
*untrained* model (0% acceptance), a live bandit tuning draft length
(never beat a tuned constant), CPU/GPU split decode (passed its isolated
throughput gate, then made the engine 0.52x once integrated). Full
reasoning for each: [docs/RESULTS_LOG.md](docs/RESULTS_LOG.md).

---

## What will never be claimed

- No lossless bf16 ratio above ~1.51x — proven impossible, not a current
  limitation.
- No "lossless" label on any run using `quantize="q8"` or
  `lm_head_slice_rows > 0`.
- No speed or memory number that wasn't measured on real hardware with a
  cold cache, on the same counter as whatever it's compared against.
- No "faster than AirLLM" claim without stating whether VRAM was matched.

---

## Prior research (archived, not deleted)

An earlier phase explored a different idea: caching a linear layer's
*outputs* for previously-seen activation directions. Real math, 67 tests,
correctly killed after a real-model measurement came in 250-450x above the
success threshold. Code kept and marked archived in its own docstrings.
Full history: [docs/archive/](docs/archive/).

---

## Documents

| | |
|---|---|
| [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | **start here** — every method, next to AirLLM, with the honest verdict on each |
| [docs/RESULTS_LOG.md](docs/RESULTS_LOG.md) | append-only ledger of every measured run, including regressions and corrections |
| [docs/LITERATURE.md](docs/LITERATURE.md) | research survey — lossless codecs, MoE offloading, adaptive/RL-for-speculation |
| [docs/archive/](docs/archive/) | superseded planning docs and early results, kept for traceability |

---

## Name

"Afterimage" — what persists after the light has passed through.
