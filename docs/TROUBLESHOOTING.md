# Troubleshooting

Every failure below is real. It actually happened during this project's own
development. If you hit something not listed here, `afterimage --debug ...`
shows the full traceback instead of the short diagnosis.

## `ModuleNotFoundError: No module named 'transformers'`

The `[gpu,server]` (or `[server]`) extras aren't installed.

```bash
pip install -e ".[gpu,server]"    # NVIDIA
pip install -e ".[server]"        # CPU / AMD
```

## `No compressed store at ... -- run afterimage compress`

`afterimage run`/`serve` needs a compressed store first:

```bash
afterimage compress MODEL_ID
```

## Ran out of disk mid-compression

Nothing checked free disk space before this project added
`afterimage compress --dry-run` and the automatic preflight. If you're on
an old build, upgrade, or estimate manually: a model needs roughly
`download_size + download_size / 1.45` bytes of disk at once. That's the
original download plus the compressed store, both present until the pass
finishes.

```bash
afterimage compress MODEL_ID --dry-run
```

## `CUDA out of memory`

- Lower `--vram-budget-gb`, or drop it entirely for the minimum-memory floor.
- `--lm-head-slice-rows 8192` (or similar) shrinks the output-head VRAM floor
  by over a gigabyte on a 14B model, but is **not bit-exact**. See
  [CONFIGURATION.md](CONFIGURATION.md).
- PyTorch's caching allocator never returns freed blocks to the driver, so
  observed VRAM climbs to a high-water mark across a long process even when
  nothing is actually leaking. `--vram-cap-gb` makes the allocator reuse its
  own cache instead of requesting more, turning a slow creep toward OOM into
  an immediate, legible error at a known ceiling.

## WSL2: pinned-RAM / `pin_memory()` failures

WSL2 commonly limits locked memory (`ulimit -l`) to 64 MB, smaller than a
single realistic decoder-layer tensor. `--ram-tier-format decoded` (the
default RAM-tier format) needs to pin memory. When it can't, the engine
degrades to pageable RAM automatically and warns instead of failing, unless
`--require-pinned-ram` is set, which fails closed instead (this is what the
H9 regulated research protocol needs).

Fixes, in order of effort:
- `--ram-tier-format compressed` avoids `pin_memory()` entirely. It caches
  compressed bytes instead of a decoded pinned tensor, at the cost of a real
  GPU decode per token instead of a memcpy.
- Raise the limit: add `ulimit -l unlimited` (or a specific value) to your
  WSL2 startup, or set it in `/etc/security/limits.conf` inside the distro.
- Run on native Linux, where this ceiling doesn't usually apply.

## "It's taking forever and I don't know if it's stuck"

`afterimage run` prints a rough ETA before generation starts; `--stats`
prints the real, measured numbers once generation finishes. On the
reference 14B model and 8 GB card, the exact minimum-memory profile
measures 32.514 s/token, exact residency at 4 GB measures 17.360 s/token,
and fixed speculation measures 9.150 s/token -- see the README's benchmark
table for the full picture. `afterimage quickstart` only reports timing for
its small (0.6B) model, so it isn't a predictor of these 14B numbers;
`--stats` on a real run is the way to see what your own hardware does.
That's expected, not a hang. `--quiet` suppresses progress output if you'd
rather redirect to a log.

## Unsupported model architecture

```
RuntimeError: unsupported model architecture [...] for MODEL_ID: this engine
requires a Llama-family layout (top-level .model.layers) -- Qwen, Llama and
Mistral-family checkpoints work; GPT-2/Falcon/MPT-family ones do not.
```

This is a real, permanent limitation, not a bug. See
[CONFIGURATION.md](CONFIGURATION.md#supported-model-architectures). There's
no workaround short of a different model.

## The web UI shows "not feasible" for a budget that should fit

`/api/plan` checks the *compressed store's* manifest, which must already
exist (`afterimage compress` first). It also doesn't know about other
processes using the GPU at the same time. Check `nvidia-smi` if the
numbers look wrong.

## macOS: everything works but it's slow

Expected. Afterimage's GPU decode kernels need CUDA, which Macs don't
have, so it falls back to CPU. A streamed 14B model on CPU is slow enough
to feel broken even though it's working correctly. If your Mac's unified
memory is large enough to hold the model directly, a normal
(non-streaming) runtime will just be faster for you: Afterimage exists for
the case where the model doesn't fit, and unified memory often means it
does.

## Docker: container starts but health check fails

```bash
docker compose logs afterimage
curl http://localhost:8420/health
```

`/health` reports `model_loaded: false` until the first
`/v1/chat/completions` or `/api/compress` call. That's expected right
after startup, not a failure. A genuinely stuck container (health check
failing for minutes) is usually a GPU passthrough problem: confirm
`nvidia-ctk`/`nvidia-docker2` is installed and
`docker compose run afterimage nvidia-smi` sees the GPU.
