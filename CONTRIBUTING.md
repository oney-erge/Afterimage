# Contributing

## Setup

```bash
git clone https://github.com/iodriller/Afterimage.git
cd Afterimage
pip install -e ".[dev,server]"    # add "gpu" too if you have an NVIDIA card
```

## Tests

```bash
python -m pytest -q                 # default run -- CPU-safe, excludes the
                                     # killed Phase-0 research branch and GPU-only tests
python -m pytest -q -m ""           # everything, including the Phase-0 branch
python -m pytest -q -m archive      # only the Phase-0 branch
```

GPU-dependent tests (`test_*_gpu.py`) skip automatically without CUDA +
Triton. `pytest.importorskip` guards optional-dependency tests (`transformers`,
`fastapi`) the same way. CI runs the default set plus the Phase-0 branch as
separate jobs (`.github/workflows/ci.yml`) on CPU only.

```bash
python -m compileall -q afterimage  # what CI checks before tests
```

## Code style

No enforced formatter yet. Match the surrounding file: comments explain
*why*, not what (the code should read for what); avoid fabricated numbers —
every measured claim in this codebase traces back to a real run in
`results/` or `docs/RESULTS_LOG.md`.

## Where things live

- `afterimage/runtime/` — the streaming engine, compression, planners
- `afterimage/server/` — the FastAPI control API + web UI
- `afterimage/probe/`, `afterimage/testing/`, and the modules marked
  `ARCHIVED` in their own docstrings — the killed Phase-0 branch, kept for
  traceability, not the current engine (`docs/archive/README.md`)
- `docs/RESEARCH_METHODS.md` — the H0-H18 opt-in research layer's
  hypotheses, protocols, and kill gates
- `docs/RESULTS_LOG.md` — the append-only measurement ledger; a real run's
  numbers get appended here, never edited retroactively

## Adding a research hypothesis

If you're extending the H0-H18 program: read
[RESEARCH_METHODS.md](docs/RESEARCH_METHODS.md) section 4 (the experiment
contract) first. Every hypothesis needs a named `MethodProfile`, a named
control, a numeric kill gate declared *before* running it, and its result
recorded in `docs/RESULTS_LOG.md` regardless of outcome — a negative result
is still a result and must not be deleted or silently retried away.

## Pull requests

Describe what changed and, if it's a performance claim, how it was
measured (hardware, cache state, repeat count) — see
`docs/RESEARCH_METHODS.md` section 4 for what a credible measurement needs.
Small, focused PRs over large ones.

## Reporting issues

Use GitHub Issues. For anything that might be a security concern, see
[SECURITY.md](SECURITY.md) instead of a public issue.
