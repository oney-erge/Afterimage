# Contributing

## Setup

```bash
git clone https://github.com/oney-erge/Afterimage.git
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

GPU-dependent tests skip automatically without CUDA + Triton. Most are named
`test_*_gpu.py`, but not all (`test_sliced_decompress.py` is guarded the same
way), so read the skip reasons rather than the filenames:

```bash
python -m pytest -q -rs             # -rs prints why each test skipped
```

`pytest.importorskip` guards optional-dependency tests (`transformers`,
`fastapi`) the same way. CI runs the default set plus the Phase-0 branch as
separate jobs (`.github/workflows/ci.yml`) **on CPU only**, so the GPU decode
kernels, the streaming engine, and the chunked LM head are not covered by any
automated run. Before a release, run the suite on a CUDA host as well; on
Windows that means WSL2, because Triton publishes no native Windows wheel
(see `docs/TROUBLESHOOTING.md`). The suite grows over time, so don't trust a
specific pass/skip count in prose here -- run `python -m pytest -q` yourself
and read its final summary line; `-rs` names every skip's reason. Reference
point (this file's date, native Windows, CPU-only torch, `[dev,server]`
extras): 741 passed / 76 skipped, 2026-09-27.

```bash
python -m compileall -q afterimage           # what CI checks before tests
python -m ruff check afterimage scripts tests  # narrow ruleset: F + E9
python scripts/check_prose.py                # zero-em-dash docs check
```

### Before tagging a release

CI never touches a GPU, so nothing above proves the engine actually runs.
Before a release:

1. The CPU checks above, clean.
2. The GPU test suite on a CUDA host (see above).
3. `python evidence/h65-paper1/verify_paper_evidence.py` -- proves the
   H6.5 evidence bundle itself is internally consistent (no GPU needed).
4. One real `afterimage compress` + `afterimage run` on a small model
   (`afterimage quickstart`), on the platform you're about to claim support
   for. Attach that log to the release.
5. If the install scripts changed: `./run.sh repair` (or `.\run.ps1 repair`)
   on a clean checkout, on at least one NVIDIA machine.

Tag as `vX.Y.Z` matching `pyproject.toml`'s `version`, and publish a GitHub
release from that tag. Publishing the release does two things automatically,
no further action needed: [`.github/workflows/publish.yml`](.github/workflows/publish.yml)
builds the sdist and wheel and publishes them to PyPI (trusted publishing;
see that file for the one-time pypi.org setup), and, if Zenodo-GitHub
archiving is enabled for this repository, Zenodo mints a DOI for the release.

## Code style

No enforced formatter yet. Match the surrounding file: comments explain
*why*, not what (the code should read for what); avoid fabricated numbers.
Every measured claim in this codebase traces back to a real run in
`results/` or `docs/RESULTS_LOG.md`.

## Where things live

- `afterimage/runtime/`: the streaming engine, compression, planners
- `afterimage/server/`: the FastAPI control API + web UI
- `afterimage/probe/`, `afterimage/testing/`, and the modules marked
  `ARCHIVED` in their own docstrings: the killed Phase-0 branch, kept for
  traceability, not the current engine
- `docs/RESEARCH_METHODS.md`: the H0-H18 opt-in research layer's
  hypotheses, protocols, and kill gates
- `docs/RESULTS_LOG.md`: the append-only measurement ledger; a real run's
  numbers get appended here, never edited retroactively

## Adding a research hypothesis

If you're extending the H0-H18 program: read
[RESEARCH_METHODS.md](docs/RESEARCH_METHODS.md) section 4 (the experiment
contract) first. Every hypothesis needs a named `MethodProfile`, a named
control, a numeric kill gate declared *before* running it, and its result
recorded in `docs/RESULTS_LOG.md` regardless of outcome. A negative result
is still a result and must not be deleted or silently retried away.

## Pull requests

Describe what changed and, if it's a performance claim, how it was
measured (hardware, cache state, repeat count). See
`docs/RESEARCH_METHODS.md` section 4 for what a credible measurement needs.
Small, focused PRs over large ones.

## Reporting issues

Use GitHub Issues. For anything that might be a security concern, see
[SECURITY.md](SECURITY.md) instead of a public issue.
