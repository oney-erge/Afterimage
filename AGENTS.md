# Afterimage Repository Instructions

Afterimage is a lossless weight-streaming runtime and research workbench for
running supported BF16 language models under limited GPU memory. Package code
lives under `src/afterimage/`; tests and benchmark evidence must distinguish
measured results from projections.

## Focused commands

- Lint and focused tests: use the commands documented in `README.md` and
  `pyproject.toml` for the changed subsystem.
- Native application: `./run.sh` or `.\run.ps1`
- Docker application: `./run.sh docker` or `.\run.ps1 docker`

Do not generalize model-family, CUDA, ROCm, latency, or memory claims beyond
observed evidence. Preserve locked dependencies, model caches, generated
artifacts, and unrelated worktree changes.


## Install and run contract

- Keep `run.bat`, `run.ps1`, `run.command`, and `run.sh` as the stable
  user entry points. They must keep the same `run`, `doctor`, `repair`,
  `docker`, `logs`, and `stop` actions where the application supports them.
- Use the `native-app-delivery` Codex skill when changing first-run setup,
  repair, Docker, or launcher behavior. That is an internal workflow name and
  must not appear in product copy or the public README.
- Keep shared install mechanics in `scripts/install-utils.ps1` and
  `scripts/install-utils.sh`. Preserve idempotent reruns, bounded transient
  retries, install locking, disk checks, user state, and `.setup/install.log`.
- Verify launcher changes with PowerShell parsing, `bash -n`, the focused
  delivery audit, and `docker compose config`. Do not run the full application
  test suite unless the change affects application behavior.
