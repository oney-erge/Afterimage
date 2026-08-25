# Changelog

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
No tagged releases yet. Everything below is `main`.

## [Unreleased]

### Added
- `afterimage quickstart`: compress + run a small model end to end (~2 GB,
  minutes) to prove an install works before committing to a 14B-class model.
- `--profile {min-memory,balanced,fast}` and `--auto` on `afterimage run` --
  the README's measured operating points as presets, with automatic
  selection from detected VRAM.
- Speculative decoding reachable from the CLI (`--draft-model`, `--spec-k`,
  `--spec-temperature`) and the API (`draft_model`, `spec_k` on
  `/v1/chat/completions`), previously only reachable by writing Python
  against `generate_adaptive` directly.
- `afterimage compress --dry-run` and an automatic disk-space preflight
  before a real compression pass.
- Operability: `GET /health`, `GET /api/version`, `GET /api/stats`;
  `afterimage --version`; structured logging in the server
  (`--log-level` on `afterimage serve`).
- Measured per-token stats (`seconds_per_token`, `peak_vram_gb`,
  `bytes_read_gb`, prefetch/speculation counters) in the chat completion
  response's `usage.afterimage` block, and live in the web UI.
- The web UI's chat panel now has the same dial as the CLI: a profile
  picker, a VRAM budget field with live `/api/plan` feasibility checking,
  and a draft-model field.
- Research subcommands grouped under `afterimage research …`
  (`experiments`, `test-plan`, `pin-preflight`, `profile-trace`,
  `optimize-residency`) instead of top-level, separating the H0-H15
  research layer from ordinary use.
- H12 (Bayesian chance-constrained prefetch), H13 (event-interference QUBO
  residency), H14 (coalesced contiguous storage reads), H15 (physical-extent
  QUBO residency) research candidates.
- `docs/CONFIGURATION.md`, `docs/USAGE.md`, `docs/TROUBLESHOOTING.md`,
  `docs/HYPOTHESIS_LINEAGE.md`.

### Fixed
- The default `per_blob` storage-read path was synthesizing byte-proportional
  per-tensor read timing instead of measuring it for real, which would have
  silently corrupted any critical-path profile built from it.
- Both QUBO residency planners' `repair()` step backfilled freed VRAM budget
  using the same value ranking their own control already uses, so an
  annealed candidate always collapsed back onto the deterministic control
  before scoring (H13/H15 could never diverge from their seed).
- The server's engine cache keyed on 5 hand-picked `EngineConfig` fields
  instead of the full config fingerprint, so a request that changed e.g.
  `draft_mode` but matched on those 5 fields silently reused the previous
  request's engine with the previous request's settings.
- `require_pinned_ram` only checked at the point of an actual `pin_memory()`
  failure, deep inside generation; now checked at engine construction, so a
  regulated H9 run fails closed before loading anything.
- Version was declared twice (`pyproject.toml` and `__init__.py`) and had
  drifted out of sync; now single-sourced via package metadata.

### Removed
- `configs/hardware.yaml` and `configs/models.yaml`, read by no code
  (`pyyaml` was not even a dependency) and described a stale pre-measurement
  state of the project (no GPU, `transformers` not installed) that
  contradicted the extensive real-hardware measurements now in
  `docs/RESULTS_LOG.md`.

## Earlier history

Not tracked as discrete releases. See `git log` for the full commit history;
`docs/RESULTS_LOG.md` for the append-only measurement history;
git history for superseded design documents.
