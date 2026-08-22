#!/usr/bin/env bash
# Afterimage installer for Linux/WSL2: detects GPU vendor, installs the
# matching torch build, creates a venv, editable-installs the package, and
# runs the hardware diagnosis. If Afterimage is already installed here,
# re-running this script just launches the server instead of reinstalling
# -- safe to bind to a desktop shortcut or a `make run`-style habit.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

log() { echo "[install] $*"; }

detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "nvidia"
  elif command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; then
    echo "amd"
  else
    echo "none"
  fi
}

if [ -x "$VENV_DIR/bin/afterimage" ]; then
  log "already installed at $VENV_DIR -- launching (re-run with --reinstall to rebuild the venv)"
  if [ "${1:-}" = "--reinstall" ]; then
    log "removing existing venv for a clean reinstall"
    rm -rf "$VENV_DIR"
  else
    exec "$VENV_DIR/bin/afterimage" serve
  fi
fi

log "setting up a new environment at $VENV_DIR"
GPU_VENDOR="$(detect_gpu)"
log "detected GPU vendor: $GPU_VENDOR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel >/dev/null

case "$GPU_VENDOR" in
  nvidia)
    log "installing CUDA torch build"
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install -e "$REPO_DIR[gpu,server]"
    ;;
  amd)
    log "installing ROCm torch build"
    log "NOTE: ROCm/AMD support in this project is built against a device"
    log "      abstraction but has not been exercised on real AMD hardware --"
    log "      treat it as untested, not verified. See docs/archive/MASTER_PLAN.md."
    pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
    pip install -e "$REPO_DIR[server]"
    ;;
  none)
    log "no GPU detected -- installing CPU-only torch (inference will be slow;"
    log "the GPU decode kernels in afterimage/runtime/gpu_decode*.py require CUDA)"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -e "$REPO_DIR[server]"
    ;;
esac

log "running hardware diagnosis"
afterimage doctor || true

log "install complete."
log "Launching the server (Ctrl-C to stop; re-run this script any time to relaunch)."
exec afterimage serve
