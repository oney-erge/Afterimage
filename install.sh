#!/usr/bin/env bash
# Sets up Afterimage on Linux/WSL2/macOS the first time; every time after
# that, just starts it. Run it with `./start` at the repo root, or this
# file directly -- same thing.
#
# Detects your GPU, installs the matching torch build, creates a venv,
# editable-installs the package, and runs the hardware check. If it's
# already set up here, it skips straight to launching the server.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

log() { echo "[start] $*"; }

detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "nvidia"
  elif command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; then
    echo "amd"
  elif [ "$(uname -s)" = "Darwin" ]; then
    echo "mac"
  else
    echo "none"
  fi
}

if [ -x "$VENV_DIR/bin/afterimage" ]; then
  if [ "${1:-}" = "--reinstall" ]; then
    log "rebuilding from scratch"
    rm -rf "$VENV_DIR"
  else
    exec "$VENV_DIR/bin/afterimage" serve
  fi
fi

log "first run -- setting things up (this takes a few minutes)"
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
  mac)
    log "macOS: this runs CPU-only, and that's a real limit worth stating plainly."
    log "Afterimage's decode kernels need CUDA, which Macs don't have. A streamed"
    log "14B model on CPU is slow enough to be a demo, not a daily tool."
    log "If your Mac has enough unified memory to hold the model directly, a"
    log "normal (non-streaming) runtime will just be faster -- Afterimage exists"
    log "for the case where the model doesn't fit, which unified memory often avoids."
    pip install torch
    pip install -e "$REPO_DIR[server]"
    ;;
  none)
    log "no GPU detected -- installing CPU-only torch (inference will be slow;"
    log "the GPU decode kernels in afterimage/runtime/gpu_decode*.py require CUDA)"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -e "$REPO_DIR[server]"
    ;;
esac

afterimage doctor || true

log "set up. Running a small model end to end first, so you can see it work"
log "before waiting on a big download. Ctrl-C to skip straight to the server."
afterimage quickstart --yes || log "that didn't finish -- see above, but the server will still start"

log "starting the server (Ctrl-C to stop; run this script again any time)"
exec afterimage serve
