#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./scripts/install-utils.sh
install_init "$PWD" "Afterimage"
install_enable_traps
action=run
case "${1:-}" in run|doctor|repair|docker|stop|logs) action=$1; shift ;; esac
no_browser=0
for arg in "$@"; do [ "$arg" = --no-browser ] && no_browser=1 || { echo "unknown option: $arg" >&2; exit 2; }; done
uv_version=0.12.5
url=http://127.0.0.1:8420
find_uv() { command -v uv 2>/dev/null || { [ -x "$HOME/.local/bin/uv" ] && { echo "$HOME/.local/bin/uv"; return; }; [ -x "$HOME/.cargo/bin/uv" ] && { echo "$HOME/.cargo/bin/uv"; return; }; return 1; }; }
install_uv() {
  local file; file=$(mktemp)
  install_download "https://astral.sh/uv/${uv_version}/install.sh" "$file" "uv download"
  sh "$file"; rm -f "$file"; find_uv
}
health_check() { if command -v curl >/dev/null 2>&1; then curl -fsS --max-time 2 "$url/health" >/dev/null; else wget -qO- --timeout=2 "$url/health" >/dev/null; fi; }
wait_ready() { for _ in $(seq 1 120); do health_check 2>/dev/null && return; sleep 0.5; done; return 1; }
open_url() { [ "$no_browser" -eq 1 ] && return; command -v open >/dev/null 2>&1 && open "$url" || command -v xdg-open >/dev/null 2>&1 && xdg-open "$url" || true; }

case "$action" in
  docker|stop|logs)
    if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
      [ "$action" = stop ] && { echo "The native server runs in the foreground. Press Ctrl+C in its terminal to stop it."; exit 0; }
      [ "$action" = logs ] && { echo "The native server writes logs to its foreground terminal."; exit 0; }
      command -v docker >/dev/null 2>&1 || { echo "Docker is not installed." >&2; exit 1; }
      echo "Docker is installed but its engine is not running." >&2
      exit 1
    fi
    [ "$action" = stop ] && exec docker compose down
    [ "$action" = logs ] && exec docker compose logs --follow
    install_lock
    install_require_space "$PWD" 6
    docker compose up --detach --build
    wait_ready || { docker compose logs; echo "Afterimage did not become ready at $url." >&2; exit 1; }
    install_complete
    echo "Afterimage is ready at $url"; open_url; exit 0 ;;
esac

exe=.venv/bin/afterimage
if [ "$action" = doctor ]; then [ -x "$exe" ] || { echo "Afterimage is not installed. Run ./run.sh once." >&2; exit 1; }; exec "$exe" doctor; fi
install_lock
install_require_space "$PWD" 6
uv=$(find_uv || true); [ -n "$uv" ] || uv=$(install_uv)
install_retry "Python installation" "$uv" python install 3.11
[ -x .venv/bin/python ] || "$uv" venv --python 3.11 .venv
gpu=cpu
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then gpu=nvidia
elif command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; then gpu=amd; fi
if command -v sha256sum >/dev/null 2>&1; then source_hash=$(sha256sum pyproject.toml | cut -d' ' -f1); else source_hash=$(shasum -a 256 pyproject.toml | cut -d' ' -f1); fi
fingerprint="$source_hash|uv=$uv_version|python=3.11|$gpu"
installed=$(cat .venv/.afterimage-sync 2>/dev/null || true)
if [ "$action" = repair ] || [ "$installed" != "$fingerprint" ] || [ ! -x "$exe" ]; then
  reinstall=(); [ "$action" = repair ] && reinstall+=(--reinstall)
  case "$gpu" in
    nvidia) torch_index=https://download.pytorch.org/whl/cu124; extras='.[gpu,server]' ;;
    amd) torch_index=https://download.pytorch.org/whl/rocm6.1; extras='.[server]'; echo "AMD support is experimental." ;;
    *) torch_index=https://download.pytorch.org/whl/cpu; extras='.[server]' ;;
  esac
  install_retry "PyTorch installation" "$uv" pip install --python .venv/bin/python "${reinstall[@]}" torch --index-url "$torch_index"
  install_retry "Afterimage installation" "$uv" pip install --python .venv/bin/python "${reinstall[@]}" --editable "$extras"
  printf '%s' "$fingerprint" > .venv/.afterimage-sync
fi
"$exe" doctor
install_complete
serve_args=(serve); [ "$no_browser" -eq 0 ] && serve_args+=(--open)
exec "$exe" "${serve_args[@]}"
