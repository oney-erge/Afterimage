#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "canonical" ]]; then
  printf 'Usage: ./benchmark.sh canonical [runner options...]\n' >&2
  exit 2
fi
shift

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

model="${AFTERIMAGE_MODEL:-Qwen/Qwen3-14B}"
store="${AFTERIMAGE_STORE:-/root/afterimage/store_14b}"
offload_dir="${AFTERIMAGE_HF_OFFLOAD_DIR:-/root/afterimage/hf_offload_14b}"
out="${AFTERIMAGE_BENCHMARK_OUT:-results/qwen3-14b-canonical.json}"
export AFTERIMAGE_HF_OFFLOAD_DIR="$offload_dir"

[[ -z "$(git status --short)" ]] || {
  echo 'refusing to run: git tree is dirty; commit/stash changes or use the suite directly with --allow-dirty-tree' >&2
  exit 1
}
python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "CUDA is unavailable")'
[[ -f "$store/manifest.json" ]] || { echo "missing Afterimage store: $store" >&2; exit 1; }
command -v sync >/dev/null || { echo 'missing sync command required for cache drops' >&2; exit 1; }
[[ -w /proc/sys/vm/drop_caches ]] || { echo 'cache-drop permission unavailable: /proc/sys/vm/drop_caches is not writable' >&2; exit 1; }
python - "$model" "$store" "$offload_dir" <<'PY'
import importlib.metadata as metadata
import pathlib
import shutil
import sys

for package in ("airllm", "accelerate"):
    try:
        print(f"{package} {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        raise SystemExit(f"missing benchmark dependency: {package}")

store, offload = map(pathlib.Path, sys.argv[2:])
for path in (store, offload):
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    if usage.free < 10 * 1024**3:
        raise SystemExit(f"less than 10 GiB free near {path}")
print(f"model {sys.argv[1]}")
print(f"store {store}")
print(f"offload {offload}")
PY

python -u scripts/run_bounded_suite.py \
  --model "$model" \
  --store "$store" \
  --methods airllm,accelerate,exact-min,exact-resident,spec-fixed \
  --repeats "${AFTERIMAGE_BENCHMARK_REPEATS:-5}" \
  --time-budget-minutes "${AFTERIMAGE_BENCHMARK_TIME_MINUTES:-300}" \
  --out "$out" "$@"