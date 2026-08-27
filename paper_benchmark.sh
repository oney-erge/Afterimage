#!/usr/bin/env bash
# One-click paper comparison: Afterimage vs AirLLM vs HF Accelerate vs
# DFloat11, randomized-block, 3 passes by default, at 4/32/128 output
# tokens. Wraps scripts/run_paper_comparison.py the same way benchmark.sh
# wraps scripts/run_bounded_suite.py -- see that script's own module
# docstring for why block-major randomization exists and what it costs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

model="${AFTERIMAGE_MODEL:-Qwen/Qwen3-14B}"
dfloat11_model="${AFTERIMAGE_DFLOAT11_MODEL:-DFloat11/Qwen3-14B-DF11}"
store="${AFTERIMAGE_STORE:-/root/afterimage/store_14b}"
offload_dir="${AFTERIMAGE_HF_OFFLOAD_DIR:-/root/afterimage/hf_offload_14b}"
out_dir="${AFTERIMAGE_PAPER_OUT_DIR:-results/paper-comparison}"
export AFTERIMAGE_HF_OFFLOAD_DIR="$offload_dir"

[[ -z "$(git status --short)" ]] || {
  echo 'refusing to run: git tree is dirty; commit/stash changes or use run_paper_comparison.py directly with --allow-dirty-tree' >&2
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

missing = []
for package in ("airllm", "accelerate", "dfloat11"):
    try:
        print(f"{package} {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        missing.append(package)
if missing:
    print(f"WARNING: missing packages {missing} -- those methods will fail "
          f"per-block rather than block the run. Install with "
          f"`pip install -e .[bench]` to include all three.", file=sys.stderr)

store, offload = map(pathlib.Path, sys.argv[2:])
for path in (store, offload):
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    if usage.free < 10 * 1024**3:
        raise SystemExit(f"less than 10 GiB free near {path}")
print(f"model {sys.argv[1]}")
print(f"store {store}")
print(f"offload {offload}")
PY

python -u scripts/run_paper_comparison.py \
  --model "$model" \
  --dfloat11-model "$dfloat11_model" \
  --store "$store" \
  --methods "${AFTERIMAGE_PAPER_METHODS:-airllm,accelerate,dfloat11,exact-min,exact-resident,spec-fixed}" \
  --token-lengths "${AFTERIMAGE_PAPER_TOKEN_LENGTHS:-4,32,128}" \
  --blocks "${AFTERIMAGE_PAPER_BLOCKS:-3}" \
  --warmup-tokens "${AFTERIMAGE_PAPER_WARMUP_TOKENS:-8}" \
  --cooldown-seconds "${AFTERIMAGE_PAPER_COOLDOWN_SECONDS:-20}" \
  --cooldown-max-temp-c "${AFTERIMAGE_PAPER_COOLDOWN_MAX_TEMP_C:-75}" \
  --time-budget-minutes-per-length "${AFTERIMAGE_PAPER_TIME_BUDGET_MINUTES:-90}" \
  --out-dir "$out_dir" "$@"

python scripts/build_results_index.py > results/INDEX.md
echo "wrote $out_dir/*.json and refreshed results/INDEX.md"
