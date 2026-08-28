#!/usr/bin/env bash
# Restartable paper comparison. It runs short factual prompts separately from
# realistic long-form generation so TTFT and decode measurements are not mixed
# across incompatible workloads. Each output is finalized only when every
# requested method succeeds in every randomized block.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

model="${AFTERIMAGE_MODEL:-Qwen/Qwen3-14B}"
dfloat11_model="${AFTERIMAGE_DFLOAT11_MODEL:-DFloat11/Qwen3-14B-DF11}"
store="${AFTERIMAGE_STORE:-/root/afterimage/store_14b}"
hf_offload_dir="${AFTERIMAGE_HF_OFFLOAD_DIR:-/root/afterimage/hf_offload_14b}"
deepspeed_offload_dir="${AFTERIMAGE_DEEPSPEED_OFFLOAD_DIR:-/root/afterimage/deepspeed_offload_14b}"
out_dir="${AFTERIMAGE_PAPER_OUT_DIR:-results/paper-comparison}"
base_label="${AFTERIMAGE_PAPER_RUN_LABEL:-qwen3-14b-paper}"
methods="${AFTERIMAGE_PAPER_METHODS:-airllm,accelerate,deepspeed-zero-inference,exact-min,exact-resident,spec-fixed}"
export AFTERIMAGE_HF_OFFLOAD_DIR="$hf_offload_dir"
export AFTERIMAGE_DEEPSPEED_OFFLOAD_DIR="$deepspeed_offload_dir"

[[ -z "$(git status --short)" ]] || {
  echo 'refusing to run: git tree is dirty; commit or stash source changes first' >&2
  exit 1
}
python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "CUDA is unavailable")'
[[ -f "$store/manifest.json" ]] || { echo "missing Afterimage store: $store" >&2; exit 1; }
command -v sync >/dev/null || { echo 'missing sync command required for cache drops' >&2; exit 1; }
[[ -w /proc/sys/vm/drop_caches ]] || { echo 'cache-drop permission unavailable: /proc/sys/vm/drop_caches is not writable' >&2; exit 1; }

python - "$model" "$store" "$hf_offload_dir" "$deepspeed_offload_dir" "$methods" <<'PY'
import importlib.metadata as metadata
import pathlib
import shutil
import sys

model, store_value, hf_value, ds_value, methods_value = sys.argv[1:]
package_for_method = {
    "airllm": "airllm",
    "accelerate": "accelerate",
    "dfloat11": "dfloat11",
    "dfloat11-gpu-resident": "dfloat11",
    "deepspeed-zero-inference": "deepspeed",
}
selected = [part.strip() for part in methods_value.split(",") if part.strip()]
required = sorted({package_for_method[method] for method in selected
                   if method in package_for_method})
missing = []
for package in required:
    try:
        print(f"{package} {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        missing.append(package)
if missing:
    raise SystemExit(
        f"missing benchmark packages {missing}; run `pip install -e .[bench]` before the campaign")
if "deepspeed-zero-inference" in selected:
    try:
        from deepspeed.ops.op_builder import AsyncIOBuilder
        aio_ready = AsyncIOBuilder().is_compatible()
    except Exception as exc:
        raise SystemExit(f"DeepSpeed NVMe preflight could not inspect async I/O: {exc}")
    if not aio_ready:
        raise SystemExit(
            "DeepSpeed NVMe async I/O is unavailable; install libaio development headers "
            "and reinstall deepspeed before starting the campaign")

store = pathlib.Path(store_value)
paths = [store, pathlib.Path(hf_value)]
if "deepspeed-zero-inference" in selected:
    paths.append(pathlib.Path(ds_value))
for path in paths:
    probe = path if path.exists() else path.parent
    usage = shutil.disk_usage(probe)
    if usage.free < 35 * 1024**3:
        raise SystemExit(f"less than 35 GiB free near {path}")
print(f"model {model}")
print(f"store {store}")
print(f"HF offload {hf_value}")
if "deepspeed-zero-inference" in selected:
    print(f"DeepSpeed NVMe offload {ds_value}")
PY

common_args=(
  --model "$model"
  --dfloat11-model "$dfloat11_model"
  --store "$store"
  --methods "$methods"
  --blocks "${AFTERIMAGE_PAPER_BLOCKS:-3}"
  --warmup-tokens "${AFTERIMAGE_PAPER_WARMUP_TOKENS:-8}"
  --cooldown-seconds "${AFTERIMAGE_PAPER_COOLDOWN_SECONDS:-20}"
  --cooldown-max-temp-c "${AFTERIMAGE_PAPER_COOLDOWN_MAX_TEMP_C:-75}"
  --time-budget-minutes-per-length "${AFTERIMAGE_PAPER_TIME_BUDGET_MINUTES:-240}"
  --out-dir "$out_dir"
)

status=0
python -u scripts/run_paper_comparison.py "${common_args[@]}" "$@" \
  --prompt-suite evaluation \
  --token-lengths "${AFTERIMAGE_PAPER_EVALUATION_LENGTHS:-1,4}" \
  --run-label "${base_label}-evaluation" \
  --resume --require-complete || status=$?

python -u scripts/run_paper_comparison.py "${common_args[@]}" "$@" \
  --prompt-suite paper_generation \
  --token-lengths "${AFTERIMAGE_PAPER_GENERATION_LENGTHS:-1,32,128}" \
  --run-label "${base_label}-generation" \
  --resume --require-complete || status=$?

echo "paper campaign outputs: $out_dir/${base_label}-{evaluation,generation}-*.json"
echo "results/INDEX.md is unchanged; curate accepted artifacts before rebuilding the public index"
exit "$status"
