#!/usr/bin/env bash
# Paper 1: H6.5 practical-throughput Gemma 2 27B campaign on the RTX 3080.
#
# The Gemma plan is calibrated independently from Qwen, frozen before the
# reporting matrices, and reused unchanged across matched, external, and
# sustained-decode stages. Every runner is checkpointed and resume-safe.
set -uo pipefail

REPO="/mnt/c/for fun/Afterimage"
PYTHON="/root/.venv/bin/python"
MODEL="google/gemma-2-27b-it"
STORE="/root/afterimage/paper1/store_gemma2_27b"
STAMP="20260903"
ROOT="$REPO/scripts/local/paper1/output/paper-h65-gemma-practical-$STAMP"
LOG_DIR="$ROOT/logs"
STATUS_FILE="$ROOT/campaign-status.txt"
PILOT_OUT="$ROOT/planner/gemma2-27b-h65-pilot-practical-$STAMP.json"
PILOT_ROOT="$ROOT/planner/gemma2-27b-h65-pilot-practical-$STAMP-artifacts"
H65_PLAN="$PILOT_ROOT/h65-full-candidate.json"
DISK_PLAN="$PILOT_ROOT/disk-plan.json"
H2D="$REPO/scripts/local/paper1/output/pageable-h2d-rtx3080-20260828.json"
HOST_NVIDIA_SMI="/mnt/c/Windows/System32/nvidia-smi.exe"

mkdir -p "$ROOT/planner" "$ROOT/matched-ttft" "$ROOT/external-ttft" \
    "$ROOT/decode" "$LOG_DIR"
cd "$REPO"

record_gpu() {
    local label=$1
    {
        printf '\n[%s] %s\n' "$(date -Is)" "$label"
        nvidia-smi -i 0 --query-gpu=name,temperature.gpu,power.draw,enforced.power.limit,pstate,clocks_throttle_reasons.active --format=csv,noheader,nounits
        if [[ -x "$HOST_NVIDIA_SMI" ]]; then
            "$HOST_NVIDIA_SMI" -i 0 --query-gpu=name,temperature.gpu,power.draw,pstate,clocks_throttle_reasons.active --format=csv,noheader,nounits
        fi
    } >> "$LOG_DIR/gpu-state.log" 2>&1 || true
}

finish() {
    local status=$?
    printf 'finished_at=%s exit_status=%s\n' "$(date -Is)" "$status" \
        >> "$STATUS_FILE"
}
trap finish EXIT

run_stage() {
    local name=$1
    shift
    local log="$LOG_DIR/$name.log"
    printf '\n[%s] START %s\n' "$(date -Is)" "$name" \
        | tee -a "$log" "$STATUS_FILE"
    record_gpu "before $name"
    "$@" 2>&1 | tee -a "$log"
    local stage_status=${PIPESTATUS[0]}
    record_gpu "after $name"
    printf '[%s] END %s exit_status=%s\n' "$(date -Is)" "$name" \
        "$stage_status" | tee -a "$log" "$STATUS_FILE"
    return "$stage_status"
}

pilot_is_valid() {
    [[ -s "$PILOT_OUT" && -s "$H65_PLAN" && -s "$DISK_PLAN" ]] || return 1
    "$PYTHON" - "$PILOT_OUT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
gates = result.get("gates", {})
required = (
    "all_cells_completed",
    "expected_evaluation_rows",
    "exact_output_tokens",
    "requested_token_count_completed",
    "cold_cache_confirmed",
    "whole_cell_vram_measured",
    "full_candidate_within_budget",
    "full_candidate_diverged_from_traffic",
    "source_snapshot_retained",
)
valid = (
    result.get("status") == "complete"
    and not result.get("exactness_failures")
    and all(gates.get(key) is True for key in required)
)
raise SystemExit(0 if valid else 1)
PY
}

if [[ ! -f "$STORE/manifest.json" || ! -f "$H2D" ]]; then
    printf 'ABORT missing Gemma store manifest or H2D calibration\n' \
        | tee -a "$STATUS_FILE"
    exit 2
fi

export AFTERIMAGE_BENCHMARK_POWER_PROFILE="unmodified Windows laptop profile; software thermal scaling recorded, not excluded"
export AFTERIMAGE_GPU_POWER_CONTROL="unavailable: nvidia-smi power-limit and clock-lock requests rejected"
# Gemma's 27B tensor stream allocates several large temporary blocks.  Use the
# same expandable CUDA allocator policy for every method so fragmentation does
# not turn a feasible frozen plan into a method-specific OOM.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
printf 'started_at=%s model=%s thermal_policy=record_and_pair\n' \
    "$(date -Is)" "$MODEL" | tee -a "$STATUS_FILE"

if ! pilot_is_valid; then
    pilot_args=(
        "$PYTHON" -u scripts/run_h65_paper_matrix.py
        --model "$MODEL" --store "$STORE" --h2d "$H2D"
        --vram-gb 4 --ram-gb 8 --vram-safety-margin-gb 0.5
        --decode-slice-elems 4194304 --search-iterations 128
        --minimum-live-improvement 0.0 --max-new-tokens 1 --blocks 4
        --seed 20260902 --cooldown-seconds 45 --cooldown-max-temp-c 50
        --cell-timeout-minutes 45 --wait-for-gpu-minutes 10
        --out "$PILOT_OUT"
    )
    if [[ -s "${PILOT_OUT}.partial" ]]; then
        pilot_args+=(--resume)
    fi
    run_stage h65_planner_and_four_block_gate "${pilot_args[@]}" || true
fi

if ! pilot_is_valid; then
    printf 'ABORT Gemma H6.5 pilot failed an exactness, completeness, budget, or divergence gate\n' \
        | tee -a "$STATUS_FILE"
    exit 3
fi

# Same-budget causal comparison with the independently frozen Gemma plan.
run_stage matched_tier_ttft_8block \
    "$PYTHON" -u scripts/run_paper_comparison.py \
    --model "$MODEL" --store "$STORE" \
    --methods exact-min,simple-v4-r8 \
    --afterimage-plan-method "disk-frozen=per_tensor:$DISK_PLAN" \
    --afterimage-plan-method "h65-selected=$H65_PLAN" \
    --token-lengths 1 --blocks 8 --warmup-tokens 1 \
    --cooldown-seconds 45 --cooldown-max-temp-c 50 \
    --time-budget-minutes-per-length 720 --out-dir "$ROOT/matched-ttft" \
    --run-label "gemma2-27b-h65-matched-ttft-practical-$STAMP" \
    --resume --require-complete || true

# External systems retain their naturally achieved memory points.
run_stage external_ttft_8block \
    "$PYTHON" -u scripts/run_paper_comparison.py \
    --model "$MODEL" --store "$STORE" \
    --methods exact-min,airllm,accelerate,deepspeed-zero-inference \
    --afterimage-plan-method "h65-selected=$H65_PLAN" \
    --token-lengths 1 --blocks 8 --warmup-tokens 1 \
    --cooldown-seconds 45 --cooldown-max-temp-c 50 \
    --time-budget-minutes-per-length 720 --out-dir "$ROOT/external-ttft" \
    --run-label "gemma2-27b-h65-external-ttft-practical-$STAMP" \
    --resume --require-complete || true

# Same three-block 32-token sustained-generation protocol used for Qwen.
run_stage matched_tier_decode_32tok_3block \
    "$PYTHON" -u scripts/run_paper_comparison.py \
    --model "$MODEL" --store "$STORE" \
    --methods exact-min,simple-v4-r8 \
    --afterimage-plan-method "disk-frozen=per_tensor:$DISK_PLAN" \
    --afterimage-plan-method "h65-selected=$H65_PLAN" \
    --prompt-suite paper_generation --token-lengths 32 --blocks 3 \
    --warmup-tokens 1 --cooldown-seconds 45 --cooldown-max-temp-c 50 \
    --time-budget-minutes-per-length 1440 --out-dir "$ROOT/decode" \
    --run-label "gemma2-27b-h65-matched-decode-practical-$STAMP" \
    --resume --require-complete || true

printf 'campaign_stages_finished_at=%s\n' "$(date -Is)" \
    | tee -a "$STATUS_FILE"
