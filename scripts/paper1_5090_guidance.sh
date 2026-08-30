#!/usr/bin/env bash
# Paper 1 (H6 tier-planning) data campaign, self-contained on a fresh clone.
#
# SCOPE: this reproduces Paper 1's own figure/table set only -- the H6
# representation-and-tier-planning story (Figures 2, 3, 4, 5, 7, 9 and
# Table 2), the same experiments already run on this project's RTX 3080 for
# Qwen3-14B and Gemma-2-27B. It is NOT this repository's full multi-
# hypothesis research program: H0-H18 live in
# docs/ALL_HYPOTHESES_AND_BASELINES.md and the H19-H34 speculation-tree line
# lives in docs/SPECULATION_TREE_RESEARCH.md, each with its own protocol and
# its own driver scripts. Run this one when the goal is specifically "get
# Paper 1's numbers on a second GPU," not as a stand-in for either of those.
#
# Runs every Paper 1 figure/table experiment on a second machine (a
# different GPU, e.g. an RTX 5090) against its own hardware and tags its own
# output by GPU, so the two result sets combine into one cross-hardware
# paper without the two machines confusing each other's numbers. Every
# prerequisite this script needs is either code already tracked in this
# repository or something it generates itself on first run (the
# compressed/raw stores, the model-specific H1 critical-path profile, and
# the hardware-specific pinned-H2D benchmark) -- nothing under
# scripts/local/paper1/output/ on the original machine is required, and
# none of it is copied here on purpose.
#
# All of this script's own output stays local and gitignored (under
# scripts/local/paper1/output/campaign/ by default), same as every other
# Paper 1 result -- it does NOT sync back to the repo automatically.
# Combining a second machine's numbers with this one's means copying the
# relevant JSON files over by hand (scp, a shared drive, whatever) and
# reviewing them before citing anything, the same as this project treats
# every other raw campaign artifact. Only curated, reviewed evidence gets
# committed into the tracked results/ tree, and that step is a deliberate
# decision this script does not make for you.
#
# Usage (two independent invocations -- run one, both, or repeat with
# --resume after an interruption; see the resume note near the bottom):
#
#   AFTERIMAGE_MODEL=Qwen/Qwen3-14B \
#   AFTERIMAGE_DRAFT_MODEL=Qwen/Qwen3-0.6B \
#     bash scripts/paper1_5090_guidance.sh
#
#   AFTERIMAGE_MODEL=google/gemma-2-27b-it \
#   AFTERIMAGE_DRAFT_MODEL=google/gemma-2-2b-it \
#   AFTERIMAGE_EXACT_MIN_VRAM_GB=3.0 \
#     bash scripts/paper1_5090_guidance.sh
#
# The AFTERIMAGE_EXACT_MIN_VRAM_GB override on the Gemma line is not
# optional cosmetics: Gemma 2 27B's 256K-token vocabulary makes its largest
# single tensor 2.36 GB, above the 1.80 GB "minimum memory" floor tuned for
# Qwen3-14B's smaller vocabulary. Without raising it, exact-min (the paper's
# own control method) fails outright with "VRAM budget is infeasible" --
# found by actually running this campaign, not derived on paper. If a third
# model hits the same wall, the engine's own error message states the exact
# GB the model's largest tensor needs; set this variable to at least that,
# with headroom for decode scratch.
#
# What this script does NOT do: it does not decide the storage location for
# you beyond a same-disk default, does not gate on a specific GPU vendor
# (CUDA only, checked below), and does not silently retry a failed step --
# every step's own log is kept, and a failure in one step does not stop the
# steps after it (matching the original overnight-run scripts this is
# generalized from), so one bad cell costs that figure, not the whole night.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
PY=python

# ---------------------------------------------------------------------
# Configuration. Every value is overridable; only MODEL/DRAFT_MODEL have
# no safe cross-model default and should be set explicitly per invocation
# (see the two example invocations above).
# ---------------------------------------------------------------------
MODEL="${AFTERIMAGE_MODEL:?set AFTERIMAGE_MODEL, e.g. Qwen/Qwen3-14B or google/gemma-2-27b-it}"
DRAFT_MODEL="${AFTERIMAGE_DRAFT_MODEL:?set AFTERIMAGE_DRAFT_MODEL, e.g. Qwen/Qwen3-0.6B or google/gemma-2-2b-it}"
EXACT_MIN_VRAM_GB="${AFTERIMAGE_EXACT_MIN_VRAM_GB:-1.80}"

# A filesystem-safe tag derived from the model id (org/Name -> org-name,
# lowercased) so every output file names the model that produced it without
# depending on the caller to spell one consistently.
MODEL_TAG="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/_' '--')"

# GPU tag: parsed from nvidia-smi's product name (e.g. "NVIDIA GeForce RTX
# 5090" -> "rtx5090") so 3080 and 5090 result files never collide. Override
# with AFTERIMAGE_HW_TAG if a machine has more than one distinct GPU, or if
# the parse below produces something you don't want in a filename.
if [[ -n "${AFTERIMAGE_HW_TAG:-}" ]]; then
  HW_TAG="$AFTERIMAGE_HW_TAG"
else
  HW_TAG="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 \
    | grep -oiE '(RTX|GTX|A[0-9]{3,4}|H[0-9]{2,3}|L[0-9]{2,3})[ -]?[0-9]{3,4}[a-zA-Z]*' \
    | tr -d ' -' | tr '[:upper:]' '[:lower:]')"
  HW_TAG="${HW_TAG:-unknown-gpu}"
fi

STORE_ROOT="${AFTERIMAGE_STORE_ROOT:-/root/afterimage/paper1_campaign}"
STORE="${AFTERIMAGE_STORE:-$STORE_ROOT/${MODEL_TAG}/store}"
RAW_STORE="${AFTERIMAGE_RAW_STORE:-$STORE_ROOT/${MODEL_TAG}/store_raw}"
# Everything this script produces is raw campaign output, not curated
# published evidence -- it stays under scripts/local/paper1/, which is
# entirely gitignored, same as every other Paper 1 result on this project
# (see docs/README.md's "What is deliberately not here"). This is a
# deliberate choice, not an oversight: publishing a result means curating
# it into the date-stamped results/ set by hand after review, not letting a
# campaign script write there directly.
CAMPAIGN_ROOT="${AFTERIMAGE_PAPER1_CAMPAIGN_ROOT:-scripts/local/paper1/output/campaign}"
H1_FILE="${AFTERIMAGE_H1_FILE:-$CAMPAIGN_ROOT/h1_critical_path_${MODEL_TAG}_${HW_TAG}.json}"
H2D_FILE="${AFTERIMAGE_H2D_FILE:-$CAMPAIGN_ROOT/pinned_h2d_${HW_TAG}.json}"
OUT="${AFTERIMAGE_PAPER1_OUT_DIR:-$CAMPAIGN_ROOT}"
mkdir -p "$(dirname "$H1_FILE")" "$(dirname "$H2D_FILE")"
mkdir -p "$OUT"

VRAM_BUDGETS="${AFTERIMAGE_VRAM_BUDGETS:-3:6,4:8}"
E6_VRAM_BUDGET="${AFTERIMAGE_E6_VRAM_BUDGET:-4:8}"
MAX_NEW_TOKENS="${AFTERIMAGE_MAX_NEW_TOKENS:-16}"
REPEATS="${AFTERIMAGE_REPEATS:-3}"
COOLDOWN_SECONDS="${AFTERIMAGE_COOLDOWN_SECONDS:-20}"
COOLDOWN_MAX_TEMP_C="${AFTERIMAGE_COOLDOWN_MAX_TEMP_C:-75}"
E6_BLOCKS="${AFTERIMAGE_E6_BLOCKS:-3}"
COMPRESS_WORKERS="${AFTERIMAGE_COMPRESS_WORKERS:-1}"

LOG_DIR="$OUT/logs-${MODEL_TAG}-${HW_TAG}-$(date +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
step_status=0

# ---------------------------------------------------------------------
# Preflight. Every check here exists because it failed silently, loudly, or
# expensively the first time this protocol ran without it -- see the git
# history of this file's non-portable ancestor,
# scripts/local/paper1/overnight_run.sh, for which check maps to which
# incident. Fail fast and specifically; do not let a bad environment burn
# GPU-hours before the first real cell.
# ---------------------------------------------------------------------
log "=== Paper 1 campaign: $MODEL on $HW_TAG, tree at $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ==="

[[ -z "$(git status --short)" ]] || {
  echo 'refusing to run: git tree is dirty (every underlying script checks this' >&2
  echo 'again and will refuse independently -- fix it here so the whole' >&2
  echo 'campaign does not die on step 2 after step 1 already spent an hour)' >&2
  exit 1
}

$PY -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "CUDA is unavailable")' || {
  echo 'refusing to run: CUDA is not available to this Python/torch install' >&2
  exit 1
}

$PY - "$MODEL" <<'PY'
import sys
from huggingface_hub import model_info
model = sys.argv[1]
try:
    info = model_info(model)
except Exception as exc:
    raise SystemExit(
        f"cannot reach the Hugging Face Hub or resolve {model!r}: {exc}\n"
        "If this is a gated model (Gemma, Llama, etc.), make sure an "
        "account that has accepted its license is authenticated: run "
        "`huggingface-cli login` once, or place a token at "
        "~/.cache/huggingface/token, before starting this campaign.")
if getattr(info, "gated", False):
    from huggingface_hub import whoami
    try:
        who = whoami()
    except Exception as exc:
        raise SystemExit(
            f"{model} is gated and no authenticated Hugging Face account was "
            f"found ({exc}). Accept the model's license on huggingface.co "
            "with an account, then run `huggingface-cli login` (or place a "
            "token at ~/.cache/huggingface/token) before starting this "
            "campaign -- a multi-hour run should not begin only to fail on "
            "the first download.")
    print(f"gated model {model}: authenticated as {who.get('name')}, proceeding")
PY
[[ $? -eq 0 ]] || exit 1

df_check() {
  local path="$1" need_gib="$2"
  local probe="$path"
  [[ -e "$probe" ]] || probe="$(dirname "$probe")"
  while [[ ! -e "$probe" ]]; do probe="$(dirname "$probe")"; done
  local avail_kib
  avail_kib="$(df -Pk "$probe" | tail -1 | awk '{print $4}')"
  local avail_gib=$((avail_kib / 1024 / 1024))
  if (( avail_gib < need_gib )); then
    echo "refusing to run: only ${avail_gib} GiB free near $probe, want at least ${need_gib} GiB" >&2
    echo "(a 27B-class model needs roughly: download + compressed store + raw store" >&2
    echo " all present at once mid-pass -- see the compress --dry-run estimate below)" >&2
    exit 1
  fi
  log "disk check: ${avail_gib} GiB free near $probe (wanted >= ${need_gib} GiB)"
}
df_check "$STORE_ROOT" "${AFTERIMAGE_MIN_FREE_GIB:-200}"

log "compress --dry-run size estimate for $MODEL:"
$PY -m afterimage.cli compress "$MODEL" --out "$STORE" --dry-run

# ---------------------------------------------------------------------
# Step A: compressed store (the one every experiment reads from).
# ---------------------------------------------------------------------
log "--- step A: build compressed store ---"
if [[ -f "$STORE/manifest.json" ]]; then
  log "compressed store already exists at $STORE, skipping"
else
  $PY -m afterimage.cli compress "$MODEL" --out "$STORE" \
    --workers "$COMPRESS_WORKERS" --yes \
    > "$LOG_DIR/A_compress.log" 2>&1
  status=$?
  log "compress exit=$status (log: $LOG_DIR/A_compress.log)"
  [[ $status -eq 0 ]] || { echo "compressed store build failed; nothing downstream can run without it" >&2; exit 1; }
fi

# ---------------------------------------------------------------------
# Step B: raw-BF16 store, the Figure 5 (E4.1) compression-ablation control.
# ---------------------------------------------------------------------
log "--- step B: build raw-BF16 store (Figure 5 prerequisite) ---"
if [[ -f "$RAW_STORE/manifest.json" ]]; then
  log "raw store already exists, skipping"
else
  $PY -m afterimage.cli compress "$MODEL" --force-raw-storage --out "$RAW_STORE" \
    --workers "$COMPRESS_WORKERS" --yes \
    > "$LOG_DIR/B_raw_store.log" 2>&1
  status=$?
  log "raw store build exit=$status (log: $LOG_DIR/B_raw_store.log)"
  [[ $status -eq 0 ]] || log "WARNING: raw store failed; step 4 (Figure 5) below will skip itself"
fi

# ---------------------------------------------------------------------
# Step C: hardware-specific pinned-H2D bandwidth benchmark. Not reused
# across machines on purpose -- it measures THIS machine's PCIe/pinned-
# memory transfer characteristic, not anything about the model, so a value
# copied from another GPU would silently mislabel this run's own hardware.
# ---------------------------------------------------------------------
log "--- step C: pinned H2D bandwidth benchmark (hardware-specific) ---"
if [[ -f "$H2D_FILE" ]]; then
  log "H2D benchmark already exists at $H2D_FILE, skipping"
else
  $PY -u scripts/benchmark_pinned_h2d.py --out "$H2D_FILE" \
    > "$LOG_DIR/C_h2d_benchmark.log" 2>&1
  status=$?
  log "H2D benchmark exit=$status (log: $LOG_DIR/C_h2d_benchmark.log)"
  [[ $status -eq 0 ]] || { echo "H2D benchmark failed; steps 2/5 below need it" >&2; step_status=1; }
fi

# ---------------------------------------------------------------------
# Step D: model-specific H1 critical-path calibration profile. Minimal
# scope on purpose (1 token, 1 repeat): its only output of interest is
# calibration_artifacts.critical_path, not comparative timing.
# ---------------------------------------------------------------------
log "--- step D: H1 critical-path calibration profile (model-specific) ---"
if [[ -f "$H1_FILE" ]]; then
  log "H1 profile already exists at $H1_FILE, skipping"
else
  $PY -u scripts/run_bounded_suite.py \
    --model "$MODEL" --store "$STORE" \
    --methods critical-path --case-ids fact-gold \
    --max-new-tokens 1 --repeats 1 \
    --cooldown-seconds "$COOLDOWN_SECONDS" --cooldown-max-temp-c "$COOLDOWN_MAX_TEMP_C" \
    --time-budget-minutes 30 \
    --out "$H1_FILE" \
    > "$LOG_DIR/D_h1_critical_path.log" 2>&1
  status=$?
  log "H1 profile exit=$status (log: $LOG_DIR/D_h1_critical_path.log)"
  [[ $status -eq 0 ]] || { echo "H1 profile failed; steps 2/5 below need it" >&2; step_status=1; }
fi

# ---------------------------------------------------------------------
# Smoke gate. A real single-token generation on the three memory-
# constrained Afterimage methods, BEFORE the multi-hour campaign below
# starts. This is the check that would have caught, in under two minutes
# instead of after an hour of dead cells, both real problems found the
# first time this exact protocol ran against a new model: a chat template
# that rejects a system-role message, and exact-min's VRAM floor being
# tuned for a smaller-vocabulary model than the one being run now. Both
# are fixed in this codebase as of the commit this script runs from,
# but the gate stays: the NEXT new model may hit something neither of
# those fixes covers, and this is where you want to find out.
# ---------------------------------------------------------------------
log "--- smoke gate: real single-token generation before committing hours ---"
SMOKE_OUT="$LOG_DIR/smoke.json"
$PY -u scripts/run_bounded_suite.py \
  --model "$MODEL" --draft-model "$DRAFT_MODEL" \
  --store "$STORE" \
  --methods exact-min,exact-resident,spec-fixed \
  --ram-overlay-vram-budget-gb "$EXACT_MIN_VRAM_GB" \
  --case-ids fact-gold \
  --max-new-tokens 4 --repeats 1 \
  --cooldown-seconds 0 \
  --time-budget-minutes 20 \
  --out "$SMOKE_OUT" \
  > "$LOG_DIR/smoke.log" 2>&1
smoke_status=$?
$PY - "$SMOKE_OUT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
failed = [m["method_id"] for m in r["methods"] if m.get("error")]
if failed:
    print(f"SMOKE GATE FAILED: {failed}")
    for m in r["methods"]:
        if m.get("error"):
            print(f"--- {m['method_id']} ---")
            print(m["error"])
            print((m.get("traceback") or "")[-1500:])
    raise SystemExit(1)
print("smoke gate: all methods produced a real token with no error")
PY
if [[ $? -ne 0 || $smoke_status -ne 0 ]]; then
  echo "" >&2
  echo "SMOKE GATE FAILED (log: $LOG_DIR/smoke.log, result: $SMOKE_OUT)." >&2
  echo "This is exactly the check tonight's Gemma campaign relied on to avoid" >&2
  echo "burning hours on a broken configuration. Fix the real error above" >&2
  echo "before removing this gate or forcing the campaign past it -- do not" >&2
  echo "comment this block out." >&2
  exit 1
fi

# ---------------------------------------------------------------------
# The six figure/table experiments. Parameters are identical across every
# model this protocol has been run against (Qwen3-14B, Gemma-2-27B): same
# budgets, same case IDs, same token/repeat/block counts, so results from
# different hardware and different models are directly comparable cells in
# the same table, not incidentally-different protocols.
# ---------------------------------------------------------------------
tag="${MODEL_TAG}-${HW_TAG}"

log "--- step 1/6: Figure 2/3 -- H6 budget sweep ---"
$PY -u scripts/local/paper1/run_h6_budget_sweep.py \
  --model "$MODEL" --store "$STORE" \
  --h1 "$H1_FILE" --h2d "$H2D_FILE" \
  --budgets "$VRAM_BUDGETS" --methods simple,h1,h6-disk,h6-live \
  --case-ids fact-gold \
  --max-new-tokens "$MAX_NEW_TOKENS" --repeats "$REPEATS" \
  --cooldown-seconds "$COOLDOWN_SECONDS" --cooldown-max-temp-c "$COOLDOWN_MAX_TEMP_C" \
  --time-budget-minutes 240 --resume \
  --out "$OUT/e2.2-h6-budget-sweep-${tag}.json" \
  > "$LOG_DIR/1_e2.2_h6_budget.log" 2>&1 || step_status=1
log "step 1/6 exit=$? (log: $LOG_DIR/1_e2.2_h6_budget.log)"

log "--- step 2/6: Figure 9 -- stage-level breakdown ---"
$PY -u scripts/run_bounded_suite.py \
  --model "$MODEL" --draft-model "$DRAFT_MODEL" --store "$STORE" \
  --methods breakdown-exact,breakdown-spec \
  --case-ids fact-gold,retrieval-7319 \
  --max-new-tokens "$MAX_NEW_TOKENS" --repeats "$REPEATS" \
  --cooldown-seconds "$COOLDOWN_SECONDS" --cooldown-max-temp-c "$COOLDOWN_MAX_TEMP_C" \
  --time-budget-minutes 90 \
  --out "$OUT/e8.2-breakdown-${tag}.json" \
  > "$LOG_DIR/2_e8.2_breakdown.log" 2>&1 || step_status=1
log "step 2/6 exit=$? (log: $LOG_DIR/2_e8.2_breakdown.log)"

log "--- step 3/6: Figure 5 -- compression ablation ---"
if [[ -f "$RAW_STORE/manifest.json" ]]; then
  $PY -u scripts/local/paper1/run_compression_ablation.py \
    --model "$MODEL" \
    --compressed-store "$STORE" --raw-store "$RAW_STORE" \
    --budgets "$E6_VRAM_BUDGET" --case-ids fact-gold,retrieval-7319 \
    --max-new-tokens "$MAX_NEW_TOKENS" --repeats "$REPEATS" \
    --cooldown-seconds "$COOLDOWN_SECONDS" --cooldown-max-temp-c "$COOLDOWN_MAX_TEMP_C" \
    --time-budget-minutes 150 --resume \
    --out "$OUT/e4.1-compression-ablation-${tag}.json" \
    > "$LOG_DIR/3_e4.1_compression.log" 2>&1 || step_status=1
  log "step 3/6 exit=$? (log: $LOG_DIR/3_e4.1_compression.log)"
else
  log "step 3/6 SKIPPED: raw store (step B) is missing"
fi

log "--- step 4/6: Figure 4 -- H6 plan-robustness sweep ---"
$PY -u scripts/local/paper1/run_h6_budget_sweep.py \
  --model "$MODEL" --store "$STORE" --h1 "$H1_FILE" \
  --budgets "$E6_VRAM_BUDGET" --methods h6-live \
  --h2d-gbps-overrides 1,2,4,8,16 \
  --case-ids fact-gold \
  --max-new-tokens "$MAX_NEW_TOKENS" --repeats 2 \
  --cooldown-seconds "$COOLDOWN_SECONDS" --cooldown-max-temp-c "$COOLDOWN_MAX_TEMP_C" \
  --time-budget-minutes 90 --resume \
  --out "$OUT/e3.2-h6-plan-robustness-${tag}.json" \
  > "$LOG_DIR/4_e3.2_robustness.log" 2>&1 || step_status=1
log "step 4/6 exit=$? (log: $LOG_DIR/4_e3.2_robustness.log)"

log "--- step 5/6: Figure 7 / Table 2 -- headline TTFT comparison ---"
$PY -u scripts/run_paper_comparison.py \
  --model "$MODEL" --draft-model "$DRAFT_MODEL" --store "$STORE" \
  --methods airllm,accelerate,deepspeed-zero-inference,exact-min,exact-resident,spec-fixed \
  --prompt-suite evaluation --token-lengths 1 --blocks "$E6_BLOCKS" \
  --warmup-tokens 8 --cooldown-seconds "$COOLDOWN_SECONDS" --cooldown-max-temp-c "$COOLDOWN_MAX_TEMP_C" \
  --time-budget-minutes-per-length 150 \
  --out-dir "$OUT/e6.3-${tag}" --run-label "overnight-ttft" \
  --resume --require-complete --require-thermally-clean \
  > "$LOG_DIR/5_e6.3_ttft.log" 2>&1 || step_status=1
log "step 5/6 exit=$? (log: $LOG_DIR/5_e6.3_ttft.log)"

log "=== campaign complete for $MODEL on $HW_TAG ==="
log "outputs: $OUT/*-${tag}.json and $OUT/e6.3-${tag}/"
log "logs: $LOG_DIR"
if [[ $step_status -ne 0 ]]; then
  log "one or more steps above reported a nonzero exit -- check the named log" \
      "before treating this campaign as complete; per-step failure does not" \
      "stop later steps, by design, so partial output can still exist."
fi

# Resuming after an interruption: rerun this exact script with the same
# environment variables. Steps A-D and the smoke gate skip themselves once
# their output file exists; steps 1, 3, 4, and 5 all pass --resume, which is
# a no-op on a first run (nothing to resume) and picks up an interrupted
# .partial correctly on a rerun -- confirmed safe on both paths by reading
# each script's own resume guard before relying on it here, not assumed.
# Do not delete or hand-edit a .partial file to "fix" a stuck resume; that
# breaks this project's evidence-integrity convention (see
# docs/RESULTS_LOG.md) -- remove it only if you intend to restart that one
# step from zero.
exit "$step_status"
