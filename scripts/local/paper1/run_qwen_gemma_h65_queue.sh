#!/usr/bin/env bash
# Sequential, crash-safe RTX 3080 Paper 1 queue: finish Qwen, then run Gemma.
set -uo pipefail

REPO="/mnt/c/for fun/Afterimage"
PYTHON="/root/.venv/bin/python"
ROOT="$REPO/scripts/local/paper1/output/paper-h65-qwen-gemma-queue-20260902"
STATUS_FILE="$ROOT/queue-status.txt"
QWEN_FINAL="$REPO/scripts/local/paper1/output/paper-h65-practical-20260901/decode-64tok/qwen3-14b-h65-matched-decode64-practical-20260901-64tok.json"

mkdir -p "$ROOT"
cd "$REPO"

artifact_is_eligible() {
    local path=$1
    [[ -s "$path" ]] || return 1
    "$PYTHON" - "$path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
raise SystemExit(0 if result.get("paper_eligible") is True else 1)
PY
}

finish() {
    local status=$?
    printf 'queue_finished_at=%s exit_status=%s\n' "$(date -Is)" "$status" \
        >> "$STATUS_FILE"
}
trap finish EXIT

printf 'queue_started_at=%s commit=%s\n' "$(date -Is)" "$(git rev-parse HEAD)" \
    | tee -a "$STATUS_FILE"

if ! artifact_is_eligible "$QWEN_FINAL"; then
    printf '[%s] START qwen_decode64_resume\n' "$(date -Is)" \
        | tee -a "$STATUS_FILE"
    bash scripts/local/paper1/run_h65_practical_decode_extension.sh
    qwen_status=$?
    printf '[%s] END qwen_decode64_resume exit_status=%s\n' \
        "$(date -Is)" "$qwen_status" | tee -a "$STATUS_FILE"
fi

if artifact_is_eligible "$QWEN_FINAL"; then
    printf '[%s] Qwen 64-token artifact complete and eligible\n' "$(date -Is)" \
        | tee -a "$STATUS_FILE"
else
    printf '[%s] Qwen remains incomplete; Gemma will run and Qwen will retry afterward\n' \
        "$(date -Is)" | tee -a "$STATUS_FILE"
fi

printf '[%s] START gemma_h65_practical\n' "$(date -Is)" \
    | tee -a "$STATUS_FILE"
bash scripts/local/paper1/run_full_h65_gemma_practical.sh
gemma_status=$?
printf '[%s] END gemma_h65_practical exit_status=%s\n' \
    "$(date -Is)" "$gemma_status" | tee -a "$STATUS_FILE"

if ! artifact_is_eligible "$QWEN_FINAL"; then
    printf '[%s] RETRY qwen_decode64_resume\n' "$(date -Is)" \
        | tee -a "$STATUS_FILE"
    bash scripts/local/paper1/run_h65_practical_decode_extension.sh
fi

printf 'queue_stages_finished_at=%s\n' "$(date -Is)" | tee -a "$STATUS_FILE"
