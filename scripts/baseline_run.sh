#!/usr/bin/env bash
# Fills baseline rows for one model across dtypes (VALIDATION_PLAN.md #2).
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
shift 2>/dev/null

for DT in fp16 fp32; do
    echo "==================================================================="
    echo "MODEL=$MODEL  DTYPE=$DT"
    echo "==================================================================="
    python -u scripts/run_baseline_table.py --model "$MODEL" --dtype "$DT" "$@"
    echo ""
done
echo "SCRIPT_EXIT_CODE:$?"
