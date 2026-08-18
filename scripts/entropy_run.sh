#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate
MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
for DT in bf16 fp16; do
    python -u scripts/run_entropy_audit.py --model "$MODEL" --dtype "$DT"
    echo ""
done
echo "SCRIPT_EXIT_CODE:$?"
