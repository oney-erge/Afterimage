#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate
python -u scripts/run_probe_real.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-depths 2 --ranks 4,8,16 \
    --workloads focused_code --skip-closed-loop \
    --out /tmp/diag.json
echo "SCRIPT_EXIT_CODE:$?"
