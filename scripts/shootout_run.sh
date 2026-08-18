#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate
python -u scripts/run_shootout.py --model "${1:-Qwen/Qwen2.5-1.5B-Instruct}"
echo "SCRIPT_EXIT_CODE:$?"
