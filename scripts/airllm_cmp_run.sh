#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate
python -u scripts/run_airllm_comparison.py --n-layers 8 2>&1 | grep -vE "Loading weights|Fetching|Warning:"
