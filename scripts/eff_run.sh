#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate
python -u scripts/run_coding_efficiency.py --n-layers 12 2>&1 | grep -vE "Loading weights|Fetching|Warning:"
