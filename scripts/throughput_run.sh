#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate
python -u scripts/run_decode_throughput.py 2>&1
echo "SCRIPT_EXIT_CODE:$?"
