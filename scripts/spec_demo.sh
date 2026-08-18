#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" && source ~/.venv/bin/activate
stdbuf -oL -eL python -u scripts/run_speculative_demo.py --n-tokens 20 --k 8 2>&1 | stdbuf -oL grep -vE "Loading|Fetching|it/s]|Warning"
echo "EXIT:$?"
