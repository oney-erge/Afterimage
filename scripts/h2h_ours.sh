#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" && source ~/.venv/bin/activate
python -u scripts/run_headtohead.py --n-tokens 3 --skip-airllm 2>&1 | grep -vE "Loading|Fetching|it/s\]|Warning"
echo "EXIT:$?"
