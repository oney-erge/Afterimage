#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" && source ~/.venv/bin/activate
stdbuf -oL python -u scripts/airllm_only.py 2>&1 | stdbuf -oL grep -vE "it/s]|Loading|Fetching"
echo "EXIT:$?"
