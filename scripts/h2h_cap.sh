#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" && source ~/.venv/bin/activate
stdbuf -oL -eL python -u scripts/run_headtohead.py --n-tokens 2 --skip-airllm --vram-cap-gb 6.0 --empty-cache-every 1 2>&1 | stdbuf -oL grep -vE "Loading|Fetching|it/s]|Warning"
echo "EXIT:$?"
