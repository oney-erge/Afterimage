#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate
python -u scripts/validate_streaming.py 2>&1 | grep -vE "Loading|Fetching|Warning|it/s]"
echo "EXIT:$?"
