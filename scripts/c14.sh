#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" && source ~/.venv/bin/activate && python -u scripts/compress_14b.py 2>&1 | grep -vE "Loading|Fetching|it/s\]|Warning"
echo "EXIT:$?"
