#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate
python -u scripts/download_model.py Qwen/Qwen3-14B
echo "DL_EXIT:$?"
