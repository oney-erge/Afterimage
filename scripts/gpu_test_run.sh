#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate
python -u -m pytest tests/test_gpu_decode.py -v 2>&1
echo "SCRIPT_EXIT_CODE:$?"
