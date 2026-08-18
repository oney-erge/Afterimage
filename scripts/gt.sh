#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" && source ~/.venv/bin/activate
python -u -m pytest tests/test_gpu_decode_v2.py tests/test_gpu_decode.py -q 2>&1 | tail -5
