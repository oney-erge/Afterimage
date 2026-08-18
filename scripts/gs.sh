#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" && source ~/.venv/bin/activate
python -u -m pytest tests/test_sliced_decompress.py tests/test_gpu_decode_v2.py tests/test_compressed_store.py -q 2>&1 | tail -6
