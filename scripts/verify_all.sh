#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate
echo "=== GPU CORRECTNESS (v1 + v2) ==="
python -u -m pytest tests/test_gpu_decode.py tests/test_gpu_decode_v2.py -q 2>&1 | tail -6
echo ""
echo "=== RE-MEASURED SIZE BREAKDOWN ==="
python -u scripts/run_airllm_comparison.py --n-layers 8 2>&1 | grep -vE "Loading|Fetching|Warning" | head -22
