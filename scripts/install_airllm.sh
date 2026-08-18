#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate
pip install airllm 2>&1 | tail -8
echo "=== import check ==="
python -c "import airllm; print('airllm', getattr(airllm,'__version__','?'))" 2>&1 | tail -5
