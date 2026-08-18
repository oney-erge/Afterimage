#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" && source ~/.venv/bin/activate && python -u scripts/_layer_breakdown.py 2>&1 | grep -vE "Loading|Fetching|Warning"
