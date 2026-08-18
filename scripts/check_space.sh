#!/usr/bin/env bash
echo "=== WSL disk ==="; df -h ~ | tail -1
echo "=== RAM ==="; free -g | awk 'NR==2{print "total="$2"GB avail="$7"GB"}'
echo "=== VRAM ==="; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "=== HF cache size ==="; du -sh ~/.cache/huggingface 2>/dev/null || echo "none"
