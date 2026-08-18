#!/usr/bin/env bash
# Verifies the benchmarking preconditions from docs/EXECUTION_PLAN.md Stage A.
# Run inside WSL2:  bash scripts/verify_rig.sh
set -u

PROBE_DIR="${1:-/mnt/d/afterimage_probe}"
SIZE_MB="${2:-512}"

echo "=============================================="
echo "Afterimage rig verification"
echo "=============================================="

echo "--- kernel ---"
uname -r

echo "--- GPU passthrough ---"
if [ -e /dev/dxg ]; then echo "/dev/dxg present: YES"; else echo "/dev/dxg present: NO"; fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
else
    echo "nvidia-smi: NOT FOUND"
fi

echo "--- torch ---"
python3 - <<'PY' 2>&1 | sed 's/^/    /'
try:
    import torch
    print("torch", torch.__version__, "cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        free, total = torch.cuda.mem_get_info()
        print(f"vram free: {free/1e9:.2f} GB / total: {total/1e9:.2f} GB")
except ImportError:
    print("torch NOT INSTALLED in WSL -- see EXECUTION_PLAN.md A.2")
PY

echo "--- memory cap (EXECUTION_PLAN.md A.3) ---"
free -g | awk 'NR<=2'
TOTAL_GB=$(free -g | awk 'NR==2{print $2}')
if [ "$TOTAL_GB" -le 16 ]; then
    echo "RAM cap OK for the NVMe research config (<=16 GB)"
else
    echo "WARNING: ${TOTAL_GB} GB visible. A 16 GB model CAN be fully page-cached."
    echo "         Set memory=14GB in C:\\Users\\oneye\\.wslconfig and 'wsl --shutdown'."
fi

echo "--- sudo ---"
if sudo -n true 2>/dev/null; then echo "passwordless sudo: YES"; else echo "passwordless sudo: NO"; fi

echo "--- page cache drop (EXECUTION_PLAN.md A.4) ---"
mkdir -p "$PROBE_DIR" || { echo "cannot create $PROBE_DIR"; exit 1; }
TESTFILE="$PROBE_DIR/cachetest.bin"
dd if=/dev/zero of="$TESTFILE" bs=1M count="$SIZE_MB" status=none
sync

sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null \
    && echo "drop_caches write: OK" || echo "drop_caches write: FAILED"

COLD_START=$(date +%s%N)
cat "$TESTFILE" > /dev/null
COLD_END=$(date +%s%N)

WARM_START=$(date +%s%N)
cat "$TESTFILE" > /dev/null
WARM_END=$(date +%s%N)

COLD_MS=$(( (COLD_END - COLD_START) / 1000000 ))
WARM_MS=$(( (WARM_END - WARM_START) / 1000000 ))

echo "cold read: ${COLD_MS} ms   warm read: ${WARM_MS} ms   (${SIZE_MB} MB)"

if [ "$WARM_MS" -gt 0 ]; then
    RATIO=$(( COLD_MS * 100 / WARM_MS ))
    echo "cold/warm ratio: ${RATIO}%"
    if [ "$RATIO" -ge 200 ]; then
        echo "VERDICT: cache drop is EFFECTIVE -- NVMe numbers can be certified"
    else
        echo "VERDICT: cache drop INEFFECTIVE -- do NOT report NVMe numbers."
        echo "         Fall back to the RAM cap as the primary control."
    fi
fi

if [ "$COLD_MS" -gt 0 ]; then
    echo "apparent cold read bandwidth: $(( SIZE_MB * 1000 / COLD_MS )) MB/s"
fi

rm -f "$TESTFILE"
echo "=============================================="
