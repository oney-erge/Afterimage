#!/usr/bin/env bash
# Storage-path diagnosis for docs/EXECUTION_PLAN.md Stage A.
#
# Determines whether this machine can produce trustworthy cold-read NVMe
# numbers, and by which mechanism: drop_caches, O_DIRECT, or neither.
# O_DIRECT (dd iflag=direct) bypasses the page cache at the syscall level and
# is what databases use; if it works, it is strictly more reliable than
# drop_caches because it cannot be defeated by a cache layer we do not
# control (e.g. Windows caching the WSL VHDX underneath Linux).
set -u

SIZE_MB="${1:-2048}"
BS="${2:-1M}"

test_path() {
    local label="$1" dir="$2"
    echo ""
    echo "### $label  ($dir)"
    if ! mkdir -p "$dir" 2>/dev/null; then echo "  unavailable"; return; fi

    local f="$dir/sttest.bin"
    dd if=/dev/zero of="$f" bs="$BS" count="$SIZE_MB" status=none conv=fsync 2>/dev/null
    sync

    # 1. buffered read after drop_caches
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
    local r1
    r1=$(dd if="$f" of=/dev/null bs="$BS" 2>&1 | tail -1)
    echo "  buffered, after drop_caches : $r1"

    # 2. buffered read again (should be page-cache warm)
    local r2
    r2=$(dd if="$f" of=/dev/null bs="$BS" 2>&1 | tail -1)
    echo "  buffered, warm              : $r2"

    # 3. O_DIRECT read -- bypasses page cache entirely
    local r3
    r3=$(dd if="$f" of=/dev/null bs="$BS" iflag=direct 2>&1 | tail -1)
    if echo "$r3" | grep -qi "invalid argument"; then
        echo "  O_DIRECT                    : NOT SUPPORTED on this filesystem"
    else
        echo "  O_DIRECT                    : $r3"
    fi

    rm -f "$f"
}

echo "=================================================================="
echo "Storage path diagnosis  (${SIZE_MB} MB, bs=${BS})"
echo "=================================================================="
echo "Interpretation:"
echo "  If O_DIRECT is much faster than buffered-after-drop_caches, the"
echo "  buffered path is being served by a cache Linux does not control."
echo "  In that case the runtime MUST use O_DIRECT for weight reads and"
echo "  drop_caches alone must NOT be trusted to certify NVMe numbers."

test_path "WSL native ext4 (VHDX)" "$HOME/afterimage_probe"
test_path "DrvFs /mnt/d (Samsung 980 PRO via Windows)" "/mnt/d/afterimage_probe"
test_path "DrvFs /mnt/c (Intel NVMe via Windows)" "/mnt/c/afterimage_probe"

echo ""
echo "=================================================================="
echo "Underlying block devices:"
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT 2>/dev/null | head -20
echo "=================================================================="
