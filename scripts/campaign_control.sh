#!/usr/bin/env bash
# Pause/resume control for a run_paper_comparison.py campaign, WSL2/Linux
# only (matches the rest of this project's bench tooling).
#
# Why a whole-process-group kill is the right (and safe) way to pause
# --------------------------------------------------------------------
# run_paper_comparison.py checkpoints its .json.partial file after every
# single (block, method) cell -- see checkpoint(partial, result) in
# run_one_token_length(). That means killing the campaign at any point
# loses at most the one cell that was in flight, never previously
# completed work, and --resume already knows how to pick back up (it
# matches the invocation's own settings and skips every (block, method)
# pair already recorded). "Pause" is therefore just "stop the process
# tree cleanly"; there is no separate pause/serialize mechanism to build.
#
# The orchestrator (run_paper_comparison.py), the worker subprocess it
# spawns per cell, and the bash wrapper that chains multiple invocations
# all share one process group (confirmed via `ps -o pid,pgid,ppid`), so
# `kill -TERM -<pgid>` reaches all three in one shot -- killing only the
# orchestrator's own PID would strand the in-flight worker still holding
# GPU memory.
#
# Usage:
#   scripts/campaign_control.sh status
#   scripts/campaign_control.sh pause
#   scripts/campaign_control.sh resume <log-dir> <out-dir> <store> <run-label-run1> <run-label-run2>
set -euo pipefail

find_pgid() {
    # The orchestrator's own argv is distinctive enough to match on
    # without depending on a fixed PID across pause/resume cycles.
    pgrep -f 'run_paper_comparison\.py --store' | while read -r pid; do
        ps -o pgid= -p "$pid" | tr -d ' '
    done | sort -u | head -n1
}

cmd="${1:-status}"

case "$cmd" in
    status)
        pgid="$(find_pgid || true)"
        if [ -z "${pgid:-}" ]; then
            echo "no run_paper_comparison.py campaign is currently running"
            exit 0
        fi
        echo "campaign running under process group $pgid:"
        ps -o pid,pgid,ppid,etime,stat,cmd -g "$pgid"
        ;;
    pause)
        pgid="$(find_pgid || true)"
        if [ -z "${pgid:-}" ]; then
            echo "nothing to pause: no run_paper_comparison.py campaign is running"
            exit 0
        fi
        echo "pausing campaign: sending SIGTERM to process group $pgid"
        kill -TERM "-$pgid"
        sleep 2
        if kill -0 "-$pgid" 2>/dev/null; then
            echo "still alive after SIGTERM, escalating to SIGKILL"
            kill -KILL "-$pgid" 2>/dev/null || true
        fi
        echo "paused. Last completed cell for each in-progress length is preserved in its"
        echo ".json.partial file. Resume with:  scripts/campaign_control.sh resume ..."
        ;;
    resume)
        log_dir="${2:?usage: campaign_control.sh resume <log-dir> <out-dir> <store> <run-label-run1> <run-label-run2>}"
        out_dir="${3:?missing out-dir}"
        store="${4:?missing store path}"
        label1="${5:?missing run-label for run 1 (paper_generation)}"
        label2="${6:?missing run-label for run 2 (evaluation)}"
        methods="airllm,accelerate,dfloat11,exact-min,exact-resident,spec-fixed"
        mkdir -p "$log_dir"
        echo "resuming run 1 ($label1, paper_generation, lengths 1/32/128) -> $log_dir/run1_paper_generation.log"
        /root/.venv/bin/python3 -u scripts/run_paper_comparison.py \
            --store "$store" --methods "$methods" \
            --prompt-suite paper_generation --token-lengths 1,32,128 --blocks 3 \
            --warmup-tokens 8 --cooldown-seconds 20 --cooldown-max-temp-c 75 \
            --time-budget-minutes-per-length 240 --out-dir "$out_dir" \
            --run-label "$label1" --resume \
            >> "$log_dir/run1_paper_generation.log" 2>&1
        run1_exit=$?
        echo "RUN1_RESUME_EXIT=$run1_exit" >> "$log_dir/run1_paper_generation.log"

        echo "resuming run 2 ($label2, evaluation, length 4) -> $log_dir/run2_short_cold_start.log"
        /root/.venv/bin/python3 -u scripts/run_paper_comparison.py \
            --store "$store" --methods "$methods" \
            --prompt-suite evaluation --token-lengths 4 --blocks 3 \
            --warmup-tokens 8 --cooldown-seconds 20 --cooldown-max-temp-c 75 \
            --time-budget-minutes-per-length 240 --out-dir "$out_dir" \
            --run-label "$label2" --resume \
            >> "$log_dir/run2_short_cold_start.log" 2>&1
        run2_exit=$?
        echo "RUN2_RESUME_EXIT=$run2_exit" >> "$log_dir/run2_short_cold_start.log"
        echo "CAMPAIGN_RESUME_DONE run1=$run1_exit run2=$run2_exit"
        ;;
    *)
        echo "usage: $0 {status|pause|resume ...}" >&2
        exit 2
        ;;
esac
