#!/usr/bin/env bash
cd "/mnt/c/for fun/Afterimage" || exit 1
source ~/.venv/bin/activate
python -u scripts/run_probe_real.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-depths 6 \
    --ranks 4,8,16,32,64,128,256 \
    --closed-loop-ranks 8,32,128 \
    --workloads focused_code,multi_turn_chat,long_form_prose,adversarial_topic_switch \
    --out ~/afterimage/results/phase0_real.json
echo "SCRIPT_EXIT_CODE:$?"
