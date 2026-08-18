pkill -f run_headtohead && echo "killed old run"
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader
