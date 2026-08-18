ps aux | grep run_headtohead | grep -v grep | head -2 || echo "NOT RUNNING"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
