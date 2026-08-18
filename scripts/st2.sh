ps aux | grep run_headtohead | grep -v grep | wc -l | xargs echo "processes running:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
