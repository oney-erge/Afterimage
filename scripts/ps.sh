ps aux | grep -E "run_headtohead" | grep -v grep | head -3
echo "--- GPU util ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- python CPU time ---"
ps -o pid,etime,time,cmd -C python 2>/dev/null | head -5
