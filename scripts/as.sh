ps aux | grep airllm_only | grep -v grep | awk "{print \$11, \$12, \$13}" | head -2 || echo "NOT RUNNING"
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- airllm splits dir ---"
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B/splitted_model* 2>/dev/null || echo "no split dir yet"
