import sys

from huggingface_hub import snapshot_download

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
path = snapshot_download(model_id)
print("DOWNLOADED_TO:", path)
