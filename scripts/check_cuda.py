import torch

print("torch", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"vram free: {free/1e9:.2f} GB / total: {total/1e9:.2f} GB")
