import shutil
import subprocess

try:
    import triton
    print("triton", triton.__version__)
except ImportError as e:
    print("triton: NOT AVAILABLE -", e)

nvcc = shutil.which("nvcc")
print("nvcc:", nvcc or "NOT FOUND")

import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device capability:", torch.cuda.get_device_capability())
    print("device name:", torch.cuda.get_device_name(0))

try:
    from torch.utils.cpp_extension import CUDA_HOME
    print("CUDA_HOME:", CUDA_HOME)
except Exception as e:
    print("CUDA_HOME check failed:", e)
