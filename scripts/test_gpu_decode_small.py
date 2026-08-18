import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from afterimage.runtime.gpu_decode import decode_gpu
from afterimage.runtime.huffman_chunked import decode_chunked_cpu_reference, encode_chunked

torch.manual_seed(0)
exponents = torch.randint(100, 140, (2000,))
enc = encode_chunked(exponents, chunk_size=64, max_bits=16)

print(f"n_chunks={enc.n_chunks} chunk_size={enc.chunk_size} max_bits={enc.max_bits}")

cpu_ref = decode_chunked_cpu_reference(enc)
print("CPU reference: OK, shape", cpu_ref.shape)

assert np.array_equal(cpu_ref, exponents.numpy()), "CPU reference itself is wrong!"
print("CPU reference matches original: PASS")

gpu_out = decode_gpu(enc)
gpu_np = gpu_out.cpu().numpy()

print("GPU output shape:", gpu_np.shape)
match = np.array_equal(gpu_np, exponents.numpy())
print(f"GPU matches original: {match}")

if not match:
    diff = np.where(gpu_np != exponents.numpy())[0]
    print(f"first mismatches at indices: {diff[:20]}")
    print(f"expected: {exponents.numpy()[diff[:20]]}")
    print(f"got:      {gpu_np[diff[:20]]}")
    print(f"total mismatches: {len(diff)} / {len(exponents)}")
else:
    print("SUCCESS: bit-exact GPU decode")
