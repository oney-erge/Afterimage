import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from afterimage.runtime.gpu_decode_v2 import decode_gpu_v2
from afterimage.runtime.huffman_chunked import encode_chunked

torch.manual_seed(0)
exponents = torch.randint(100, 140, (20000,))
enc = encode_chunked(exponents, chunk_size=64, max_bits=16)
print(f"n_chunks={enc.n_chunks} chunk_size={enc.chunk_size} max_bits={enc.max_bits}")

for block_chunks in [1, 4, 32, 128]:
    gpu_out = decode_gpu_v2(enc, block_chunks=block_chunks).cpu().numpy()
    match = np.array_equal(gpu_out, exponents.numpy())
    print(f"block_chunks={block_chunks:>4}: match={match}")
    if not match:
        diff = np.where(gpu_out != exponents.numpy())[0]
        print(f"  first mismatches at {diff[:10]}, total {len(diff)}/{len(exponents)}")
