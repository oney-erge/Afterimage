import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, torch
from afterimage.runtime.huffman_chunked import encode_chunked

torch.manual_seed(0)
W = (torch.randn(8960, 1536) * 0.02).to(torch.bfloat16)
bits = W.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
exp = ((bits >> 7) & 0xFF).flatten()

enc = encode_chunked(exp, chunk_size=1024, max_bits=16)
nb = enc.chunk_nbytes
actual = int(nb.sum())
padded = int(enc.packed.shape[0])
print(f"chunks={enc.n_chunks}  mean={nb.mean():.1f}B  max={nb.max()}B  min={nb.min()}B")
print(f"actual bytes needed : {actual/1e6:.2f} MB")
print(f"padded bytes stored : {padded/1e6:.2f} MB")
print(f"PADDING WASTE       : {(padded-actual)/actual*100:.1f}%")
print(f"bits/weight actual  : {actual*8/exp.numel():.3f}")
print(f"bits/weight padded  : {padded*8/exp.numel():.3f}")
