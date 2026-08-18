import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, torch, torch.nn as nn
from transformers import AutoModelForCausalLM
from afterimage.runtime.huffman_chunked import encode_chunked
from afterimage.probe.entropy import analyze_tensor

m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.bfloat16)
lin = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, nn.Linear)]
sizes = sorted(set(mod.weight.numel() for _, mod in lin))
print(f"{len(lin)} layers, distinct sizes: {[f'{s/1e6:.2f}M' for s in sizes]}")
print()
print(f"{'size':>10} {'n':>4} {'entropy':>8} {'packed b/w':>11} {'pad%':>6} {'LUT MB':>7} {'LUTbits/w':>10} {'ratio':>7}")

seen = {}
for name, mod in lin:
    n = mod.weight.numel()
    if n in seen: continue
    seen[n] = True
    W = mod.weight.data
    rep = analyze_tensor(W)
    bits = W.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exp = ((bits >> 7) & 0xFF).flatten()
    enc = encode_chunked(exp, chunk_size=1024, max_bits=16)
    actual = int(enc.chunk_nbytes.sum()); padded = int(enc.packed.shape[0])
    lut = enc.sym_lut.nbytes + enc.len_lut.nbytes
    packed_bw = padded*8/n
    lut_bw = lut*8/n
    total_bw = 8 + packed_bw + lut_bw
    cnt = sum(1 for _, mm in lin if mm.weight.numel() == n)
    print(f"{n/1e6:>9.2f}M {cnt:>4} {rep.exponent_entropy:>8.3f} {packed_bw:>11.3f} "
          f"{(padded-actual)/actual*100:>5.1f}% {lut/1e6:>7.2f} {lut_bw:>10.3f} {16/total_bw:>6.3f}x")
