import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, torch
from afterimage.runtime.compressed_store import CompressedLayer, decompress_layer_gpu
from afterimage.runtime.huffman_chunked import ChunkedEncoded

STORE = pathlib.Path("/root/afterimage/store_1.5b")
man = json.loads((STORE/"manifest.json").read_text())
key = "model.layers.0.mlp.down_proj.weight"
meta = man["tensors"][key]
path = STORE / (key.replace("/","__") + ".npz")
print("tensor:", key, "shape", meta["shape"], "comp_bytes", meta["comp_bytes"])
print("file on disk: %.1f MB" % (path.stat().st_size/1e6))

N=5
# 1. np.load + materialize arrays
t=time.perf_counter()
for _ in range(N):
    d = np.load(path)
    arrs = {k: d[k] for k in d.files}
t_load = (time.perf_counter()-t)/N
print("np.load + materialize : %.3f s  -> %.1f MB/s" % (t_load, path.stat().st_size/1e6/t_load))

d = np.load(path); arrs = {k: d[k] for k in d.files}
enc = ChunkedEncoded(packed=arrs["packed"], chunk_offsets=arrs["chunk_offsets"],
    chunk_nbytes=arrs["chunk_nbytes"], sym_lut=arrs["sym_lut"], len_lut=arrs["len_lut"],
    max_bits=int(meta["max_bits"]), chunk_size=int(meta["chunk_size"]),
    n_symbols=int(meta["n_symbols"]), shape=tuple(meta["shape"]))
layer = CompressedLayer(sign_mantissa=torch.from_numpy(arrs["sign_mantissa"]),
                        encoded=enc, shape=tuple(meta["shape"]))

# 2. GPU decode only (arrays already in RAM)
decompress_layer_gpu(layer)  # warmup
torch.cuda.synchronize()
t=time.perf_counter()
for _ in range(N):
    W = decompress_layer_gpu(layer)
torch.cuda.synchronize()
t_dec = (time.perf_counter()-t)/N
nbytes_out = int(np.prod(meta["shape"]))*2
print("GPU decode (in RAM)   : %.3f s  -> %.2f GB/s bf16 out" % (t_dec, nbytes_out/t_dec/1e9))
print()
print("VERDICT: load is %.0f%% of total, decode is %.0f%%" % (
    100*t_load/(t_load+t_dec), 100*t_dec/(t_load+t_dec)))
