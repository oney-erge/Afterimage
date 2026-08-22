"""docs/archive/PROPOSAL.md's own H2 gate (unrelated to the current H2
hazard-cost stopping hypothesis): measure real CPU decode throughput on the
actual compressed 14B store before writing a single line of dispatcher
code. If aggregate throughput is below ~1 GB/s of decoded weight output,
H2 is dead and this prints that plainly -- no engine change is justified.
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from afterimage.runtime.binstore import BinaryWeightReader
from afterimage.runtime.cpu_decode import _HAS_NUMBA, decode_chunks_numba, decode_chunks_threaded
from afterimage.runtime.huffman_chunked import ChunkedEncoded, decode_chunked_cpu_reference

STORE = "/root/afterimage/store_14b"


def main() -> int:
    store = pathlib.Path(STORE)
    manifest = json.loads((store / "manifest.json").read_text())

    # A real, mid-sized decoder-layer tensor -- representative of the ~440
    # tensors that dominate per-token traffic, not the two outlier-sized ones.
    key = "model.layers.10.mlp.down_proj.weight"
    meta = manifest["tensors"][key]
    assert meta["compressed"], key

    with BinaryWeightReader(store / "weights.bin") as reader:
        arrays = {name: reader.read(ref) for name, ref in meta["blobs"].items()}

    enc = ChunkedEncoded(
        packed=arrays["packed"], chunk_offsets=arrays["chunk_offsets"],
        chunk_nbytes=arrays["chunk_nbytes"], sym_lut=arrays["sym_lut"],
        len_lut=arrays["len_lut"], max_bits=int(meta["max_bits"]),
        chunk_size=int(meta["chunk_size"]), n_symbols=int(meta["n_symbols"]),
        shape=tuple(meta["shape"]))

    n_weights = enc.n_symbols
    output_bytes = n_weights  # exponent decodes to 1 byte/weight
    print("tensor: %s  shape=%s  weights=%d (%.1f MB decoded output)"
          % (key, meta["shape"], n_weights, output_bytes / 1e6), flush=True)
    print("chunks=%d  chunk_size=%d  max_bits=%d  compressed_bytes=%d"
          % (enc.n_chunks, enc.chunk_size, enc.max_bits, enc.packed.nbytes), flush=True)
    print("", flush=True)

    import os
    n_cores = os.cpu_count() or 4
    print("cpu_count() = %d" % n_cores, flush=True)
    print("", flush=True)

    print("=== numpy fancy-indexing path (decode_chunks_threaded) ===", flush=True)
    print("%-10s %12s %14s %10s" % ("threads", "wall (s)", "GB/s output", "speedup"))
    baseline = None
    for n_threads in [1, 2, 4, 8, min(16, n_cores), n_cores]:
        if n_threads < 1:
            continue
        t0 = time.perf_counter()
        out = decode_chunks_threaded(enc, n_threads)
        wall = time.perf_counter() - t0
        gbps = output_bytes / 1e9 / wall
        if baseline is None:
            baseline = wall
        print("%-10d %12.4f %14.3f %9.2fx" % (n_threads, wall, gbps, baseline / wall))
        del out

    if not _HAS_NUMBA:
        print("\nnumba not installed -- skipping compiled path")
        return 0

    print("\n=== numba-JIT compiled path (decode_chunks_numba, prange) ===", flush=True)
    # warm up: first call pays JIT compilation cost, not representative of
    # steady-state throughput.
    print("compiling...", flush=True)
    t0 = time.perf_counter()
    warm = decode_chunks_numba(enc, 0, min(64, enc.n_chunks))
    compile_s = time.perf_counter() - t0
    print("compile + first-call time: %.2fs" % compile_s, flush=True)

    ref = decode_chunked_cpu_reference(enc)
    got = decode_chunks_numba(enc)[: enc.n_symbols]
    print("bit-exact vs reference: %s" % bool(np.array_equal(got, ref)), flush=True)

    import numba as _nb
    orig_threads = _nb.get_num_threads()
    for n_threads in [1, 2, 4, 8, min(16, n_cores), n_cores]:
        _nb.set_num_threads(max(1, n_threads))
        t0 = time.perf_counter()
        out = decode_chunks_numba(enc)
        wall = time.perf_counter() - t0
        gbps = output_bytes / 1e9 / wall
        print("%-10d %12.4f %14.3f" % (n_threads, wall, gbps))
        del out
    _nb.set_num_threads(orig_threads)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
