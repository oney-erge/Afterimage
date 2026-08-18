"""Lossless layer-streaming inference -- AirLLM's algorithm, but the bytes
crossing the bus are entropy-coded (LOSSLESS_ENGINE.md Phase B).

AirLLM's cost model: read the FULL uncompressed model from disk once per
generated token. It is bit-exact (it changes nothing about the weights) and
it is slow for exactly one reason -- I/O volume. This engine keeps the
algorithm identical and attacks only that: same layer-at-a-time execution,
same bit-exact output, fewer bytes read per token.

Two-stage design, because a 29.5 GB model cannot be held in this machine's
19 GB of RAM at any point:

  1. `compress_model_to_disk` streams the safetensors shards lazily with
     safe_open, compresses tensors in parallel across a bounded worker pool
     (docs/IMPROVEMENT_PLAN.md lever 4), and writes them into one flat
     `weights.bin` (lever 1) instead of per-tensor `.npz` files.
  2. `StreamingLosslessModel` builds the network on the meta device (no
     storage at all), then materializes exactly one decoder layer's weights
     at a time on the GPU, runs it, and frees it -- prefetching the next
     layer's bytes off disk in the background while the current one computes
     (lever 3), and row-gathering the embedding table instead of fully
     materializing it when the model does not tie it to lm_head (lever 2).

`generate_speculative` (lever 5) is a separate decoding mode built on top of
the same weight-streaming machinery; see its docstring for why it is
validated differently from `generate_greedy`.

Quantization is opt-in (see config.EngineConfig). The default is strictly
lossless, because AirLLM does not quantize either -- quantizing by default
would make the head-to-head measure two different things.
"""
from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import pathlib
import threading
import time

import numpy as np
import torch

from .binstore import BinaryWeightReader, BinaryWeightWriter, blobref_to_dict
from .compressed_store import CompressedLayer, compress_layer, decompress_layer_gpu


@dataclasses.dataclass
class StreamStats:
    bytes_read: int = 0
    layer_loads: int = 0
    decode_seconds: float = 0.0
    io_seconds: float = 0.0
    compute_seconds: float = 0.0
    spec_sweeps: int = 0
    spec_accepted_tokens: int = 0

    def reset(self) -> None:
        self.bytes_read = 0
        self.layer_loads = 0
        self.decode_seconds = 0.0
        self.io_seconds = 0.0
        self.compute_seconds = 0.0
        self.spec_sweeps = 0
        self.spec_accepted_tokens = 0


# -- offline compression ---------------------------------------------------

_BIG_TENSOR_SUFFIXES = ("embed_tokens.weight", "lm_head.weight")


def _compress_one_tensor(task: tuple) -> dict:
    """Runs in a worker process (or the main process for the "big" tensors
    -- see compress_model_to_disk). Returns plain numpy arrays, not
    BlobRefs: offsets depend on write order into the single shared
    weights.bin, which only the main process (the sole writer) can assign.
    """
    shard_path, key, chunk_size, quantize, row_gather = task
    from safetensors import safe_open

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        W = f.get_tensor(key)
    orig = W.numel() * W.element_size()

    if row_gather:
        # Stored raw (no entropy coding) and row-addressable: the whole
        # point is to skip ever materializing this tensor in full, so
        # compressing it would only add decode cost to a path designed to
        # avoid touching most of it at all (lever 2).
        assert W.dtype == torch.bfloat16 and W.dim() == 2, (
            "row-gather storage assumes a 2D bf16 embedding table, got "
            "%s %s for %s" % (W.dtype, tuple(W.shape), key))
        raw16 = W.contiguous().view(torch.int16).numpy()
        return {
            "key": key, "kind": "row_gather", "shape": list(W.shape),
            "hidden_size": int(W.shape[1]), "orig_bytes": orig,
            "comp_bytes": orig, "arrays": {"raw": raw16},
        }

    use_compression = (W.dtype == torch.bfloat16 and W.dim() == 2
                       and W.numel() > 65536)
    if use_compression:
        if quantize == "q8":
            from ..probe.approximations import quantize_grouped
            W = quantize_grouped(W, bits=8, group_size=64).to(torch.bfloat16)
        layer = compress_layer(W, chunk_size=chunk_size)
        return {
            "key": key, "kind": "compressed", "shape": list(W.shape),
            "max_bits": int(layer.encoded.max_bits),
            "n_symbols": int(layer.encoded.n_symbols),
            "chunk_size": int(layer.encoded.chunk_size),
            "orig_bytes": orig, "comp_bytes": layer.compressed_bytes,
            "arrays": {
                "sign_mantissa": layer.sign_mantissa.numpy(),
                "packed": layer.encoded.packed,
                "chunk_offsets": layer.encoded.chunk_offsets,
                "chunk_nbytes": layer.encoded.chunk_nbytes,
                "sym_lut": layer.encoded.sym_lut,
                "len_lut": layer.encoded.len_lut,
            },
        }

    # small / non-2D / non-bf16 tensors: store raw. The LUT fixed cost would
    # exceed any gain (see test_compressed_store.py's amortization-boundary
    # test).
    return {
        "key": key, "kind": "raw", "shape": list(W.shape),
        "dtype": str(W.dtype), "orig_bytes": orig, "comp_bytes": orig,
        "arrays": {"raw": W.to(torch.float32).numpy()},
    }


def compress_model_to_disk(model_id: str, out_dir, chunk_size: int = 1024,
                           quantize=None, progress_every: int = 50,
                           max_workers: int | None = None) -> dict:
    """Offline pass: safetensors -> one compressed weights.bin + manifest.

    Parallel across tensors (lever 4), except embed_tokens/lm_head-sized
    ones, which run serially in the main process. Those two are ~9x bigger
    than every other tensor in a 14B-class model, and compress_layer's
    working set for a tensor that size is roughly 4x its bf16 bytes (the
    exponent/sign/mantissa fields are unpacked to int32 before being
    repacked) -- two of them running concurrently in a worker pool could
    exceed this machine's 19 GB RAM ceiling. There are only ever one or two
    such tensors per model, so serializing just them costs seconds, not the
    minutes parallelism saves on the other ~440.

    CALLER REQUIREMENT: the worker pool uses the "spawn" start method (see
    the note above the Pool call for why fork is unsafe here), which
    re-imports the launching script as __main__ in every worker. Any script
    that calls this function directly (not through pytest, which is already
    spawn-safe) MUST put its top-level driver code behind
    `if __name__ == "__main__":` -- otherwise each worker re-executes the
    whole script, which recursively re-invokes this function. This bit
    scripts/validate_streaming.py and scripts/compress_14b.py the first
    time this was wired up; both now carry the guard.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snap = pathlib.Path(snapshot_download(model_id))
    shards = sorted(snap.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError("no .safetensors in " + str(snap))

    cfg = AutoConfig.from_pretrained(model_id)
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    # Row-gather only helps when embed_tokens is NOT also serving as
    # lm_head: lm_head needs the full output-projection matrix regardless,
    # so a tied model gains nothing from skipping embed_tokens's
    # materialization -- it would just have to load the identical bytes
    # under a different name a moment later.
    row_gather_key = None if tied else "model.embed_tokens.weight"

    all_tasks = []
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tasks.append((str(shard), key, chunk_size, quantize,
                                  key == row_gather_key))

    big_tasks = [t for t in all_tasks if t[1].endswith(_BIG_TENSOR_SUFFIXES)]
    small_tasks = [t for t in all_tasks if not t[1].endswith(_BIG_TENSOR_SUFFIXES)]

    if max_workers is None:
        max_workers = min(8, mp.cpu_count() or 4)

    manifest = {"model_id": model_id, "quantize": quantize,
                "chunk_size": chunk_size, "tied": tied, "tensors": {}}
    total_orig = 0
    total_comp = 0
    n = 0
    n_total = len(all_tasks)

    def _write_result(res: dict, writer: BinaryWeightWriter) -> None:
        nonlocal total_orig, total_comp, n
        blobs = {name: blobref_to_dict(writer.write(arr))
                 for name, arr in res["arrays"].items()}
        entry = {"orig_bytes": res["orig_bytes"], "comp_bytes": res["comp_bytes"],
                 "shape": res["shape"], "blobs": blobs}
        if res["kind"] == "compressed":
            entry.update(compressed=True, max_bits=res["max_bits"],
                        n_symbols=res["n_symbols"], chunk_size=res["chunk_size"])
        elif res["kind"] == "row_gather":
            entry.update(compressed=False, row_gather=True,
                        hidden_size=res["hidden_size"], dtype="bfloat16")
        else:
            entry.update(compressed=False, dtype=res["dtype"])
        manifest["tensors"][res["key"]] = entry

        total_orig += res["orig_bytes"]
        total_comp += res["comp_bytes"]
        n += 1
        if n % progress_every == 0:
            ratio = total_orig / max(total_comp, 1)
            print("  [%d/%d] %.2f GB -> %.2f GB (%.3fx)"
                  % (n, n_total, total_orig / 1e9, total_comp / 1e9, ratio), flush=True)

    bin_path = out_dir / "weights.bin"
    with BinaryWeightWriter(bin_path) as writer:
        for t in big_tasks:
            _write_result(_compress_one_tensor(t), writer)

        if small_tasks:
            # "spawn", not "fork": forking a process that has already
            # touched torch (its CPU intra-op thread pool is OpenMP-backed)
            # inherits a thread pool with only the forking thread alive --
            # a well-known hazard that hung every worker at 0% CPU rather
            # than crashing, which is worse than a crash. spawn re-imports
            # each worker fresh, which costs a one-time startup per worker,
            # not per tensor, and is negligible next to the ~40 minutes this
            # lever exists to cut down.
            # imap, NOT imap_unordered: computation still parallelizes fully
            # across workers, but results are written to weights.bin in the
            # SAME order small_tasks was submitted in -- the safetensors
            # shards' own key order, which for a standard HF checkpoint is
            # close to model structure order (layer 0's tensors together,
            # then layer 1's, ...). imap_unordered writes in whichever
            # order workers happen to finish, which is uncorrelated with
            # layer order; StreamingLosslessModel then reads layers 0..N-1
            # sequentially on every token, so an unordered store turns that
            # into scattered random access across a 20 GB file instead of
            # mostly-sequential reads. Measured cost on the real 14B store:
            # imap_unordered produced a store that streamed at roughly half
            # the old .npz format's throughput (io_seconds alone exceeded
            # the OLD format's combined io+decode time) -- exactly what
            # random access over sequential access predicts, and enough to
            # erase this lever's entire gain and then some.
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=max_workers) as pool:
                for res in pool.imap(_compress_one_tensor, small_tasks, chunksize=1):
                    _write_result(res, writer)

    manifest["total_orig_bytes"] = total_orig
    manifest["total_comp_bytes"] = total_comp
    manifest["ratio"] = total_orig / max(total_comp, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class StreamingLosslessModel:
    """Layer-at-a-time execution from the compressed store."""

    def __init__(self, model_id: str, store_dir, device: str = "cuda",
                 vram_cap_gb=None, empty_cache_every: int = 0,
                 progress: bool = False, prefetch: bool = True):
        """vram_cap_gb: hard ceiling on this process's GPU memory.

        Layer streaming only needs one layer resident at a time, so peak
        LIVE memory is small -- but PyTorch's caching allocator never
        returns freed blocks to the driver, so observed VRAM climbs to the
        high-water mark and keeps growing until it looks like the model
        barely fits (measured: 7.8 GB of 8 GB on a 14B, against ~3.9 GB
        actually live). Capping makes the allocator reuse its own cache
        instead of requesting more, and turns a silent creep toward OOM into
        an immediate, legible error.

        empty_cache_every: release cached blocks back to the driver every N
        layer frees. 0 disables. Non-zero costs a synchronize per call, so
        it is a memory/throughput tradeoff, not a free win.

        prefetch: overlap the NEXT layer's disk I/O with the CURRENT
        layer's GPU compute (lever 3), using a background thread with its
        own file handle so it never races the main thread's reads. Decode
        (the GPU Huffman kernel) still runs synchronously when a layer is
        loaded -- this overlaps I/O with compute, not decode with compute --
        so treat it as a partial implementation of "double-buffer
        everything," not the full three-way overlap.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        self.store = pathlib.Path(store_dir)
        self.manifest = json.loads((self.store / "manifest.json").read_text())
        self.device = device
        self.stats = StreamStats()

        self.vram_cap_gb = vram_cap_gb
        self.empty_cache_every = empty_cache_every
        self.progress = progress
        self.prefetch = prefetch
        self._frees = 0
        self._tok_i = 0
        self._tok_total = 0
        self._gen_t0 = 0.0

        self._reader = BinaryWeightReader(self.store / "weights.bin")
        self._prefetch_reader = (BinaryWeightReader(self.store / "weights.bin")
                                 if prefetch else None)
        self._prefetch_cache: dict[int, dict] = {}
        self._prefetch_threads: dict[int, threading.Thread] = {}
        self._prefetch_lock = threading.Lock()

        if vram_cap_gb is not None and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            frac = min(1.0, (vram_cap_gb * 1e9) / total)
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            print("VRAM capped to %.2f GB (%.1f%% of %.2f GB device)"
                  % (vram_cap_gb, frac * 100, total / 1e9), flush=True)

        cfg = AutoConfig.from_pretrained(model_id)
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
        self.model.eval()

        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)

        self._materialize_resident()
        self._install_hooks()
        if self.prefetch:
            self._start_prefetch(0)

    def close(self) -> None:
        self._reader.close()
        if self._prefetch_reader is not None:
            self._prefetch_reader.close()

    # -- store access ----------------------------------------------------

    def _read_tensor_arrays(self, key: str) -> tuple[dict, float]:
        meta = self.manifest["tensors"][key]
        t0 = time.perf_counter()
        arrays = {name: self._reader.read(ref) for name, ref in meta["blobs"].items()}
        return arrays, time.perf_counter() - t0

    def _decode_tensor(self, key: str, arrays: dict) -> torch.Tensor:
        meta = self.manifest["tensors"][key]

        if not meta["compressed"]:
            if meta.get("row_gather"):
                raise RuntimeError(
                    "%s is stored row-gather-only and should never be "
                    "loaded whole -- that defeats the reason it is stored "
                    "that way" % key)
            out = torch.from_numpy(arrays["raw"])
            dt = meta.get("dtype", "bfloat16")
            if "bfloat16" in dt:
                out = out.to(torch.bfloat16)
            elif "float16" in dt:
                out = out.to(torch.float16)
            return out.to(self.device)

        t1 = time.perf_counter()
        from .huffman_chunked import ChunkedEncoded
        enc = ChunkedEncoded(
            packed=arrays["packed"],
            chunk_offsets=arrays["chunk_offsets"],
            chunk_nbytes=arrays["chunk_nbytes"],
            sym_lut=arrays["sym_lut"],
            len_lut=arrays["len_lut"],
            max_bits=int(meta["max_bits"]),
            chunk_size=int(meta["chunk_size"]),
            n_symbols=int(meta["n_symbols"]),
            shape=tuple(meta["shape"]),
        )
        layer = CompressedLayer(
            sign_mantissa=torch.from_numpy(arrays["sign_mantissa"]),
            encoded=enc,
            shape=tuple(meta["shape"]),
        )
        out = decompress_layer_gpu(layer, device=self.device)
        self.stats.decode_seconds += time.perf_counter() - t1
        return out

    def _load_tensor(self, key: str) -> torch.Tensor:
        """Synchronous read-then-decode -- the path used whenever a
        prefetch did not already do the read (resident materialization at
        startup, and any layer tensor the background thread has not
        finished reading yet)."""
        arrays, io_s = self._read_tensor_arrays(key)
        self.stats.bytes_read += self.manifest["tensors"][key]["comp_bytes"]
        self.stats.io_seconds += io_s
        return self._decode_tensor(key, arrays)

    def _assign(self, root, dotted: str, value: torch.Tensor) -> None:
        target = root
        parts = dotted.split(".")
        for p in parts[:-1]:
            target = getattr(target, p)
        getattr(target, parts[-1]).data.copy_(value)

    # -- residency -------------------------------------------------------

    def _install_embed_row_gather(self, key: str) -> None:
        """For untied models, embed_tokens.weight is stored raw and
        row-addressable rather than fully materialized (lever 2): a token
        embedding lookup only ever needs one row per input token, not all
        151936 of them, so pulling the whole 1.56 GB table into VRAM for
        that is pure waste. lm_head still needs its full matrix to produce
        logits over the whole vocabulary and stays resident as before --
        this only applies where the two are NOT tied together, since a tied
        model has to materialize the full table anyway to serve as lm_head.
        """
        meta = self.manifest["tensors"][key]
        ref = meta["blobs"]["raw"]
        hidden = meta["hidden_size"]
        row_nbytes = hidden * 2  # bf16: 2 bytes/element, row-major contiguous
        base_offset = ref["offset"]
        device = self.device
        stats = self.stats
        reader = self._reader

        def forward(input_ids: torch.Tensor) -> torch.Tensor:
            t0 = time.perf_counter()
            flat = input_ids.reshape(-1).detach().cpu().numpy()
            rows = np.stack([
                reader.read_row(base_offset, int(i), row_nbytes, "int16")
                for i in flat
            ])
            t = torch.from_numpy(rows).view(torch.bfloat16)
            t = t.to(device).view(*input_ids.shape, hidden)
            stats.bytes_read += rows.nbytes
            stats.io_seconds += time.perf_counter() - t0
            return t

        self.model.model.embed_tokens.forward = forward

    def _materialize_resident(self) -> None:
        """Materialize everything that is not a decoder layer, then re-tie
        weights the checkpoint does not store separately.

        Weight tying is the subtle part. Qwen2.5 (and many others) set
        tie_word_embeddings=True, so the safetensors contains NO
        lm_head.weight -- it is meant to alias embed_tokens.weight. Calling
        to_empty() on each module independently allocates a FRESH tensor per
        parameter, silently breaking that aliasing, and since there is no
        lm_head.weight in the store there is then nothing to load into it:
        lm_head runs on uninitialized memory. The model still produces
        logits, so nothing crashes -- it just returns confident nonsense
        (measured: max abs logit diff 22.1, argmax token 100628 vs the
        correct 12095). Re-tying after materialization is what fixes it, and
        any parameter left with no source now raises instead of quietly
        running on garbage.
        """
        embed_key = "model.embed_tokens.weight"
        row_gather = bool(self.manifest["tensors"].get(embed_key, {}).get("row_gather"))
        embed_mod = self.model.model.embed_tokens if row_gather else None

        missing = []
        for name, mod in self.model.named_modules():
            if name.startswith("model.layers."):
                continue
            if row_gather and mod is embed_mod:
                continue  # handled by _install_embed_row_gather instead
            params = list(mod.named_parameters(recurse=False))
            # Non-persistent buffers (rotary inv_freq and friends) are
            # COMPUTED from config, never stored in the checkpoint, so they
            # must be regenerated rather than looked up -- see the rebuild
            # below. Including them here would flag a condition that is
            # correct by design.
            nonpersistent = getattr(mod, "_non_persistent_buffers_set", set())
            buffers = [(bn, b) for bn, b in mod.named_buffers(recurse=False)
                       if bn not in nonpersistent]
            if not params and not buffers:
                continue
            mod.to_empty(device=self.device)
            for pname, _ in params + buffers:
                key = (name + "." + pname) if name else pname
                if key in self.manifest["tensors"]:
                    getattr(mod, pname).data.copy_(self._load_tensor(key))
                else:
                    missing.append(key)

        # Rebuild modules whose state is computed, not loaded. to_empty()
        # left their non-persistent buffers as uninitialized memory; a fresh
        # instance recomputes them from config on the target device.
        m = self.model.model
        if getattr(m, "rotary_emb", None) is not None:
            try:
                m.rotary_emb = type(m.rotary_emb)(config=self.model.config,
                                                  device=self.device)
            except TypeError:
                m.rotary_emb = type(m.rotary_emb)(self.model.config).to(self.device)

        tied = getattr(self.model.config, "tie_word_embeddings", False)
        if tied and "lm_head.weight" in missing:
            self.model.lm_head.weight = self.model.model.embed_tokens.weight
            missing.remove("lm_head.weight")
        if row_gather and embed_key in missing:
            missing.remove(embed_key)

        if missing:
            raise RuntimeError(
                "resident parameters with no source in the store and no "
                "tying rule to cover them: " + ", ".join(missing[:8]))

        if row_gather:
            self._install_embed_row_gather(embed_key)

        self.stats.reset()  # resident load is startup cost, not per-token

    def _report_progress(self, layer_idx: int) -> None:
        """Live progress. Layer streaming is slow by construction (the whole
        model crosses the bus per token), so a run with no output is
        indistinguishable from a hang -- which is exactly what happened
        before this existed. ETA comes from layers completed so far rather
        than a fixed guess, since per-layer cost varies with layer size.

        One plain line every few layers, not a cursor-rewound bar: this
        output is captured to a log file, where rewind characters smear
        everything onto a single unreadable line.
        """
        done = self._tok_i * self.n_layers + layer_idx + 1
        total = max(1, self._tok_total * self.n_layers)
        elapsed = time.perf_counter() - self._gen_t0
        rate = done / max(elapsed, 1e-6)
        eta = (total - done) / max(rate, 1e-9)
        pct = 100.0 * done / total
        filled = int(pct / 4)
        bar = "#" * filled + "-" * (25 - filled)
        if layer_idx % 5 == 0 or layer_idx == self.n_layers - 1:
            print("  [%s] %5.1f%%  tok %d/%d  layer %2d/%d  %.1f GB  "
                  "%.0fs elapsed  ETA %.0fs"
                  % (bar, pct, self._tok_i + 1, self._tok_total,
                     layer_idx + 1, self.n_layers,
                     self.stats.bytes_read / 1e9, elapsed, eta), flush=True)

    def _read_layer_tensor_arrays(self, idx: int, reader: BinaryWeightReader) -> dict:
        layer = self.layers[idx]
        result = {}
        for pname, _ in layer.named_parameters():
            key = "model.layers.%d.%s" % (idx, pname)
            if key not in self.manifest["tensors"]:
                continue
            meta = self.manifest["tensors"][key]
            t0 = time.perf_counter()
            arrays = {name: reader.read(ref) for name, ref in meta["blobs"].items()}
            result[key] = (arrays, time.perf_counter() - t0)
        return result

    def _start_prefetch(self, idx: int) -> None:
        """Kick off a background read of layer idx's bytes. Uses a SEPARATE
        BinaryWeightReader (its own file handle) from the main thread's
        reader -- seek()+read() on one shared handle is not thread-safe, and
        giving the background thread its own fd sidesteps that race
        entirely rather than adding locking around every read.

        The Thread object itself is kept in _prefetch_threads so _load_layer
        can JOIN it instead of racing it: an earlier version only tracked
        "pending" as a set membership flag, and _load_layer fell back to an
        entirely separate synchronous read (its own reader, same file) the
        instant the cache wasn't ready yet -- meaning two readers ended up
        pulling the SAME bytes off the SAME disk at the SAME time whenever
        prefetch didn't win the race. Measured on the real 14B store, that
        made streaming SLOWER with prefetch on than off (76.9s vs 51.3s for
        2 tokens) -- contention, not overlap. Joining the in-flight thread
        instead means the worst case degrades to "wait for the one read
        already happening," not "start a second one that competes with it."
        """
        if idx >= self.n_layers:
            return

        def worker():
            result = self._read_layer_tensor_arrays(idx, self._prefetch_reader)
            with self._prefetch_lock:
                self._prefetch_cache[idx] = result

        with self._prefetch_lock:
            if idx in self._prefetch_cache or idx in self._prefetch_threads:
                return
            th = threading.Thread(target=worker, daemon=True)
            self._prefetch_threads[idx] = th
        th.start()

    def _load_layer(self, idx: int) -> None:
        layer = self.layers[idx]
        layer.to_empty(device=self.device)

        cached = None
        if self.prefetch:
            with self._prefetch_lock:
                th = self._prefetch_threads.pop(idx, None)
            if th is not None:
                th.join()
            with self._prefetch_lock:
                cached = self._prefetch_cache.pop(idx, None)
            # Fire idx+1's read NOW, before decoding idx's bytes, not after
            # this whole method returns. Per-layer GPU compute measured at
            # ~19 ms (1.5s / 80 layer-loads on the real 14B) -- starting the
            # next prefetch only after compute leaves it almost nothing to
            # hide behind, which is why the first version of this overlap
            # barely beat no-prefetch at all (50.9s vs 51.3s for 2 tokens).
            # Decode is the actually-sizable stage (measured ~19-53s per
            # 2-token run): overlapping the NEXT read with THIS layer's
            # decode-plus-compute gives the background thread a window an
            # order of magnitude larger to hide the read inside.
            self._start_prefetch(idx + 1)

        for pname, _ in layer.named_parameters():
            key = "model.layers.%d.%s" % (idx, pname)
            if key not in self.manifest["tensors"]:
                continue
            if cached is not None and key in cached:
                arrays, io_s = cached[key]
                self.stats.io_seconds += io_s
                self.stats.bytes_read += self.manifest["tensors"][key]["comp_bytes"]
                out = self._decode_tensor(key, arrays)
            else:
                out = self._load_tensor(key)
            self._assign(layer, pname, out)

        self.stats.layer_loads += 1
        if self.progress:
            self._report_progress(idx)

    def _free_layer(self, idx: int) -> None:
        self.layers[idx].to("meta")
        self._frees += 1
        if self.empty_cache_every and self._frees % self.empty_cache_every == 0:
            torch.cuda.empty_cache()

    # -- execution -------------------------------------------------------

    def _install_hooks(self) -> None:
        """Load a layer's weights just before it runs, free them right after.

        Hooks rather than a hand-written layer loop. The first version of
        this engine drove the decoder stack manually and produced WRONG
        logits (max abs diff 22.1, different argmax token) because that loop
        silently omitted the causal attention mask -- every position
        attended to every other, so the model was reading the future. Rotary
        embeddings, mask construction, cache handling and layer signatures
        are all version-specific transformers internals; letting the
        library's own forward drive them and confining this engine to weight
        residency is both correct and far less brittle.
        """
        def make_pre(idx):
            def pre(module, args, kwargs):
                self._load_layer(idx)
                return None
            return pre

        def make_post(idx):
            def post(module, args, kwargs, output):
                torch.cuda.synchronize()
                self._free_layer(idx)
                return output
            return post

        for i, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(make_pre(i), with_kwargs=True)
            layer.register_forward_hook(make_post(i), with_kwargs=True)

    @torch.no_grad()
    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """One full forward pass. transformers drives the stack; the hooks
        stream each layer's weights in and out around it.

        Deliberately reused by generate_speculative for chain verification:
        a causal LM computes logits at every position of an arbitrary-length
        input in one sweep by construction, so verifying a whole draft chain
        is just calling this once on [seq, draft_tokens] and slicing the
        result -- no separate batched-verification path is needed.
        """
        t0 = time.perf_counter()
        out = self.model(input_ids=input_ids, use_cache=False)
        self.stats.compute_seconds += time.perf_counter() - t0
        return out.logits

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        seq = input_ids
        self._tok_total = max_new_tokens
        self._gen_t0 = time.perf_counter()
        for t in range(max_new_tokens):
            self._tok_i = t
            logits = self.forward_logits(seq)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
        return seq

    @torch.no_grad()
    def generate_speculative(self, input_ids: torch.Tensor, max_new_tokens: int,
                             draft_model, k: int = 8, temperature: float = 1.0,
                             generator: torch.Generator | None = None) -> torch.Tensor:
        """Draft/verify speculative decoding (lever 5).

        draft_model is a small, FULLY resident HF model sharing this
        model's tokenizer/vocabulary (see load_draft_model). It proposes k
        tokens autoregressively -- fast, since nothing but its own small
        weights ever crosses the bus. The target then verifies the WHOLE
        chain in one call to forward_logits (see that method's docstring),
        and speculative_sample_step (runtime/verify.py, tested for exact
        distributional correctness regardless of draft quality) turns the
        per-position draft/target distributions into an accepted prefix
        plus one correction or bonus token. One target weight-sweep can
        therefore yield several accepted tokens instead of one, at
        unchanged bus traffic per sweep.

        WHAT "LOSSLESS" MEANS HERE, and why this is validated differently
        from generate_greedy: this samples from the target model's EXACT
        distribution at the given temperature. It does not, and cannot,
        reproduce generate_greedy's argmax sequence -- sampling and greedy
        decoding are different decoding modes by definition, independent of
        this implementation. Use generate_greedy where byte-identical-to-
        greedy output is the requirement (e.g. the AirLLM head-to-head);
        use this where throughput matters more and exact-distribution
        sampling is what "lossless" needs to mean. Correctness for this
        path means matching the target's distribution, which is what
        tests/test_verify.py checks statistically, not token-for-token
        equality against any other run.
        """
        from .verify import sample_categorical, speculative_sample_step

        seq = input_ids
        self._tok_total = max_new_tokens
        self._gen_t0 = time.perf_counter()
        n_generated = 0

        while n_generated < max_new_tokens:
            step_k = min(k, max_new_tokens - n_generated)

            draft_seq = seq
            draft_tokens: list[int] = []
            draft_probs: list[torch.Tensor] = []
            for _ in range(step_k):
                dlogits = draft_model(input_ids=draft_seq).logits[:, -1, :]
                probs = torch.softmax(dlogits[0] / temperature, dim=-1)
                tok = sample_categorical(probs, generator)
                draft_tokens.append(tok)
                draft_probs.append(probs)
                draft_seq = torch.cat(
                    [draft_seq, torch.tensor([[tok]], device=draft_seq.device)], dim=1)

            chain = torch.cat(
                [seq, torch.tensor([draft_tokens], device=seq.device)], dim=1)
            target_logits = self.forward_logits(chain)
            base = seq.shape[1] - 1
            target_probs = [
                torch.softmax(target_logits[0, base + i, :] / temperature, dim=-1)
                for i in range(step_k)
            ]
            bonus_probs = torch.softmax(
                target_logits[0, base + step_k, :] / temperature, dim=-1)

            accepted, n_from_draft = speculative_sample_step(
                draft_probs, target_probs, draft_tokens, bonus_probs, generator)

            seq = torch.cat([seq, torch.tensor([accepted], device=seq.device)], dim=1)
            n_generated += len(accepted)
            self._tok_i = min(n_generated, max_new_tokens)
            self.stats.spec_sweeps += 1
            self.stats.spec_accepted_tokens += n_from_draft

        return seq[:, :input_ids.shape[1] + max_new_tokens]


def load_draft_model(model_id: str = "Qwen/Qwen3-0.6B", device: str = "cuda"):
    """A small, fully resident model for generate_speculative's draft step.

    Must share the target's tokenizer/vocabulary for speculative_sample_step
    to compare valid distributions -- Qwen3-0.6B shares Qwen3-14B's
    tokenizer and 151936-token vocabulary because both are the same model
    family. It is loaded the ordinary way (no streaming): it is small
    enough (~1.2 GB bf16) to live resident, and the entire point of it is to
    be fast, which streaming would defeat.
    """
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    m.to(device)
    m.eval()
    return m
