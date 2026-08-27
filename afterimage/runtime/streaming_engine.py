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
     safe_open, compresses tensors in parallel across a bounded worker pool,
     and writes them into one flat `weights.bin` instead of per-tensor
     `.npz` files, with a schema version and a per-blob CRC32 so a truncated
     or corrupted store is caught explicitly rather than silently served.
  2. `StreamingLosslessModel` builds the network on the meta device (no
     storage at all), then decides -- via EngineConfig and
     vram_planner.plan_tiers -- which tensors live permanently in VRAM,
     which live in pinned host RAM and get memcpy'd to GPU every token, and
     which are read + decoded from disk every token. Disk-tier layers are
     prefetched some configurable number ahead while the current layer
     decodes and computes, and an untied model's embedding table is
     row-gathered rather than ever fully materialized.

`generate_speculative` is a separate decoding mode built on top of the same
weight-streaming machinery; see its docstring for why it is validated
differently from `generate_greedy`.

Quantization is opt-in (see config.EngineConfig). The default is strictly
lossless, because AirLLM does not quantize either -- quantizing by default
would make the head-to-head measure two different things.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import multiprocessing as mp
import pathlib
import sys
import threading
import time
import warnings

import numpy as np
import torch

from .binstore import BinaryWeightReader, BinaryWeightWriter, blobref_to_dict
from .adapters import classify_config, resolve_model_adapter
from .compressed_store import CompressedLayer, compress_layer, decompress_layer_gpu
from .config import EngineConfig
from .controllers import PrefetchObservation, build_prefetch_controller
from .critical_path import CriticalPathProfile, TraceRecorder

# Bumped whenever the manifest/weights.bin layout changes in a way that
# makes an old store unreadable by new code (or vice versa) -- e.g. the
# .npz -> binstore switch would have been version 1 -> 2 had this existed
# at the time. A store built before this field existed has no
# "schema_version" key at all, which StreamingLosslessModel treats as
# version 1 and refuses, rather than failing deep inside tensor decoding
# with a confusing KeyError on "blobs".
CURRENT_SCHEMA_VERSION = 2


@dataclasses.dataclass
class StreamStats:
    bytes_read: int = 0
    layer_loads: int = 0
    decode_seconds: float = 0.0
    io_seconds: float = 0.0
    compute_seconds: float = 0.0
    spec_sweeps: int = 0
    spec_accepted_tokens: int = 0
    spec_cache_crops: int = 0
    spec_cached_prefix_tokens: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    prefetch_wait_seconds: float = 0.0
    prefetch_peak_inflight_bytes: int = 0
    storage_read_calls: int = 0
    storage_extent_bytes: int = 0
    mips_certified: int = 0
    mips_fallbacks: int = 0
    mips_rows_evaluated: int = 0
    mips_rows_pruned: int = 0
    mips_index_build_seconds: float = 0.0

    def reset(self) -> None:
        self.bytes_read = 0
        self.layer_loads = 0
        self.decode_seconds = 0.0
        self.io_seconds = 0.0
        self.compute_seconds = 0.0
        self.spec_sweeps = 0
        self.spec_accepted_tokens = 0
        self.spec_cache_crops = 0
        self.spec_cached_prefix_tokens = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.prefetch_wait_seconds = 0.0
        self.prefetch_peak_inflight_bytes = 0
        self.storage_read_calls = 0
        self.storage_extent_bytes = 0
        self.mips_certified = 0
        self.mips_fallbacks = 0
        self.mips_rows_evaluated = 0
        self.mips_rows_pruned = 0
        # mips_index_build_seconds is startup cost, deliberately retained
        # when callers reset steady-state generation counters.


# -- offline compression ---------------------------------------------------

_BIG_TENSOR_SUFFIXES = ("embed_tokens.weight", "lm_head.weight")


def _transformers_weight_shards(snapshot: pathlib.Path) -> list[pathlib.Path]:
    """Return exactly the safetensors files named by Transformers metadata.

    Some model repositories publish both ``model-*.safetensors`` and a second
    ``consolidated.safetensors`` export for another runtime.  Glob-reading the
    directory would silently process both complete copies.  Prefer the
    Transformers index when present; the single-file convention remains the
    fallback for small checkpoints and local test fixtures.
    """
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        filenames = sorted(set(index.get("weight_map", {}).values()))
        if not filenames:
            raise ValueError("empty weight_map in %s" % index_path)
        shards = [snapshot / filename for filename in filenames]
        missing = [str(path) for path in shards if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Transformers weight index names missing shards: %s" % missing)
        return shards

    single = snapshot / "model.safetensors"
    if single.exists():
        return [single]

    return sorted(
        path for path in snapshot.glob("*.safetensors")
        if not path.name.startswith("consolidated"))


def _compress_one_tensor(task: tuple) -> dict:
    """Runs in a worker process (or the main process for the "big" tensors
    -- see compress_model_to_disk). Returns plain numpy arrays, not
    BlobRefs: offsets depend on write order into the single shared
    weights.bin, which only the main process (the sole writer) can assign.
    """
    if len(task) == 6:
        shard_path, key, chunk_size, quantize, max_bits, row_gather = task
        expert_index = None
    else:
        shard_path, key, chunk_size, quantize, max_bits, row_gather, expert_index = task
    from safetensors import safe_open

    parent_key = key
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        if expert_index is None:
            W = f.get_tensor(key)
        else:
            W = f.get_slice(key)[int(expert_index)].contiguous()
            key = "%s.__expert__.%d" % (key, expert_index)
    orig = W.numel() * W.element_size()

    if row_gather:
        # Stored raw (no entropy coding) and row-addressable: the whole
        # point is to skip ever materializing this tensor in full, so
        # compressing it would only add decode cost to a path designed to
        # avoid touching most of it at all.
        assert W.dtype == torch.bfloat16 and W.dim() == 2, (
            "row-gather storage assumes a 2D bf16 embedding table, got "
            "%s %s for %s" % (W.dtype, tuple(W.shape), key))
        raw16 = W.contiguous().view(torch.int16).numpy()
        return {
            "key": key, "kind": "row_gather", "shape": list(W.shape),
            "parent_key": parent_key if expert_index is not None else None,
            "hidden_size": int(W.shape[1]), "orig_bytes": orig,
            "comp_bytes": orig, "arrays": {"raw": raw16},
        }

    use_compression = (W.dtype == torch.bfloat16 and W.dim() == 2
                       and W.numel() > 65536)
    if use_compression:
        if quantize == "q8":
            from ..probe.approximations import quantize_grouped
            W = quantize_grouped(W, bits=8, group_size=64).to(torch.bfloat16)
        layer = compress_layer(W, chunk_size=chunk_size, max_bits=max_bits)
        return {
            "key": key, "kind": "compressed", "shape": list(W.shape),
            "parent_key": parent_key if expert_index is not None else None,
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
        "parent_key": parent_key if expert_index is not None else None,
        "dtype": str(W.dtype), "orig_bytes": orig, "comp_bytes": orig,
        "arrays": {"raw": W.to(torch.float32).numpy()},
    }


def compress_model_to_disk(model_id: str, out_dir, config: EngineConfig | None = None,
                           progress_every: int = 50,
                           max_workers: int | None = None,
                           control=None, source_dir=None,
                           revision: str | None = None) -> dict:
    """Offline pass: safetensors -> one compressed weights.bin + manifest.

    Parallel across tensors, except embed_tokens/lm_head-sized ones, which
    run serially in the main process. Those two are ~9x bigger than every
    other tensor in a 14B-class model, and compress_layer's working set for
    a tensor that size is roughly 4x its bf16 bytes (the exponent/sign/
    mantissa fields are unpacked to int32 before being repacked) -- two of
    them running concurrently in a worker pool could exceed this machine's
    19 GB RAM ceiling. There are only ever one or two such tensors per
    model, so serializing just them costs seconds, not the minutes
    parallelism saves on the other ~440.

    CALLER REQUIREMENT: the worker pool uses the "spawn" start method (see
    the note above the Pool call for why fork is unsafe here), which
    re-imports the launching script as __main__ in every worker. Any script
    that calls this function directly (not through pytest, which is already
    spawn-safe) MUST put its top-level driver code behind
    `if __name__ == "__main__":` -- otherwise each worker re-executes the
    whole script, which recursively re-invokes this function.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig

    cfg = config or EngineConfig()

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ``consolidated*.safetensors`` is commonly a duplicate export for a
    # non-Transformers runtime.  Avoid downloading it and independently
    # verify the remaining set against model.safetensors.index.json.
    snap = pathlib.Path(source_dir) if source_dir is not None else pathlib.Path(
        snapshot_download(
            model_id, revision=revision,
            ignore_patterns=["consolidated*.safetensors"],
        )
    )
    shards = _transformers_weight_shards(snap)
    if not shards:
        raise FileNotFoundError("no .safetensors in " + str(snap))

    hf_cfg = AutoConfig.from_pretrained(snap, trust_remote_code=False)
    tied = bool(getattr(hf_cfg, "tie_word_embeddings", False))
    # Row-gather only helps when embed_tokens is NOT also serving as
    # lm_head: lm_head needs the full output-projection matrix regardless,
    # so a tied model gains nothing from skipping embed_tokens's
    # materialization -- it would just have to load the identical bytes
    # under a different name a moment later.
    layout = classify_config(hf_cfg)
    embedding_key = (
        "model.language_model.embed_tokens.weight"
        if layout["modality"] == "vision-text"
        else "model.embed_tokens.weight"
    )
    row_gather_key = None if tied else embedding_key

    all_tasks = []
    expert_slices: dict[str, list[str]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                shape = tuple(f.get_slice(key).get_shape())
                packed_experts = (
                    len(shape) == 3
                    and key.endswith((".experts.gate_up_proj", ".experts.down_proj"))
                )
                if packed_experts:
                    expert_slices[key] = [
                        "%s.__expert__.%d" % (key, index)
                        for index in range(shape[0])
                    ]
                    all_tasks.extend(
                        (str(shard), key, cfg.chunk_size, cfg.quantize,
                         cfg.max_bits, False, index)
                        for index in range(shape[0])
                    )
                else:
                    all_tasks.append((str(shard), key, cfg.chunk_size, cfg.quantize,
                                      cfg.max_bits, key == row_gather_key, None))

    big_tasks = [t for t in all_tasks if t[1].endswith(_BIG_TENSOR_SUFFIXES)]
    small_tasks = [t for t in all_tasks if not t[1].endswith(_BIG_TENSOR_SUFFIXES)]

    if max_workers is None:
        max_workers = min(8, mp.cpu_count() or 4)

    from .control import JobControl
    ctl = control or JobControl()

    manifest = {"schema_version": CURRENT_SCHEMA_VERSION, "model_id": model_id,
                "revision": revision,
                "quantize": cfg.quantize, "chunk_size": cfg.chunk_size,
                "tied": tied, "tensors": {}, "expert_slices": expert_slices}
    total_orig = 0
    total_comp = 0
    n = 0
    n_total = len(all_tasks)

    def _write_result(res: dict, writer: BinaryWeightWriter) -> None:
        nonlocal total_orig, total_comp, n
        ctl.checkpoint()  # pause/cancel boundary: one tensor at a time
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
        ratio = total_orig / max(total_comp, 1)
        ctl.report(phase="compress", n=n, n_total=n_total,
                  orig_gb=total_orig / 1e9, comp_gb=total_comp / 1e9, ratio=ratio)
        if n % progress_every == 0:
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
            # the old .npz format's throughput -- exactly what random
            # access over sequential access predicts, and enough to erase
            # this lever's entire gain and then some.
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
    """Layer-at-a-time execution from the compressed store, with per-tensor
    residency decided by EngineConfig + vram_planner rather than hardcoded.
    """

    def __init__(self, model_id: str, store_dir, device: str = "cuda",
                 config: EngineConfig | None = None, control=None,
                 source_dir=None):
        """config: an EngineConfig. Its defaults reproduce this engine's
        original fixed policy exactly (embed_tokens/lm_head/norms
        permanently VRAM-resident, every decoder-layer weight streamed from
        disk every token, io_prefetch_depth=1) -- setting vram_budget_gb
        hands residency to vram_planner.plan_tiers instead, and additionally
        setting ram_budget_gb adds a pinned-host-RAM tier that decoder-layer
        weights can land in instead of disk.

        vram_cap_gb (on the config) is a hard ceiling on this process's GPU
        memory, independent of the residency PLANNING above: PyTorch's
        caching allocator never returns freed blocks to the driver, so
        observed VRAM climbs to the high-water mark and keeps growing until
        it looks like the model barely fits. Capping makes the allocator
        reuse its own cache instead of requesting more, and turns a silent
        creep toward OOM into an immediate, legible error.

        control: an optional runtime.control.JobControl for pause/resume/
        cancel and structured progress -- what the FastAPI server's job
        endpoints drive. Defaults to a private, un-driven JobControl (never
        paused/cancelled unless something explicitly calls it), so plain
        scripted use is unaffected.
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
        from .control import JobControl

        self.config = config or EngineConfig()
        self.control = control or JobControl()
        if self.config.execution_policy != "static":
            raise RuntimeError(
                "execution_policy=%r is a request-boundary controller, not an "
                "in-engine switch; resolve it to a concrete MethodProfile before "
                "constructing StreamingLosslessModel" % self.config.execution_policy)
        if self.config.representation_policy != "uniform":
            raise RuntimeError(
                "representation_policy=%r requires a materialized and validated "
                "RepresentationPlan; this v2 compressed store contains only the "
                "uniform representation" % self.config.representation_policy)
        if self.config.expert_codec != "independent":
            raise RuntimeError(
                "expert_codec=%r requires an XOR-reference-aware expert store; "
                "the dense v2 store must not silently substitute independent "
                "tensors" % self.config.expert_codec)
        if self.config.require_pinned_ram:
            # Fail closed here, before any store/model loading, rather than
            # deep inside _materialize_resident's pin_memory() OOM handler --
            # that handler only fires on an actual pin failure partway
            # through materialization, so a construction-time check is the
            # only way an H9-style regulated run can prove its host can pin
            # the full RAM tier before spending any time on it.
            from .memory_preflight import pinned_memory_preflight
            preflight = pinned_memory_preflight(int(self.config.ram_budget_gb * 1e9))
            if not preflight.success:
                raise RuntimeError(
                    "require_pinned_ram is set but the pinned-RAM preflight "
                    "failed: %s (requested %.3f GB)" %
                    (preflight.reason, preflight.requested_bytes / 1e9))
        self.store = pathlib.Path(store_dir)
        self.manifest = json.loads((self.store / "manifest.json").read_text())
        self._check_store_integrity()

        self.device = device
        self.stats = StreamStats()

        self.progress = self.config.progress
        self.empty_cache_every = self.config.empty_cache_every
        self.io_prefetch_depth = self.config.io_prefetch_depth
        self._prefetch_controller = build_prefetch_controller(
            self.config.prefetch_policy, initial_depth=self.io_prefetch_depth,
            max_depth=self.config.io_prefetch_max_depth,
            target_ready=self.config.prefetch_target_ready,
            kp=self.config.prefetch_kp, ki=self.config.prefetch_ki)
        self._prefetch_pool_depth = (self.io_prefetch_depth
                                     if self.config.prefetch_policy == "fixed"
                                     else self.config.io_prefetch_max_depth)
        self.prefetch = self._prefetch_pool_depth > 0
        self.ram_tier_format = self.config.ram_tier_format
        self._frees = 0
        self._tok_i = 0
        self._tok_total = 0
        self._gen_t0 = 0.0
        self._kv_cache = None
        self._certified_mips_index = None
        self.trace = TraceRecorder(enabled=self.config.trace_events)
        self._last_read_event: dict[str, str] = {}
        self._layer_prepare_events: dict[int, list[str]] = {}
        self._layer_compute_start: dict[int, float] = {}

        self._reader = BinaryWeightReader(self.store / "weights.bin")
        n_prefetch_readers = max(1, self._prefetch_pool_depth)
        self._prefetch_readers = (
            [BinaryWeightReader(self.store / "weights.bin") for _ in range(n_prefetch_readers)]
            if self.prefetch else [])
        self._prefetch_cache: dict[int, tuple[dict, int, int]] = {}
        self._prefetch_threads: dict[int, threading.Thread] = {}
        self._prefetch_started_at: dict[int, float] = {}
        self._prefetch_lead_layers: dict[int, int] = {}
        self._prefetch_inflight_bytes: dict[int, int] = {}
        self._prefetch_lock = threading.Lock()

        config_source = pathlib.Path(source_dir) if source_dir is not None else model_id
        hf_cfg = AutoConfig.from_pretrained(config_source, trust_remote_code=False)
        self.hf_cfg = hf_cfg
        classification = classify_config(hf_cfg)
        auto_model = (
            AutoModelForImageTextToText
            if classification["modality"] == "vision-text"
            else AutoModelForCausalLM
        )
        with torch.device("meta"):
            self.model = auto_model.from_config(hf_cfg, dtype=torch.bfloat16)
        self.model.eval()
        self.adapter = resolve_model_adapter(self.model)
        self.layers = self.adapter.layers
        self.n_layers = len(self.layers)
        self._initial_forward_kwargs: dict[str, torch.Tensor] = {}
        self._expert_slices: dict[str, list[str]] = self.manifest.get(
            "expert_slices", {}
        )

        self._tier = self._compute_tier_assignment(self.config)
        self._ram_cache: dict[str, torch.Tensor] = {}
        self._ram_cache_pageable_keys: set[str] = set()

        vram_cap_gb = self.config.vram_cap_gb
        if vram_cap_gb is not None and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            frac = min(1.0, (vram_cap_gb * 1e9) / total)
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            print("VRAM capped to %.2f GB (%.1f%% of %.2f GB device)"
                  % (vram_cap_gb, frac * 100, total / 1e9), flush=True)

        if (
            self.config.lm_head_policy == "certified_mips"
            and not self.adapter.capabilities.supports_certified_head
        ):
            raise RuntimeError(
                "certified_mips is not available for the %s adapter"
                % self.adapter.capabilities.layout
            )

        self._materialize_resident()
        self._install_expert_streaming()
        if (self.config.lm_head_policy == "certified_mips"
                and self._tier.get(self.CHUNKED_HEAD_KEY) != "vram"):
            raise RuntimeError(
                "lm_head_policy='certified_mips' currently requires lm_head.weight "
                "resident in VRAM; the compressed store has no certified random-row "
                "index, so silently reading the full head would defeat the method")
        if self.config.lm_head_policy == "certified_mips":
            if getattr(self.adapter.output_head, "bias", None) is not None:
                raise RuntimeError(
                    "certified_mips currently requires a bias-free lm_head; use "
                    "lm_head_policy='full' for this architecture")
            from .certified_mips import MIPSIndex
            weight = self.adapter.output_head.weight
            block_rows = 256
            blocks = (weight.shape[0] + block_rows - 1) // block_rows
            estimated_index_bytes = int(
                weight.numel() * 8 + 3 * blocks * weight.shape[1] * 8)
            limit_bytes = int(self.config.mips_index_ram_limit_gb * 1e9)
            if estimated_index_bytes > limit_bytes:
                raise RuntimeError(
                    "certified_mips index needs about %.2f GB host RAM, above "
                    "mips_index_ram_limit_gb=%.2f" %
                    (estimated_index_bytes / 1e9,
                     self.config.mips_index_ram_limit_gb))
            build_t0 = time.perf_counter()
            self._certified_mips_index = MIPSIndex.build(weight, block_rows=block_rows)
            self.stats.mips_index_build_seconds = time.perf_counter() - build_t0
        self._install_hooks()
        self._install_streamed_module_hooks()
        self._install_chunked_lm_head()
        # Research traces represent steady-state generation, not one-time
        # materialization or index construction. Clear startup spans before
        # the initial generation prefetch begins.
        if self.trace.enabled:
            self._clear_startup_trace()
        if self.prefetch:
            for ahead in range(1, self._prefetch_controller.choose_depth() + 1):
                self._start_prefetch(ahead - 1, lead_layers=ahead - 1)

    def _clear_startup_trace(self) -> None:
        """Start the measured DAG without dependencies on discarded events.

        Materializing resident tensors records startup transfers and caches
        their event IDs in the dependency maps.  Clearing only the recorder
        left those maps pointing at events no longer present in the trace,
        which made the first steady-state compute span an invalid DAG.
        """
        self.trace.clear()
        self._last_read_event.clear()
        self._layer_prepare_events.clear()
        self._layer_compute_start.clear()

    def close(self) -> None:
        """Join any still-in-flight background prefetch threads before
        closing the file handles they read through.

        Without this, a prefetch started for a layer that generation never
        actually reaches -- e.g. draft_self_logits truncates
        model.model.layers to a prefix, so the one-layer-ahead prefetch it
        fires can point past the truncated range and sit unconsumed until a
        later full forward pass reaches it -- can still be mid-read() when
        close() runs, racing the file handle out from under it
        ("I/O operation on closed file" in a background thread). Harmless
        to the caller (results already returned are unaffected either way)
        but a real race, caught by tests/test_streaming_engine_gpu.py's
        self-draft tests, which close() shortly after touching only a few
        layers.
        """
        with self._prefetch_lock:
            pending = list(self._prefetch_threads.values())
        for th in pending:
            th.join()
        self._reader.close()
        for r in self._prefetch_readers:
            r.close()
        if self.config.trace_output:
            self.trace.save(self.config.trace_output)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def mips_index_bytes(self) -> int:
        return (self._certified_mips_index.nbytes
                if self._certified_mips_index is not None else 0)

    def _check_store_integrity(self) -> None:
        """Two cheap, always-on checks that catch the two most common ways
        a store goes bad, without paying the cost of hashing the whole
        (multi-GB) file on every startup -- see binstore.verify_store for
        the expensive, explicit, opt-in full check.
        """
        version = self.manifest.get("schema_version")
        if version != CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                "store at %s has schema_version=%r, this code expects %d. "
                "Old .npz-based stores predate this field entirely and are "
                "not readable by this engine -- re-run compress_model_to_disk."
                % (self.store, version, CURRENT_SCHEMA_VERSION))

        weights_path = self.store / "weights.bin"
        if not weights_path.exists():
            raise RuntimeError("no weights.bin found at %s" % self.store)
        expected_min_size = 0
        for meta in self.manifest["tensors"].values():
            for ref in meta.get("blobs", {}).values():
                expected_min_size = max(expected_min_size, ref["offset"] + ref["nbytes"])
        actual_size = weights_path.stat().st_size
        if actual_size < expected_min_size:
            raise RuntimeError(
                "weights.bin at %s is truncated: manifest references bytes "
                "up to offset %d but the file is only %d bytes -- the store "
                "is incomplete or was corrupted; recompress it. (Run "
                "binstore.verify_store() for a full checksum pass.)"
                % (weights_path, expected_min_size, actual_size))

    # -- residency planning ------------------------------------------------

    CHUNKED_HEAD_KEY = "lm_head.weight"

    def _chunked_head_rows(self, cfg: EngineConfig) -> int:
        """Rows of lm_head to compute per block, or 0 if chunking is off or
        does not apply.

        Does not apply to a TIED model: there lm_head aliases
        embed_tokens.weight and has no manifest entry of its own, so there
        is nothing separate to stream in blocks. Does not apply to a
        raw-stored (uncompressed) head either -- decompress_rows_gpu
        decodes chunk ranges, which only exist for entropy-coded tensors.
        """
        if cfg.lm_head_slice_rows <= 0:
            return 0
        meta = self.manifest["tensors"].get(self.CHUNKED_HEAD_KEY)
        if not meta or not meta.get("compressed") or len(meta["shape"]) != 2:
            return 0
        return cfg.lm_head_slice_rows

    def _stream_only_overrides(self, cfg: EngineConfig) -> dict:
        """{key: peak bytes live at once} for tensors the engine computes in
        blocks and never holds whole -- what lets vram_planner stop
        reserving headroom for a tensor that is never fully materialized."""
        step = self._chunked_head_rows(cfg)
        if not step:
            return {}
        rows, cols = self.manifest["tensors"][self.CHUNKED_HEAD_KEY]["shape"]
        return {self.CHUNKED_HEAD_KEY: min(step, rows) * cols * 2}  # bf16

    def _compute_tier_assignment(self, cfg: EngineConfig) -> dict[str, str]:
        """Every manifest key -> "vram" | "ram" | "disk" | "row_gather".

        Without a vram_budget_gb, this reproduces the engine's original
        fixed policy exactly: every non-layer tensor (embed_tokens, lm_head,
        norms) is VRAM-resident, every decoder-layer tensor streams from
        disk every token. With a budget, vram_planner.plan_tiers ranks ALL
        tensors -- including decoder-layer weights, which the fixed policy
        never considered for residency at all -- by bus-traffic-avoided per
        byte, and fills VRAM then RAM greedily.
        """
        tiers: dict[str, str] = {}
        for key, meta in self.manifest["tensors"].items():
            if meta.get("row_gather"):
                tiers[key] = "row_gather"

        # A chunked head is computed in row blocks and must never be
        # materialized whole -- otherwise the 1.556 GB it exists to avoid
        # gets held anyway and the feature silently saves nothing. This
        # applies under BOTH residency policies: the legacy fixed policy
        # below would otherwise mark it "vram" simply for not being a
        # decoder layer. Caught by a test that asserted the weight stays on
        # the meta device.
        chunked_head = (self.CHUNKED_HEAD_KEY
                        if self._chunked_head_rows(cfg) else None)
        if chunked_head:
            tiers[chunked_head] = "disk"

        if not cfg.uses_tiered_residency:
            for key in self.manifest["tensors"]:
                if key in tiers:
                    continue
                tiers[key] = "disk" if self.adapter.is_layer_key(key) else "vram"
            return tiers

        from .vram_planner import plan_from_manifest
        stream_only = self._stream_only_overrides(cfg)
        draft_layer_indices = None
        draft_uses = 1
        if cfg.draft_mode == "self" and cfg.pin_draft_layers:
            # Self-drafting touches layers [0, draft_exit_layer) once per
            # proposed token PLUS once during verification -- (spec_k + 1)
            # times per sweep, not once -- so the planner must rank them
            # accordingly or self-drafting just re-streams them repeatedly.
            # See vram_planner's "Self-speculation breaks..." docstring
            # section and PROPOSAL_ADAPTIVE.md mechanism C. spec_k (not a
            # live bandit mean) is the estimate at plan time -- this runs
            # once at model construction, before any sweep has happened.
            draft_layer_indices = range(cfg.draft_exit_layer)
            draft_uses = cfg.spec_k + 1
        critical_profile = (CriticalPathProfile.load(cfg.critical_path_profile)
                            if cfg.critical_path_profile else None)
        if critical_profile is not None:
            eligible = {
                key for key, meta in self.manifest["tensors"].items()
                if not meta.get("row_gather") and key not in stream_only
            }
            covered = eligible & set(critical_profile.tensors)
            coverage = len(covered) / max(len(eligible), 1)
            if critical_profile.trace_count < 1 or coverage < 0.90:
                missing = sorted(eligible - covered)
                raise RuntimeError(
                    "critical_path_profile covers %.1f%% of placement candidates; "
                    "at least 90%% measured coverage is required to avoid silently "
                    "falling back to traffic estimates. Missing examples: %s" %
                    (100.0 * coverage, ", ".join(missing[:5])))
        if cfg.placement_policy in (
                "replay_cem", "replay_qubo", "replay_extent_qubo"):
            # The replay planners validate against a byte count frozen into
            # the plan at search time (ReplayResidencyPlan.to_tier_plan
            # takes no activation_slack_bytes at all), so max_context isn't
            # threaded through here -- these are opt-in research placement
            # policies, not the path max_context's docstring promises.
            from .replay_planner import ReplayResidencyPlan
            frozen = ReplayResidencyPlan.load(cfg.replay_plan_state)
            plan = frozen.to_tier_plan(
                self.manifest, vram_budget_gb=cfg.vram_budget_gb,
                ram_budget_gb=cfg.ram_budget_gb or 0.0,
                decode_slice_elems=cfg.decode_slice_elems,
                stream_only=stream_only)
        else:
            forced_ram = ({self.CHUNKED_HEAD_KEY}
                          if cfg.lm_head_policy == "ram_overlay" else set())
            if forced_ram and self.CHUNKED_HEAD_KEY not in self.manifest["tensors"]:
                raise RuntimeError(
                    "lm_head_policy='ram_overlay' requires an untied, stored "
                    "lm_head.weight; this checkpoint has no independent head")
            plan_kwargs = {}
            if cfg.max_context is not None:
                from .vram_planner import (
                    DEFAULT_ACTIVATION_SLACK_BYTES, kv_cache_bytes_per_token,
                )
                kv_reserve = kv_cache_bytes_per_token(self.hf_cfg) * cfg.max_context
                plan_kwargs["activation_slack_bytes"] = (
                    DEFAULT_ACTIVATION_SLACK_BYTES + kv_reserve)
            plan = plan_from_manifest(
                self.manifest, vram_budget_gb=cfg.vram_budget_gb,
                ram_budget_gb=cfg.ram_budget_gb or 0.0,
                decode_slice_elems=cfg.decode_slice_elems,
                draft_layer_indices=draft_layer_indices,
                draft_uses=draft_uses, stream_only=stream_only,
                critical_path_profile=critical_profile,
                forced_ram_keys=forced_ram,
                placement_policy=cfg.placement_policy,
                **plan_kwargs)
        if not plan.feasible:
            raise RuntimeError("VRAM budget is infeasible: " + plan.reason)
        for key in plan.vram_keys:
            if key != chunked_head:
                tiers[key] = "vram"
        for key in plan.ram_keys:
            if key != chunked_head:
                tiers[key] = "ram"
        for key in plan.disk_keys:
            tiers[key] = "disk"
        # Packed experts are deliberately sliced so the router can fetch
        # only selected experts. Keeping an individual slice resident is a
        # future cache policy; the current selected-expert path is disk-only
        # and must not let the generic planner claim otherwise.
        for keys in self._expert_slices.values():
            for key in keys:
                tiers[key] = "disk"
        return tiers

    # -- store access ----------------------------------------------------

    def _read_tensor_arrays(self, key: str) -> tuple[dict, float, int, int]:
        meta = self.manifest["tensors"][key]
        named_refs = list(meta["blobs"].items())
        t0 = time.perf_counter()
        if self.config.storage_read_policy in (
                "coalesced_extents", "tensor_extents"):
            decoded, read_calls, extent_bytes = self._reader.read_many(
                [ref for _, ref in named_refs],
                max_gap_bytes=self.config.storage_extent_max_gap_bytes,
                max_extent_bytes=self.config.storage_extent_max_bytes)
            arrays = {name: array for (name, _), array in zip(named_refs, decoded)}
        else:
            arrays = {name: self._reader.read(ref) for name, ref in named_refs}
            read_calls = len(named_refs)
            extent_bytes = sum(int(ref["nbytes"]) for _, ref in named_refs)
        end = time.perf_counter()
        event_id = self.trace.record("read", "storage-main", t0, end, tensor_key=key,
                                     nbytes=int(meta["comp_bytes"]))
        if event_id:
            self._last_read_event[key] = event_id
        return arrays, end - t0, read_calls, extent_bytes

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
            dependency = self._last_read_event.pop(key, None)
            t1 = time.perf_counter()
            out = out.to(self.device)
            if self.trace.enabled and torch.cuda.is_available():
                torch.cuda.synchronize()
            event_id = self.trace.record(
                "transfer", "cuda-default", t1, time.perf_counter(),
                tensor_key=key, dependencies=([dependency] if dependency else ()))
            self._remember_layer_prepare(key, event_id)
            return out

        if self.trace.enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        layer = self._compressed_layer(key, arrays)
        out = decompress_layer_gpu(layer, device=self.device,
                                   max_slice_elems=self.config.decode_slice_elems)
        if self.trace.enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        self.stats.decode_seconds += end - t1
        dependency = self._last_read_event.pop(key, None)
        event_id = self.trace.record(
            "decode", "cuda-default", t1, end, tensor_key=key,
            dependencies=([dependency] if dependency else ()))
        self._remember_layer_prepare(key, event_id)
        return out

    def _remember_layer_prepare(self, key: str, event_id: str | None) -> None:
        if not event_id or not self.adapter.is_layer_key(key):
            return
        try:
            layer_idx = int(key[len(self.adapter.layer_prefix) + 1:].split(".", 1)[0])
        except (IndexError, ValueError):
            return
        self._layer_prepare_events.setdefault(layer_idx, []).append(event_id)

    def _compressed_layer(self, key: str, arrays: dict) -> CompressedLayer:
        """Rebuild the CompressedLayer view over already-read arrays.
        Shared by whole-tensor decode (_decode_tensor) and block decode
        (the chunked lm_head projection), so both read the identical
        metadata and cannot drift apart."""
        meta = self.manifest["tensors"][key]
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
        return CompressedLayer(
            sign_mantissa=torch.from_numpy(arrays["sign_mantissa"]),
            encoded=enc,
            shape=tuple(meta["shape"]),
        )

    def _load_tensor(self, key: str) -> torch.Tensor:
        """Synchronous read-then-decode -- the path used whenever a
        prefetch did not already do the read (resident materialization at
        startup, and any disk-tier layer tensor the background thread has
        not finished reading yet)."""
        arrays, io_s, read_calls, extent_bytes = self._read_tensor_arrays(key)
        self.stats.bytes_read += self.manifest["tensors"][key]["comp_bytes"]
        self.stats.io_seconds += io_s
        self.stats.storage_read_calls += read_calls
        self.stats.storage_extent_bytes += extent_bytes
        return self._decode_tensor(key, arrays)

    # -- per-parameter storage swap ---------------------------------------
    #
    # Everything below replaces the module-level to_empty()/to("meta")
    # pattern with direct, per-parameter storage replacement. That pattern
    # required a WHOLE module's parameters to be materialized or freed
    # together, which is fine when every parameter in a layer gets the same
    # treatment (the original fixed policy: whole layer loaded, whole layer
    # freed) but breaks the moment two sibling parameters in the same layer
    # need DIFFERENT treatment -- one permanently VRAM-resident, another
    # freed every token -- which tiered residency now requires.

    def _navigate(self, root, dotted: str):
        target = root
        parts = dotted.split(".")
        for p in parts[:-1]:
            target = getattr(target, p)
        return target, parts[-1]

    def _set_param(self, root, dotted: str, tensor: torch.Tensor) -> None:
        """Replace a (possibly meta) parameter/buffer with a real tensor by
        direct assignment -- no existing real storage required first."""
        target, leaf = self._navigate(root, dotted)
        if leaf in target._parameters:
            target._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
        elif leaf in target._buffers:
            target._buffers[leaf] = tensor
        else:
            raise AttributeError("%r has neither parameter nor buffer %r" % (target, leaf))

    def _to_meta_param(self, root, dotted: str) -> None:
        """The free-side counterpart to _set_param: replace a real
        parameter/buffer with an empty meta-device placeholder."""
        target, leaf = self._navigate(root, dotted)
        if leaf in target._parameters:
            cur = target._parameters[leaf]
            target._parameters[leaf] = torch.nn.Parameter(
                torch.empty_like(cur, device="meta"), requires_grad=False)
        elif leaf in target._buffers:
            cur = target._buffers[leaf]
            target._buffers[leaf] = torch.empty_like(cur, device="meta")
        else:
            raise AttributeError("%r has neither parameter nor buffer %r" % (target, leaf))

    # -- residency -------------------------------------------------------

    def _install_embed_row_gather(self, key: str) -> None:
        """For untied models, embed_tokens.weight is stored raw and
        row-addressable rather than fully materialized: a token embedding
        lookup only ever needs one row per input token, not all 151936 of
        them, so pulling the whole 1.56 GB table into VRAM for that is pure
        waste. lm_head still needs its full matrix to produce logits over
        the whole vocabulary and is tiered normally -- this only applies
        where the two are NOT tied together, since a tied model has to
        materialize the full table anyway to serve as lm_head.
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

        self.adapter.embedding.forward = forward

    def _materialize_resident(self) -> None:
        """Materialize every VRAM- or RAM-tier tensor (which, under the
        default config, is exactly "everything that is not a decoder
        layer" -- see _compute_tier_assignment), then re-tie weights the
        checkpoint does not store separately.

        Weight tying is the subtle part. Qwen2.5 (and many others) set
        tie_word_embeddings=True, so the safetensors contains NO
        lm_head.weight -- it is meant to alias embed_tokens.weight. Since
        there is no lm_head.weight in the store, materializing it as an
        independent tensor would leave it with nothing to load: it would
        run on whatever _set_param happened to construct, which is
        uninitialized memory unless explicitly re-tied afterward. The
        model still produces logits in that state, so nothing crashes -- it
        just returns confident nonsense (measured, before this was fixed:
        max abs logit diff 22.1, argmax token 100628 vs the correct 12095).
        Re-tying after materialization is what fixes it, and any parameter
        left with no source now raises instead of quietly running on
        garbage.
        """
        embed_key = self.adapter.embedding_prefix + ".weight"
        row_gather = self._tier.get(embed_key) == "row_gather"
        embed_mod = self.adapter.embedding if row_gather else None

        missing = []
        for name, mod in self.model.named_modules():
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
            for pname, _ in params + buffers:
                key = (name + "." + pname) if name else pname
                if key not in self.manifest["tensors"]:
                    if key in self._expert_slices:
                        continue
                    missing.append(key)
                    continue
                tier = self._tier.get(key, "disk")
                if tier in ("disk", "row_gather"):
                    continue  # left on meta; loaded per-token / row-gathered
                if tier == "vram":
                    self._set_param(self.model, key, self._load_tensor(key))
                elif tier == "ram":
                    if self.ram_tier_format == "compressed":
                        # the archived streaming proposal's own H1 (unrelated to the
                        # current H1 critical-path-residency hypothesis):
                        # cache the COMPRESSED bytes (fits ~1.45x more
                        # tensors in ram_budget_gb) instead of a decoded
                        # pinned tensor -- the tradeoff is a real GPU decode
                        # on every token instead of a memcpy. See
                        # EngineConfig.ram_tier_format.
                        arrays, _, _, _ = self._read_tensor_arrays(key)
                        self._ram_cache[key] = arrays
                    else:
                        # Decode through one transient GPU tensor, copy it to
                        # host RAM, and immediately release the GPU storage.
                        # RAM-tier tensors deliberately remain meta until a
                        # live-range hook needs them; retaining this startup
                        # copy would silently turn the RAM tier into VRAM.
                        # Emptying the allocator cache bounds startup peak by
                        # one decoded tensor instead of accumulating cached
                        # transient blocks across the materialization loop.
                        gpu_tensor = self._load_tensor(key)
                        cpu_tensor = gpu_tensor.to("cpu")
                        try:
                            cached = cpu_tensor.pin_memory()
                        except RuntimeError as exc:
                            # WSL commonly has a hard 64 MB memlock ceiling.
                            # Pageable RAM preserves correctness and lifetime
                            # semantics but can transfer more slowly, so the
                            # degradation must remain observable.
                            if "out of memory" not in str(exc).lower():
                                raise
                            if self.config.require_pinned_ram:
                                raise RuntimeError(
                                    "pinned RAM is required by this configuration, "
                                    "but pin_memory failed for %s; pageable fallback "
                                    "is disabled by the experiment mechanism gate" % key
                                ) from exc
                            cached = cpu_tensor
                            self._ram_cache_pageable_keys.add(key)
                            warnings.warn(
                                "pinned RAM unavailable for %s; using pageable "
                                "host memory (raise the memlock limit for the "
                                "intended overlay experiment)" % key,
                                RuntimeWarning, stacklevel=2)
                        self._ram_cache[key] = cached
                        del cpu_tensor
                        del gpu_tensor
                        torch.cuda.empty_cache()
                    # RAM-tier tensors deliberately stay on meta between
                    # uses. Decoder-layer hooks materialize them in
                    # _load_layer; non-layer hooks below do the same around
                    # their exact live range. Keeping a disposable startup
                    # GPU copy here made RAM-tier output heads silently act
                    # VRAM-resident and defeated liveness overlays.

        # Rebuild modules whose state is computed, not loaded. Their
        # non-persistent buffers were left on the meta device above; a
        # fresh instance recomputes them from config on the target device.
        m = self.adapter.language_model
        if getattr(m, "rotary_emb", None) is not None:
            try:
                m.rotary_emb = type(m.rotary_emb)(config=self.adapter.language_config,
                                                  device=self.device)
            except TypeError:
                m.rotary_emb = type(m.rotary_emb)(
                    self.adapter.language_config
                ).to(self.device)
        visual = self.adapter.vision_model
        if visual is not None and getattr(visual, "rotary_pos_emb", None) is not None:
            rotary = visual.rotary_pos_emb
            # Qwen3-VL's vision rotary buffer is computed from two constructor
            # scalars and is intentionally absent from safetensors.
            visual.rotary_pos_emb = type(rotary)(
                dim=rotary.dim, theta=rotary.theta
            ).to(self.device)

        tied = getattr(self.model.config, "tie_word_embeddings", False)
        if tied and "lm_head.weight" in missing:
            self.adapter.output_head.weight = self.adapter.embedding.weight
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

    def _install_expert_streaming(self) -> None:
        """Load only routed experts for packed Qwen MoE modules.

        New stores split the packed 3D expert matrices into independent 2D
        slices. The ordinary router remains resident in its decoder layer;
        this replacement forward reads only experts selected for the current
        tokens and frees each pair immediately after use.
        """

        if not self._expert_slices:
            return
        for module_name, module in self.model.named_modules():
            gate_parent = module_name + ".gate_up_proj"
            down_parent = module_name + ".down_proj"
            if gate_parent not in self._expert_slices or down_parent not in self._expert_slices:
                continue
            if not all(hasattr(module, name) for name in ("num_experts", "act_fn")):
                continue
            engine = self

            def forward(hidden_states, top_k_index, top_k_weights, *, _module=module,
                        _gate=gate_parent, _down=down_parent):
                final_hidden_states = torch.zeros_like(hidden_states)
                with torch.no_grad():
                    expert_mask = torch.nn.functional.one_hot(
                        top_k_index, num_classes=_module.num_experts
                    ).permute(2, 1, 0)
                    expert_hit = torch.greater(
                        expert_mask.sum(dim=(-1, -2)), 0
                    ).nonzero()
                for expert_value in expert_hit:
                    engine.control.checkpoint()
                    expert_index = int(expert_value[0])
                    if expert_index == _module.num_experts:
                        continue
                    top_k_pos, token_idx = torch.where(expert_mask[expert_index])
                    current_state = hidden_states[token_idx]
                    gate_up = engine._load_tensor(
                        "%s.__expert__.%d" % (_gate, expert_index)
                    )
                    down = engine._load_tensor(
                        "%s.__expert__.%d" % (_down, expert_index)
                    )
                    gate, up = torch.nn.functional.linear(
                        current_state, gate_up
                    ).chunk(2, dim=-1)
                    current = _module.act_fn(gate) * up
                    current = torch.nn.functional.linear(current, down)
                    current = current * top_k_weights[token_idx, top_k_pos, None]
                    final_hidden_states.index_add_(
                        0, token_idx, current.to(final_hidden_states.dtype)
                    )
                    del gate_up, down, gate, up, current
                return final_hidden_states

            module.forward = forward

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
        self.control.report(phase="generate", pct=pct, token=self._tok_i + 1,
                            tokens_total=self._tok_total, layer=layer_idx + 1,
                            layers_total=self.n_layers, gb_read=self.stats.bytes_read / 1e9,
                            elapsed_s=elapsed, eta_s=eta)
        if self.progress and (layer_idx % 5 == 0 or layer_idx == self.n_layers - 1):
            # stderr, not stdout: generated text goes to stdout (see
            # cli.py's cmd_run), so `afterimage run ... > answer.txt` or
            # `| some_tool` gets just the answer, with this status line
            # still visible in the terminal and still capturable separately
            # (`2> progress.log`) for the "captured to a log file" case this
            # function's docstring cares about.
            print("  [%s] %5.1f%%  tok %d/%d  layer %2d/%d  %.1f GB  "
                  "%.0fs elapsed  ETA %.0fs"
                  % (bar, pct, self._tok_i + 1, self._tok_total,
                     layer_idx + 1, self.n_layers,
                     self.stats.bytes_read / 1e9, elapsed, eta),
                  file=sys.stderr, flush=True)

    def _read_layer_tensor_arrays(
            self, idx: int, reader: BinaryWeightReader) -> tuple[dict, int, int]:
        """Reads only this layer's DISK-tier tensors -- vram/ram-tier ones
        are never re-read from disk after _materialize_resident, so
        prefetching them would be pure waste."""
        layer = self.layers[idx]
        layer_keys: list[str] = []
        for pname, _ in layer.named_parameters():
            key = self.adapter.layer_key(idx, pname)
            if key not in self.manifest["tensors"]:
                continue
            if self._tier.get(key, "disk") != "disk":
                continue
            layer_keys.append(key)

        result: dict[str, tuple[dict, float]] = {}
        if not layer_keys:
            return result, 0, 0

        if self.config.storage_read_policy == "coalesced_extents":
            # An extent merges several tensors' blobs into one physical
            # read, so no per-tensor timing is directly observable -- only
            # the whole extent's wall time is real. Apportioning it by byte
            # share is a MODELLED estimate, not a measurement, and must say
            # so: a critical-path profile built from unmarked synthetic
            # spans would look identical to one built from genuinely timed
            # per_blob reads, silently reintroducing the traffic-density
            # proxy H1 exists to replace.
            requests = [(key, name, ref)
                       for key in layer_keys
                       for name, ref in self.manifest["tensors"][key]["blobs"].items()]
            read_start = time.perf_counter()
            decoded, read_calls, extent_bytes = reader.read_many(
                [ref for _, _, ref in requests],
                max_gap_bytes=self.config.storage_extent_max_gap_bytes,
                max_extent_bytes=self.config.storage_extent_max_bytes)
            read_end = time.perf_counter()

            grouped: dict[str, dict] = {}
            key_bytes: dict[str, int] = {}
            for (key, name, ref), array in zip(requests, decoded):
                grouped.setdefault(key, {})[name] = array
                key_bytes[key] = key_bytes.get(key, 0) + int(ref["nbytes"])
            total_blob_bytes = max(sum(key_bytes.values()), 1)
            cursor = read_start
            for key, arrays in grouped.items():
                io_s = (read_end - read_start) * key_bytes[key] / total_blob_bytes
                key_end = min(read_end, cursor + io_s)
                event_id = self.trace.record(
                    "read", "storage-%d" % id(reader), cursor, key_end,
                    tensor_key=key, nbytes=key_bytes[key],
                    metadata={"modelled": True})
                if event_id:
                    self._last_read_event[key] = event_id
                result[key] = (arrays, io_s)
                cursor = key_end
            return result, read_calls, extent_bytes

        # per_blob and tensor_extents keep each tensor as its own measured
        # scheduling unit.  H17 changes only the requests *inside* that
        # unit; unlike H14 it never creates one layer-wide blocking extent.
        read_calls = 0
        extent_bytes = 0
        for key in layer_keys:
            meta = self.manifest["tensors"][key]
            t0 = time.perf_counter()
            named_refs = list(meta["blobs"].items())
            if self.config.storage_read_policy == "tensor_extents":
                decoded, key_calls, key_extent_bytes = reader.read_many(
                    [ref for _, ref in named_refs],
                    max_gap_bytes=self.config.storage_extent_max_gap_bytes,
                    max_extent_bytes=self.config.storage_extent_max_bytes)
                arrays = {name: array for (name, _), array in zip(named_refs, decoded)}
            else:
                arrays = {name: reader.read(ref) for name, ref in named_refs}
                key_calls = len(named_refs)
                key_extent_bytes = int(meta["comp_bytes"])
            t1 = time.perf_counter()
            read_calls += key_calls
            extent_bytes += key_extent_bytes
            event_id = self.trace.record("read", "storage-%d" % id(reader), t0, t1,
                                         tensor_key=key, nbytes=int(meta["comp_bytes"]))
            if event_id:
                self._last_read_event[key] = event_id
            result[key] = (arrays, t1 - t0)
        return result, read_calls, extent_bytes

    def _start_prefetch(self, idx: int, *, lead_layers: int = 1) -> None:
        """Kick off a background read of layer idx's disk-tier bytes.

        Uses one of a small POOL of BinaryWeightReaders (io_prefetch_depth
        of them, each its own file handle) rather than a single shared
        reader -- concurrent seek()+read() on one handle is not
        thread-safe, and giving each in-flight prefetch its own fd
        sidesteps that race entirely instead of adding locking around
        every read. A depth-1 pool (the default) reproduces the original
        single-layer-ahead design; deeper pools let more NVMe queue depth
        be in flight at once, which the literature is unambiguous matters
        for NVMe throughput.

        The Thread object itself is kept in _prefetch_threads so
        _load_layer can JOIN it instead of racing it: an earlier version
        only tracked "pending" as a set membership flag, and _load_layer
        fell back to an entirely separate synchronous read the instant the
        cache wasn't ready yet -- meaning two readers ended up pulling the
        SAME bytes off the SAME disk at the SAME time whenever prefetch
        didn't win the race. Measured on the real 14B store, that made
        streaming SLOWER with prefetch on than off. Joining the in-flight
        thread instead means the worst case degrades to "wait for the one
        read already happening," not "start a second one that competes
        with it."
        """
        if idx >= self.n_layers or not self._prefetch_readers:
            return

        reader = self._prefetch_readers[idx % len(self._prefetch_readers)]

        def worker():
            try:
                result = self._read_layer_tensor_arrays(idx, reader)
                with self._prefetch_lock:
                    self._prefetch_cache[idx] = result
            finally:
                with self._prefetch_lock:
                    self._prefetch_inflight_bytes.pop(idx, None)

        with self._prefetch_lock:
            if idx in self._prefetch_cache or idx in self._prefetch_threads:
                return
            th = threading.Thread(target=worker, daemon=True)
            self._prefetch_threads[idx] = th
            self._prefetch_started_at[idx] = time.perf_counter()
            self._prefetch_lead_layers[idx] = max(0, int(lead_layers))
            inflight = sum(
                sum(int(ref["nbytes"])
                    for ref in self.manifest["tensors"][key]["blobs"].values())
                for key, tier in self._tier.items()
                if tier == "disk" and key.startswith(self.adapter.layer_key(idx) + "."))
            self._prefetch_inflight_bytes[idx] = inflight
            self.stats.prefetch_peak_inflight_bytes = max(
                self.stats.prefetch_peak_inflight_bytes,
                sum(self._prefetch_inflight_bytes.values()))
        th.start()

    def _load_layer(self, idx: int) -> None:
        self.control.checkpoint()  # pause/cancel boundary: one layer at a time
        demand_time = time.perf_counter()
        cached = None
        ready_before_wait = False
        prefetch_wait = 0.0
        lead_s = 0.0
        lead_layers = 0
        if self.prefetch:
            with self._prefetch_lock:
                ready_before_wait = idx in self._prefetch_cache
                th = self._prefetch_threads.pop(idx, None)
                started_at = self._prefetch_started_at.pop(idx, None)
                lead_layers = self._prefetch_lead_layers.pop(idx, 0)
            if started_at is not None:
                lead_s = max(0.0, demand_time - started_at)
            if th is not None:
                wait_t0 = time.perf_counter()
                th.join()
                prefetch_wait = time.perf_counter() - wait_t0
            with self._prefetch_lock:
                cached_batch = self._prefetch_cache.pop(idx, None)
            if cached_batch is None:
                cached, cached_read_calls, cached_extent_bytes = None, 0, 0
            else:
                cached, cached_read_calls, cached_extent_bytes = cached_batch
            # Fire the next `io_prefetch_depth` layers' reads NOW, before
            # decoding idx's own bytes, not after this whole method
            # returns. Per-layer GPU compute is a few tens of ms -- far too
            # short a window to hide a multi-hundred-ms read behind.
            # Overlapping the next read(s) with THIS layer's decode-plus-
            # compute instead gives the background threads a window an
            # order of magnitude larger to hide inside.
            active_depth = self._prefetch_controller.choose_depth()
            for ahead in range(1, active_depth + 1):
                self._start_prefetch(idx + ahead, lead_layers=ahead)

            useful_bytes = sum(
                int(self.manifest["tensors"][key]["comp_bytes"])
                for key in (cached or {}))
            io_seconds = sum(float(item[1]) for item in (cached or {}).values())
            self.stats.storage_read_calls += cached_read_calls
            self.stats.storage_extent_bytes += cached_extent_bytes
            self.stats.prefetch_wait_seconds += prefetch_wait
            if ready_before_wait:
                self.stats.prefetch_hits += 1
            else:
                self.stats.prefetch_misses += 1
            self._prefetch_controller.update(PrefetchObservation(
                ready=ready_before_wait, wait_s=prefetch_wait,
                useful_bytes=useful_bytes,
                bandwidth_bytes_s=(useful_bytes / io_seconds if io_seconds > 0 else 0.0),
                lead_s=lead_s, lead_layers=lead_layers))

        layer = self.layers[idx]
        for pname, _ in layer.named_parameters():
            key = self.adapter.layer_key(idx, pname)
            if key not in self.manifest["tensors"]:
                continue
            tier = self._tier.get(key, "disk")
            if tier == "vram":
                continue  # materialized once in _materialize_resident, permanent
            elif tier == "ram":
                if self.ram_tier_format == "compressed":
                    out = self._decode_tensor(key, self._ram_cache[key])
                else:
                    t0 = time.perf_counter()
                    out = self._ram_cache[key].to(
                        self.device,
                        non_blocking=(key not in self._ram_cache_pageable_keys))
                    if self.trace.enabled and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    event_id = self.trace.record(
                        "transfer", "cuda-default", t0, time.perf_counter(),
                        tensor_key=key)
                    self._remember_layer_prepare(key, event_id)
            elif cached is not None and key in cached:
                arrays, io_s = cached[key]
                self.stats.io_seconds += io_s
                self.stats.bytes_read += self.manifest["tensors"][key]["comp_bytes"]
                out = self._decode_tensor(key, arrays)
            else:
                out = self._load_tensor(key)
            self._set_param(layer, pname, out)

        self.stats.layer_loads += 1
        self._report_progress(idx)

    def _free_layer(self, idx: int) -> None:
        layer = self.layers[idx]
        for pname, _ in layer.named_parameters():
            key = self.adapter.layer_key(idx, pname)
            if key not in self.manifest["tensors"]:
                continue
            if self._tier.get(key, "disk") == "vram":
                continue  # permanent; never freed
            self._to_meta_param(layer, pname)
        self._frees += 1
        if (
            torch.cuda.is_available()
            and self.empty_cache_every
            and self._frees % self.empty_cache_every == 0
        ):
            torch.cuda.empty_cache()

    def _synchronize_device(self) -> None:
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

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
                if self.trace.enabled or self.config.prefetch_policy == "mpc":
                    self._layer_compute_start[idx] = time.perf_counter()
                return None
            return pre

        def make_post(idx):
            def post(module, args, kwargs, output):
                self._synchronize_device()
                if self.trace.enabled or self.config.prefetch_policy == "mpc":
                    end = time.perf_counter()
                    start = self._layer_compute_start.pop(idx, end)
                    self._prefetch_controller.update_compute(end - start)
                if self.trace.enabled:
                    dependencies = self._layer_prepare_events.pop(idx, ())
                    self.trace.record("compute", "cuda-default", start, end,
                                      dependencies=dependencies,
                                      metadata={"layer": idx})
                self._free_layer(idx)
                return output
            return post

        for i, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(make_pre(i), with_kwargs=True)
            layer.register_forward_hook(make_post(i), with_kwargs=True)

    def _streamed_module_params(self) -> dict:
        """Non-decoder-layer modules that own disk-tier parameters, as
        {module_name: [param_name, ...]}.

        In practice this is lm_head on an untied model: at 1.56 GB on a
        14B it is the single largest tensor in the network and, kept
        resident, it IS essentially the whole VRAM difference against
        AirLLM (which streams it like anything else). Decoder layers are
        excluded because _install_hooks already streams those.
        """
        embed_key = self.adapter.embedding_prefix + ".weight"
        embed_mod = (self.adapter.embedding
                     if self._tier.get(embed_key) == "row_gather" else None)
        # A chunked lm_head is driven by its own replaced forward (see
        # _install_chunked_lm_head), which loads and frees per row block.
        # Letting the whole-tensor hooks below ALSO fire for it would
        # materialize the full 1.556 GB the chunking exists to avoid.
        chunked = ({self.CHUNKED_HEAD_KEY}
                   if self._chunked_head_rows(self.config) else set())

        out: dict[str, list[str]] = {}
        for name, mod in self.model.named_modules():
            if not name or name.startswith(self.adapter.layer_prefix + ".") or mod is embed_mod:
                continue
            pnames = [
                pname for pname, _ in mod.named_parameters(recurse=False)
                if (name + "." + pname) in self.manifest["tensors"]
                and self._tier.get(name + "." + pname) in ("disk", "ram")
                and (name + "." + pname) not in chunked
            ]
            if pnames:
                out[name] = pnames
        return out

    def _install_streamed_module_hooks(self) -> None:
        """Stream non-layer modules (lm_head) the same way decoder layers
        are streamed, instead of requiring them to be permanently resident.

        Without this, a plan that assigns lm_head to the disk tier is not
        merely slow -- it is broken: _materialize_resident deliberately
        leaves disk-tier tensors on the meta device, and nothing else would
        ever load lm_head, so the forward pass would fail on a meta tensor.
        That made the lowest VRAM budgets unreachable in practice and left
        an apples-to-oranges gap in the AirLLM comparison (2.66 GB against
        AirLLM's ~1.57 GB), because AirLLM streams this tensor and we did
        not. Making it streamable is what allows a VRAM-MATCHED comparison,
        where both systems hold the same peak and only speed differs.

        The cost is honest and expected: lm_head's compressed bytes now
        cross the bus every token, so bytes-read/token rises. That tradeoff
        is the point of exposing it as a budget rather than hardcoding it.
        """
        def load_key(key: str) -> torch.Tensor:
            if self._tier.get(key) == "ram":
                cached = self._ram_cache[key]
                return (self._decode_tensor(key, cached)
                        if self.ram_tier_format == "compressed"
                        else cached.to(
                            self.device,
                            non_blocking=(key not in self._ram_cache_pageable_keys)))
            return self._load_tensor(key)

        for mod_name, pnames in self._streamed_module_params().items():
            module = self.model.get_submodule(mod_name)

            def make_pre(mn, pns):
                def pre(module, args, kwargs):
                    self.control.checkpoint()
                    for pn in pns:
                        key = mn + "." + pn
                        self._set_param(module, pn, load_key(key))
                    return None
                return pre

            def make_post(mn, pns):
                def post(module, args, kwargs, output):
                    self._synchronize_device()
                    for pn in pns:
                        self._to_meta_param(module, pn)
                    if self.empty_cache_every:
                        torch.cuda.empty_cache()
                    return output
                return post

            module.register_forward_pre_hook(make_pre(mod_name, pnames), with_kwargs=True)
            module.register_forward_hook(make_post(mod_name, pnames), with_kwargs=True)

        # A tied checkpoint stores only model.embed_tokens.weight. If that
        # tensor streams, the embedding module's post-hook replaces its live
        # parameter with a new meta placeholder immediately after lookup.
        # lm_head was tied to the *old* placeholder during startup, so without
        # this second live range it computes plausible-looking garbage. Load
        # the identical stored tensor again for the output projection; this
        # preserves the planner's low-VRAM headroom model because the full
        # embedding/head is not kept alive across decoder layers.
        embed_key = self.adapter.embedding_prefix + ".weight"
        tied_streamed = (
            bool(getattr(self.model.config, "tie_word_embeddings", False))
            and self._tier.get(embed_key) in ("disk", "ram"))
        if tied_streamed:
            head = self.adapter.output_head

            def tied_head_pre(module, args, kwargs):
                self.control.checkpoint()
                self._set_param(module, "weight", load_key(embed_key))
                return None

            def tied_head_post(module, args, kwargs, output):
                self._synchronize_device()
                self._to_meta_param(module, "weight")
                if self.empty_cache_every:
                    torch.cuda.empty_cache()
                return output

            head.register_forward_pre_hook(tied_head_pre, with_kwargs=True)
            head.register_forward_hook(tied_head_post, with_kwargs=True)

    def _kv_cache_length(self) -> int:
        """Return the cached prefix length or fail closed on an unknown cache.

        H18 mutates cache length after verification, so guessing a tensor
        dimension would turn a performance experiment into a correctness
        risk.  Current Transformers ``DynamicCache`` exposes this contract
        directly; legacy tuple caches are intentionally not accepted by the
        experimental path.
        """
        if self._kv_cache is None:
            return 0
        get_length = getattr(self._kv_cache, "get_seq_length", None)
        if not callable(get_length):
            raise RuntimeError(
                "spec_target_cache requires a cache with get_seq_length(); "
                "disable it for this Transformers/model version")
        return int(get_length())

    def _crop_kv_cache(self, maximum_length: int) -> None:
        """Crop speculative lookahead to the immutable accepted prefix."""
        if maximum_length < 0:
            raise ValueError("KV cache crop length must be non-negative")
        if self._kv_cache is None:
            raise RuntimeError("cannot crop an empty target KV cache")
        crop = getattr(self._kv_cache, "crop", None)
        if not callable(crop):
            raise RuntimeError(
                "spec_target_cache requires DynamicCache.crop(); disable it "
                "for this Transformers/model version")
        crop(maximum_length)
        actual = self._kv_cache_length()
        if actual != maximum_length:
            raise RuntimeError(
                "target KV cache crop produced length %d, expected %d" %
                (actual, maximum_length))
        self.stats.spec_cache_crops += 1

    def _spec_target_input(self, seq: torch.Tensor,
                           draft_tokens: list[int]) -> tuple[torch.Tensor, int]:
        """Build verifier input and the first draft-logit offset for H18."""
        proposals = torch.tensor(
            [draft_tokens], device=seq.device, dtype=seq.dtype)
        if not self.config.spec_target_cache or self._kv_cache is None:
            return torch.cat([seq, proposals], dim=1), seq.shape[1] - 1
        expected = seq.shape[1] - 1
        cached = self._kv_cache_length()
        if cached != expected:
            raise RuntimeError(
                "target KV cache has %d prefix tokens, expected %d before "
                "speculative verification" % (cached, expected))
        self.stats.spec_cached_prefix_tokens += cached
        # Cache owns seq[:-1]. Feed the final committed token so its logits
        # predict proposal 0, followed by the proposed lookahead tokens.
        return torch.cat([seq[:, -1:], proposals], dim=1), 0

    def set_initial_forward_kwargs(self, values: dict[str, torch.Tensor] | None) -> None:
        """Set processor outputs needed on the first multimodal forward pass."""

        self._initial_forward_kwargs = dict(values or {})

    @torch.no_grad()
    def forward_logits(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """One full forward pass. transformers drives the stack; the hooks
        stream each layer's weights in and out around it.

        compute_seconds is wall time MINUS whatever io_seconds/decode_seconds
        grew by during this call, not the raw wall time of the whole forward
        pass. The hooks that do I/O and decode run INSIDE this call (as
        pre/post hooks around each layer), so a naive wall-time measurement
        here would count that work twice -- once correctly in io_seconds/
        decode_seconds, and again, misleadingly, as if it were pure compute.
        That made compute_seconds print as roughly equal to total wall time
        regardless of how little of it was actually GPU compute.

        use_cache=True accumulates a KV cache in self._kv_cache across
        calls: input_ids should then contain ONLY the tokens generated
        since the previous call, not the whole growing sequence, and
        self-attention only recomputes over those new positions. This is
        the standard KV-cache equivalence, but bf16 matmul reduction order
        is not guaranteed bit-identical across different input shapes in
        general, so it was verified empirically (bit-exact against the
        no-cache path, tests/test_streaming_engine_gpu.py) before being
        trusted in an engine whose entire premise is bit-exactness, rather
        than assumed correct because the underlying math says it should be.

        generate_greedy uses this; generate_speculative does not, because
        its variable-length chain-and-partial-accept pattern needs cache
        TRIMMING on partial rejection, not just append, which is real
        additional correctness risk this codebase has not taken on yet --
        each speculative sweep still recomputes from the full accepted
        prefix, same as before.

        Deliberately reused by generate_speculative for chain verification:
        a causal LM computes logits at every position of an arbitrary-length
        input in one sweep by construction, so verifying a whole draft chain
        is just calling this once on [seq, draft_tokens] and slicing the
        result -- no separate batched-verification path is needed.
        """
        io0, decode0 = self.stats.io_seconds, self.stats.decode_seconds
        t0 = time.perf_counter()
        extra = self._initial_forward_kwargs if self._kv_cache is None else {}
        if use_cache:
            out = self.model(input_ids=input_ids, use_cache=True,
                             past_key_values=self._kv_cache, **extra)
            self._kv_cache = out.past_key_values
        else:
            out = self.model(input_ids=input_ids, use_cache=False, **extra)
        self._synchronize_device()
        wall = time.perf_counter() - t0
        io_delta = self.stats.io_seconds - io0
        decode_delta = self.stats.decode_seconds - decode0
        self.stats.compute_seconds += max(0.0, wall - io_delta - decode_delta)
        return out.logits

    @torch.no_grad()
    def forward_certified_argmax(self, input_ids: torch.Tensor,
                                 use_cache: bool = False) -> int:
        """Greedy-only certified MIPS head with an exact full-head fallback."""
        if self.config.lm_head_policy != "certified_mips":
            raise RuntimeError("certified argmax requested without its config")
        if input_ids.shape[0] != 1:
            raise ValueError("certified_mips currently supports batch size 1")
        from .certified_mips import MIPSIndex, certified_argmax

        io0, decode0 = self.stats.io_seconds, self.stats.decode_seconds
        t0 = time.perf_counter()
        if use_cache:
            out = self.adapter.language_model(input_ids=input_ids, use_cache=True,
                                   past_key_values=self._kv_cache)
            self._kv_cache = out.past_key_values
        else:
            out = self.adapter.language_model(input_ids=input_ids, use_cache=False)
        hidden = out.last_hidden_state[0, -1]
        weight = self.adapter.output_head.weight
        if self._certified_mips_index is None:
            self._certified_mips_index = MIPSIndex.build(weight)

        def full_fallback():
            return torch.nn.functional.linear(hidden, weight).argmax()

        result = certified_argmax(hidden, weight, self._certified_mips_index,
                                  fallback=full_fallback)
        self._synchronize_device()
        wall = time.perf_counter() - t0
        self.stats.compute_seconds += max(
            0.0, wall - (self.stats.io_seconds - io0) - (self.stats.decode_seconds - decode0))
        self.stats.mips_rows_evaluated += result.rows_evaluated
        if result.certified:
            self.stats.mips_certified += 1
            self.stats.mips_rows_pruned += max(
                0, self._certified_mips_index.rows - result.rows_evaluated)
        else:
            self.stats.mips_fallbacks += 1
        return result.index

    def _install_chunked_lm_head(self) -> None:
        """Compute logits in row blocks instead of materializing lm_head.

        This is what removes the engine's VRAM floor. Peak VRAM for a
        streaming engine is bounded below by the largest tensor it must
        hold at once, and on a 14B that is lm_head at 1.556 GB -- so no
        budget under ~1.7 GB was expressible, which is also exactly where
        AirLLM sits. Both systems were pinned to the same floor by the same
        tensor.

        But logits are a concatenation over output rows, with no
        interaction between row blocks:

            logits[..., a:b] = x @ W[a:b].T

        so the projection can be computed a block at a time with only one
        block's weights live. At lm_head_slice_rows=8192 that is ~84 MB
        instead of 1.556 GB.

        The compressed bytes are still read once per token (same disk
        traffic as before -- this trades no I/O, only peak memory) and
        decoded in blocks via compressed_store.decompress_rows_gpu, which
        is possible for free because chunks were already independently
        decodable. Bit-exactness is unaffected and asserted by tests: the
        same weights are used in the same matmul, just in pieces.
        """
        step = self._chunked_head_rows(self.config)
        if not step:
            return
        from .compressed_store import decompress_rows_gpu

        key = self.CHUNKED_HEAD_KEY
        meta = self.manifest["tensors"][key]
        rows = int(meta["shape"][0])
        comp_bytes = int(meta["comp_bytes"])

        def forward(x: torch.Tensor) -> torch.Tensor:
            self.control.checkpoint()
            arrays, io_s, read_calls, extent_bytes = self._read_tensor_arrays(key)
            self.stats.io_seconds += io_s
            self.stats.bytes_read += comp_bytes
            self.stats.storage_read_calls += read_calls
            self.stats.storage_extent_bytes += extent_bytes
            layer = self._compressed_layer(key, arrays)

            parts = []
            for r0 in range(0, rows, step):
                t1 = time.perf_counter()
                W = decompress_rows_gpu(layer, r0, min(r0 + step, rows),
                                        device=self.device)
                self.stats.decode_seconds += time.perf_counter() - t1
                parts.append(torch.nn.functional.linear(x, W))
                del W
            return torch.cat(parts, dim=-1)

        self.adapter.output_head.forward = forward

    @contextlib.contextmanager
    def _truncated_to(self, n_layers: int):
        """Temporarily swap self.model.model.layers for a prefix of itself.

        Slicing an nn.ModuleList returns a NEW ModuleList holding the SAME
        layer module objects, not copies -- so each layer's forward pre/post
        hooks (registered once in _install_hooks, attached to the module
        objects, not to the list container) stay attached and keep
        streaming/freeing weights exactly as normal for whichever layers are
        in the slice. This is what lets draft_self_logits reuse forward_logits
        (and therefore the library's own decoder forward -- rotary embeddings,
        causal mask, cache handling) UNCHANGED instead of hand-rolling a
        shortened forward pass, which is exactly the mistake that produced
        wrong logits (a silently-omitted causal mask) the one time this
        codebase tried it -- see _install_hooks's docstring.
        """
        full = self.adapter.layers
        self.adapter.layers = full[:n_layers]
        try:
            yield
        finally:
            self.adapter.layers = full

    @torch.no_grad()
    def draft_self_logits(self, input_ids: torch.Tensor, exit_layer: int) -> torch.Tensor:
        """Early-exit forward for self-speculative drafting
        (the archived adaptive-speculation proposal mechanism A): embeddings -> layers
        [0, exit_layer) -> model.norm -> lm_head, reusing the model's
        existing final norm and output head as the exit head. No new
        parameters, nothing trained -- the draft is literally the target
        model, run shallow, which is why it should agree with the target
        more often than an unrelated small model does (LayerSkip,
        arXiv:2404.16710).

        use_cache is deliberately NOT threaded through here: the draft's
        k sequential passes over layers [0, exit_layer) would each write KV
        entries for those layers, and rejected draft tokens must not leave
        cache entries the verification pass could see. v1 accepts the
        recompute cost of use_cache=False rather than take on that
        correctness risk -- see EngineConfig.draft_mode's docstring.
        """
        if not (1 <= exit_layer < self.n_layers):
            raise ValueError(
                "exit_layer must be in [1, %d) for this %d-layer model, got %d"
                % (self.n_layers, self.n_layers, exit_layer))
        with self._truncated_to(exit_layer):
            return self.forward_logits(input_ids, use_cache=False)

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int,
                        use_cache: bool = True, on_token=None,
                        stop_token_ids=()) -> torch.Tensor:
        """use_cache=True (the default): only the newly generated token is
        fed to forward_logits on every step after the first, with earlier
        positions served from a growing KV cache instead of being
        recomputed every step -- turns causal self-attention's O(seq_len^2)
        total cost back to O(seq_len). On THIS engine specifically, most
        wall time is I/O-bound weight streaming rather than attention
        compute (measured ~95%+ at short sequence lengths), so the win is
        small for short generations and grows with response length, where
        attention's quadratic term would otherwise eventually dominate even
        an I/O-bound system. use_cache=False reproduces the original
        full-recompute-every-step behaviour exactly, for comparison.

        on_token: optional callable(token_id: int) invoked right after each
        new token is generated -- this is what lets the FastAPI server
        stream tokens over SSE without duplicating this loop's logic in the
        server layer.

        stop_token_ids: generation stops early (before max_new_tokens) the
        first time a generated token's id is in this collection.
        """
        seq = input_ids
        self._tok_total = max_new_tokens
        self._gen_t0 = time.perf_counter()
        self._kv_cache = None
        step_input = input_ids
        for t in range(max_new_tokens):
            self._tok_i = t
            self.control.checkpoint()
            if self.config.lm_head_policy == "certified_mips":
                tok = self.forward_certified_argmax(step_input, use_cache=use_cache)
                nxt = torch.tensor([[tok]], device=self.device, dtype=input_ids.dtype)
            else:
                logits = self.forward_logits(step_input, use_cache=use_cache)
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
            step_input = nxt if use_cache else seq
            tok_id = int(nxt.item())
            if on_token is not None:
                on_token(tok_id)
            if tok_id in stop_token_ids:
                break
        return seq

    @torch.no_grad()
    @torch.no_grad()
    def measure_candidate_sweep_latency(
            self, input_ids: torch.Tensor, candidate_counts: list[int]) -> list[dict]:
        """H19 (Candidate-Amortization Hypothesis, see afterimage/experiments.py):
        how does one target verification sweep's wall-clock cost scale with
        the number of already-known candidate positions it verifies?

        Deliberately does NOT generate real speculative trees or run a
        draft model. The synthetic candidate tokens' VALUES are irrelevant
        here, only their COUNT is: forward_logits' cost for this engine is
        dominated by streaming and decompressing the target's own weights
        (io_seconds/decode_seconds within that same call), not by the
        marginal compute of a few more sequence positions -- so this reuses
        the real forward_logits() verification path (the same one
        generate_speculative's own target sweep calls) rather than a
        separate synthetic microbenchmark, and the candidate positions are
        just the prompt's own last token repeated, a valid in-vocabulary
        sequence with nothing case-specific about its content.

        The prompt (input_ids) is held fixed across every candidate count
        in one call, so the reported latency is not comparable in absolute
        terms across different prompts -- only the SHAPE of latency as a
        function of candidate_counts is the measurement this hypothesis
        cares about: the point past which it stops growing roughly flat,
        this project's own methodology review calls the candidate
        parallelism knee (N_free / spec_parallel_knee), which should set
        every tree-based speculation strategy's node budget from real
        measurement instead of copying a paper's number tuned for
        different hardware.
        """
        if not candidate_counts:
            raise ValueError("candidate_counts must be non-empty")
        for n in candidate_counts:
            if n < 1:
                raise ValueError("candidate_counts values must be >= 1, got %d" % n)
        results = []
        pad_token = input_ids[:, -1:]
        for n in candidate_counts:
            candidates = pad_token.expand(-1, n)
            probe_input = torch.cat([input_ids, candidates], dim=1)
            self.stats.reset()
            self._synchronize_device()
            t0 = time.perf_counter()
            self.forward_logits(probe_input, use_cache=False)
            self._synchronize_device()
            wall = time.perf_counter() - t0
            results.append({
                "candidate_positions": n,
                "verification_sweep_seconds": wall,
                "io_seconds": self.stats.io_seconds,
                "decode_seconds": self.stats.decode_seconds,
                "compute_seconds": self.stats.compute_seconds,
                "bytes_read": self.stats.bytes_read,
            })
        return results

    def generate_speculative(self, input_ids: torch.Tensor, max_new_tokens: int,
                             draft_model, k: int = 8, temperature: float = 1.0,
                             generator: torch.Generator | None = None) -> torch.Tensor:
        """Draft/verify speculative decoding.

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
        self._kv_cache = None
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

            target_input, base = self._spec_target_input(seq, draft_tokens)
            target_logits = self.forward_logits(
                target_input, use_cache=self.config.spec_target_cache)
            target_probs = [
                torch.softmax(target_logits[0, base + i, :] / temperature, dim=-1)
                for i in range(step_k)
            ]
            bonus_probs = torch.softmax(
                target_logits[0, base + step_k, :] / temperature, dim=-1)

            accepted, n_from_draft = speculative_sample_step(
                draft_probs, target_probs, draft_tokens, bonus_probs, generator)

            seq = torch.cat([seq, torch.tensor([accepted], device=seq.device)], dim=1)
            if self.config.spec_target_cache:
                self._crop_kv_cache(seq.shape[1] - 1)
            n_generated += len(accepted)
            self._tok_i = min(n_generated, max_new_tokens)
            self.stats.spec_sweeps += 1
            self.stats.spec_accepted_tokens += n_from_draft

        return seq[:, :input_ids.shape[1] + max_new_tokens]

    @torch.no_grad()
    def generate_adaptive(self, input_ids: torch.Tensor, max_new_tokens: int,
                          draft_model=None, temperature: float = 1.0,
                          generator: torch.Generator | None = None,
                          on_token=None, stop_token_ids=()):
        """generate_speculative, but with the two things
        the archived adaptive-speculation proposal proposes making adaptive:

          - EngineConfig.draft_mode="self": draft with THIS model's own
            first draft_exit_layer layers (draft_self_logits) instead of a
            separate resident draft_model -- mechanism A.
          - EngineConfig.spec_k_policy != "fixed": a SpecPolicy
            (runtime/spec_policy.py) chooses the draft chain length k before
            every sweep from the previous sweep's acceptance, instead of a
            constant -- mechanism B.

        Both default off (draft_mode="none" refuses to run this method;
        spec_k_policy="fixed" reduces to generate_speculative's constant-k
        behaviour). Correctness is UNCHANGED from generate_speculative: this
        calls the identical speculative_sample_step, so the exact-target-
        distribution guarantee holds regardless of k, policy, or draft
        source -- see runtime/verify.py and runtime/spec_policy.py's module
        docstrings. At temperature<=0 specifically, verify.temperature_probs
        makes this provably reproduce generate_greedy's argmax sequence
        token-for-token, for ANY draft_mode/k/policy -- see its docstring
        and the archived adaptive test plan §3.

        Returns (sequence, policy) -- the policy is returned (not just its
        k) so a caller can inspect state_dict() / log() without needing a
        second entry point.

        on_token: optional callable(token_id: int), invoked once per newly
        accepted token as each sweep's accepted prefix is committed (not
        one call per drafted token -- a sweep can commit several at once,
        which is the real unit of progress here). Same purpose as
        generate_greedy's on_token: lets the server stream over SSE without
        duplicating this loop.

        stop_token_ids: generation stops the first time an accepted token's
        id is in this collection, truncating the returned sequence right
        after it (any later tokens in that same sweep's accepted prefix are
        discarded, matching generate_greedy's stop-at-first-match semantics).
        """
        from .spec_policy import SweepRecord, _prob_entropy, build_policy
        from .verify import sample_categorical, speculative_sample_step, temperature_probs

        cfg = self.config
        if cfg.draft_mode == "none":
            raise ValueError(
                "generate_adaptive requires EngineConfig.draft_mode='model' "
                "or 'self' -- 'none' is generate_greedy/generate_speculative's "
                "regime, not this method's")
        if cfg.draft_mode == "model" and draft_model is None:
            raise ValueError("draft_mode='model' requires a draft_model argument")

        policy = build_policy(cfg.spec_k_policy, cfg.spec_k)
        if cfg.spec_policy_state:
            policy.load(cfg.spec_policy_state)

        seq = input_ids
        self._tok_total = max_new_tokens
        self._gen_t0 = time.perf_counter()
        self._kv_cache = None
        n_generated = 0

        while n_generated < max_new_tokens:
            sweep_t0 = time.perf_counter()
            k_request = min(policy.choose_k(), max_new_tokens - n_generated)

            draft_seq = seq
            draft_tokens: list[int] = []
            draft_probs: list[torch.Tensor] = []
            draft_t0 = time.perf_counter()
            for _ in range(k_request):
                if cfg.draft_mode == "self":
                    dlogits = self.draft_self_logits(draft_seq, cfg.draft_exit_layer)[:, -1, :]
                else:
                    dlogits = draft_model(input_ids=draft_seq).logits[:, -1, :]
                probs = temperature_probs(dlogits[0], temperature)
                # Per-position policies inspect the distribution that would
                # produce the next proposal and can stop before paying for a
                # token they expect not to survive. Fixed/gamma return False.
                if policy.should_stop(draft_probs + [probs]):
                    break
                tok = sample_categorical(probs, generator)
                draft_tokens.append(tok)
                draft_probs.append(probs)
                draft_seq = torch.cat(
                    [draft_seq, torch.tensor([[tok]], device=draft_seq.device)], dim=1)
            draft_seconds = time.perf_counter() - draft_t0

            step_k = len(draft_tokens)
            target_input, base = self._spec_target_input(seq, draft_tokens)
            target_t0 = time.perf_counter()
            target_logits = self.forward_logits(
                target_input, use_cache=cfg.spec_target_cache)
            target_seconds = time.perf_counter() - target_t0
            target_probs = [
                temperature_probs(target_logits[0, base + i, :], temperature)
                for i in range(step_k)
            ]
            bonus_probs = temperature_probs(target_logits[0, base + step_k, :], temperature)

            accepted, n_from_draft = speculative_sample_step(
                draft_probs, target_probs, draft_tokens, bonus_probs, generator)

            stopped_at = None
            for i, tok_id in enumerate(accepted):
                if tok_id in stop_token_ids:
                    stopped_at = i
                    break
            if stopped_at is not None:
                accepted = accepted[:stopped_at + 1]

            seq = torch.cat([seq, torch.tensor([accepted], device=seq.device)], dim=1)
            if cfg.spec_target_cache:
                self._crop_kv_cache(seq.shape[1] - 1)
            n_generated += len(accepted)
            self._tok_i = min(n_generated, max_new_tokens)
            self.stats.spec_sweeps += 1
            self.stats.spec_accepted_tokens += n_from_draft
            if on_token is not None:
                for tok_id in accepted:
                    on_token(int(tok_id))

            if cfg.spec_policy_learn:
                policy.update(SweepRecord(
                    step_k, n_from_draft, time.perf_counter() - sweep_t0,
                    draft_confidences=tuple(float(p.max()) for p in draft_probs),
                    draft_entropies=tuple(_prob_entropy(p) for p in draft_probs),
                    draft_seconds=draft_seconds, target_seconds=target_seconds))

            if stopped_at is not None:
                break

        if cfg.spec_policy_state and cfg.spec_policy_learn:
            policy.save(cfg.spec_policy_state)

        return seq[:, :input_ids.shape[1] + min(n_generated, max_new_tokens)], policy


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
