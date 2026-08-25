"""Command-line entry point for Afterimage.

    afterimage doctor                     hardware + disk-speed diagnosis
    afterimage compress MODEL              build a compressed store, with progress
    afterimage pull MODEL --store-repo R   fetch an already-compressed store instead
    afterimage verify MODEL                check a store's checksums
    afterimage run MODEL PROMPT            one-off generation, streamed
    afterimage serve                       launch the FastAPI server + web UI
    afterimage bench MODEL                 head-to-head vs AirLLM (needs airllm installed)

Registered as the `afterimage` console script (see pyproject.toml
[project.scripts]) so `pip install -e .` makes this runnable as a plain
command, not just `python -m afterimage.cli`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import sys
import time

# AFTERIMAGE_STORE_ROOT lets the Docker image (and anyone else) point stores
# at a mounted volume instead of the container's ephemeral home directory --
# see Dockerfile, which sets this to /data/stores under the /data VOLUME.
DEFAULT_STORE_ROOT = pathlib.Path(
    os.environ.get("AFTERIMAGE_STORE_ROOT", str(pathlib.Path.home() / ".afterimage" / "stores")))


def _store_dir_for(model_id: str, store_root: pathlib.Path | None = None) -> pathlib.Path:
    root = store_root or DEFAULT_STORE_ROOT
    return root / model_id.replace("/", "__")


# -- doctor ------------------------------------------------------------

def _detect_gpu() -> dict:
    import subprocess

    info = {"vendor": "none", "name": None, "vram_gb": None}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            name, mem_mb = out.stdout.strip().splitlines()[0].split(",")
            info.update(vendor="nvidia", name=name.strip(), vram_gb=round(float(mem_mb) / 1024, 2))
            return info
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        out = subprocess.run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            info.update(vendor="amd", name="(see rocm-smi output)")
            return info
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def _detect_ram_gb() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb * 1024 / 1e9, 1)
        except FileNotFoundError:
            pass
    return None


def _benchmark_disk_read_mb_s(target_dir: pathlib.Path, size_mb: int = 256):
    """Measures real sequential read throughput of the disk that will
    actually hold compressed stores, instead of quoting the reference
    machine's number at everyone.

    Streaming I/O dominates wall time on this engine, and it's the cost
    that varies most between machines: the reference RTX 3080 Laptop's own
    bounded suite measured 506-917 MB/s across its methods, and a desktop
    NVMe commonly does 3,000-7,000. A user on a much faster (or slower)
    disk than the reference machine has no way to know that from the
    README's table alone.

    Best-effort cache drop via posix_fadvise(DONTNEED) where available
    (Linux/WSL2); there's no equivalent syscall this project can call on
    Windows/macOS, so the read there may partly hit the OS page cache and
    read faster than a genuinely cold read would -- callers are told which
    case they got so they don't take a cache-assisted number as gospel.

    Returns (mb_per_s, cache_dropped) or None if the write/read/cleanup
    failed for any reason (read-only filesystem, disk full, permissions) --
    this must never be the reason `doctor` exits non-zero.
    """
    target_dir = pathlib.Path(target_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        probe = target_dir / ".afterimage-disk-probe.tmp"
        chunk = os.urandom(4 * 1024 * 1024)
        total_bytes = size_mb * 1024 * 1024
        try:
            with open(probe, "wb") as f:
                written = 0
                while written < total_bytes:
                    f.write(chunk)
                    written += len(chunk)
                f.flush()
                os.fsync(f.fileno())

            cache_dropped = False
            with open(probe, "rb") as f:
                if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    cache_dropped = True
                t0 = time.perf_counter()
                read_bytes = 0
                while True:
                    data = f.read(4 * 1024 * 1024)
                    if not data:
                        break
                    read_bytes += len(data)
                elapsed = max(time.perf_counter() - t0, 1e-6)
            return (read_bytes / 1e6) / elapsed, cache_dropped
        finally:
            probe.unlink(missing_ok=True)
    except OSError:
        return None


def cmd_doctor(args: argparse.Namespace) -> int:
    import torch

    print("Afterimage hardware diagnosis")
    print("=" * 60)

    gpu = _detect_gpu()
    print("GPU vendor       : %s" % gpu["vendor"])
    if gpu["name"]:
        print("GPU              : %s" % gpu["name"])
    if gpu["vram_gb"]:
        print("VRAM             : %.2f GB" % gpu["vram_gb"])

    ram_gb = _detect_ram_gb()
    print("System RAM       : %s" % (("%.1f GB" % ram_gb) if ram_gb else "unknown"))

    print("torch            : %s" % torch.__version__)
    cuda_ok = torch.cuda.is_available()
    print("CUDA available   : %s" % cuda_ok)
    if cuda_ok:
        print("CUDA device      : %s" % torch.cuda.get_device_name(0))

    try:
        import triton  # noqa: F401
        print("triton           : available")
    except ImportError:
        print("triton           : NOT installed -- GPU decode kernels will not run")

    if gpu["vendor"] == "amd":
        print()
        print("NOTE: ROCm/AMD support is built against a device abstraction but has")
        print("      not been run on real AMD hardware by this project -- treat it as")
        print("      untested, not unsupported. block_chunks (Triton kernel tuning) was")
        print("      tuned for NVIDIA's 32-wide warp; AMD's 64-wide wavefront likely")
        print("      wants a different value -- see the archived master plan.")

    print()
    if args.skip_disk_check:
        print("Disk read speed  : skipped (--skip-disk-check)")
    else:
        from afterimage.reference import MEASURED_REFERENCE
        result = _benchmark_disk_read_mb_s(DEFAULT_STORE_ROOT)
        if result is None:
            print("Disk read speed  : couldn't measure (%s isn't writable)" % DEFAULT_STORE_ROOT)
        else:
            mb_s, cache_dropped = result
            print("Disk read speed  : %.0f MB/s%s" %
                 (mb_s, "" if cache_dropped else " (may be optimistic -- OS page cache "
                                                  "couldn't be dropped on this platform)"))
            ref = MEASURED_REFERENCE
            store_gb = ref["compressed_gb_per_b_params"] * ref["params_b"]
            read_fraction = ref["min_memory_store_fraction_read_per_token"]
            s_per_token = (store_gb * read_fraction * 1000) / mb_s
            print("                   at this speed, a 14B model's minimum-memory "
                  "profile (worst case --")
            print("                   most of the store re-read every token) would run "
                  "roughly %.0fs/token." % s_per_token)
            print("                   Residency (--vram-budget-gb) and speculation "
                  "(--draft-model) both read")
            print("                   far less per token than this -- see README's "
                  "benchmark table for real ratios.")

    print()
    print("Compressed stores in %s:" % DEFAULT_STORE_ROOT)
    stores = []
    if DEFAULT_STORE_ROOT.exists():
        stores = sorted(p for p in DEFAULT_STORE_ROOT.iterdir() if (p / "manifest.json").exists())
        for s in stores:
            man = json.loads((s / "manifest.json").read_text())
            print("  %-40s %6.2f GB (%.2fx)" % (
                man.get("model_id", s.name), man["total_comp_bytes"] / 1e9, man["ratio"]))
    if not stores:
        print("  (none yet -- run `afterimage compress <model>`)")

    ok = cuda_ok or gpu["vendor"] == "amd"
    print()
    print("Overall: %s" % ("ready" if ok else "no usable GPU found -- CPU fallback only, will be slow"))

    print()
    if not stores:
        print("Next: nothing compressed yet. Try the ~10-minute quickstart first:")
        print("  afterimage quickstart")
    elif ok:
        print("Next: run something -- profiles are measured operating points, not guesses:")
        print("  afterimage run %s \"your prompt\" --profile fast" % stores[0].name.replace("__", "/"))
        print("or let it pick one from your detected hardware:")
        print("  afterimage run %s \"your prompt\" --auto" % stores[0].name.replace("__", "/"))
    return 0 if ok else 1


# -- compress ------------------------------------------------------------

# Measured, not assumed -- see docs/RESULTS_LOG.md and README's "The store"
# section: bf16 exponent-only Huffman coding lands at 1.453x on real
# checkpoints. Used only to *estimate* disk headroom before a real pass
# measures the real ratio for that specific model.
MEASURED_COMPRESSION_RATIO = 1.453


def _estimate_download_bytes(model_id: str) -> int | None:
    """Estimate the Transformers weight set, excluding duplicate exports.

    Some repositories publish both ``model-*.safetensors`` (the files named
    by ``model.safetensors.index.json``) and a second, full
    ``consolidated*.safetensors`` export.  Summing every safetensors sibling
    double-counts such checkpoints and can make the disk preflight report
    twice the real requirement.  Prefer the index's exact shard set; only
    fall back to conventional non-consolidated files when no index exists.

    Returns None on any failure (offline, gated repo, private model, network
    error, or incomplete size metadata) -- callers must treat that as
    "unknown", never as "zero bytes needed".
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi().model_info(model_id, files_metadata=True)
        siblings = {s.rfilename: s.size for s in info.siblings}
        index_name = "model.safetensors.index.json"
        if index_name in siblings:
            index_path = pathlib.Path(hf_hub_download(model_id, index_name))
            index = json.loads(index_path.read_text(encoding="utf-8"))
            filenames = set(index.get("weight_map", {}).values())
        elif "model.safetensors" in siblings:
            filenames = {"model.safetensors"}
        else:
            filenames = {
                name for name in siblings
                if name.endswith(".safetensors")
                and not pathlib.PurePosixPath(name).name.startswith("consolidated")
            }
        sizes = [siblings.get(name) for name in filenames]
        if not filenames or any(size is None or size <= 0 for size in sizes):
            return None
        total = sum(sizes)
        return total or None
    except Exception:
        return None


def _disk_preflight(model_id: str, out_dir: pathlib.Path, *, assume_yes: bool) -> bool:
    """Best-effort space check. A failed size lookup degrades to a warning
    and returns True -- this must never be the reason a real compress run
    refuses to start, only a way to catch the common case (a laptop with
    much less free space than a 14B-class model needs) before ~40 minutes
    of download discover it instead."""
    download_bytes = _estimate_download_bytes(model_id)
    if download_bytes is None:
        print("[preflight] could not look up %s's size ahead of time (offline, "
              "gated, or a network error) -- skipping the disk-space check"
              % model_id, file=sys.stderr)
        return True

    store_bytes = int(download_bytes / MEASURED_COMPRESSION_RATIO)
    needed = download_bytes + store_bytes  # both live on disk at once mid-pass
    print("[preflight] %s: ~%.1f GB to download, ~%.1f GB compressed store "
          "(~%.1f GB needed on disk at once, both present until the pass finishes)"
          % (model_id, download_bytes / 1e9, store_bytes / 1e9, needed / 1e9),
          file=sys.stderr)

    check_dir = out_dir.parent if out_dir.parent.exists() else pathlib.Path.home()
    try:
        free = shutil.disk_usage(check_dir).free
    except OSError:
        return True

    if free < needed:
        print("[preflight] only %.1f GB free at %s -- this will likely run out of "
              "disk partway through the pass" % (free / 1e9, check_dir), file=sys.stderr)
        if assume_yes:
            return True
        try:
            answer = input("Continue anyway? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        return answer in ("y", "yes")
    return True


def cmd_compress(args: argparse.Namespace) -> int:
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import compress_model_to_disk

    out_dir = pathlib.Path(args.out) if args.out else _store_dir_for(args.model, )

    if args.dry_run:
        download_bytes = _estimate_download_bytes(args.model)
        if download_bytes is None:
            print("Could not look up %s's size (offline, gated, or a network "
                  "error)." % args.model)
            return 1
        store_bytes = int(download_bytes / MEASURED_COMPRESSION_RATIO)
        print("%s (dry run)" % args.model)
        print("  download   : ~%.1f GB" % (download_bytes / 1e9))
        print("  store      : ~%.1f GB (at the measured %.3fx ratio)"
              % (store_bytes / 1e9, MEASURED_COMPRESSION_RATIO))
        print("  peak disk  : ~%.1f GB (both present at once mid-pass)"
              % ((download_bytes + store_bytes) / 1e9))
        print("  store path : %s" % out_dir)
        return 0

    if not _disk_preflight(args.model, out_dir, assume_yes=args.yes):
        print("Aborted.", file=sys.stderr)
        return 1

    print("Compressing %s -> %s" % (args.model, out_dir))
    cfg = EngineConfig(chunk_size=args.chunk_size, quantize=args.quantize)
    man = compress_model_to_disk(args.model, out_dir, config=cfg,
                                 progress_every=args.progress_every,
                                 max_workers=args.workers)
    print("=" * 60)
    print("ORIGINAL  : %.3f GB" % (man["total_orig_bytes"] / 1e9))
    print("COMPRESSED: %.3f GB" % (man["total_comp_bytes"] / 1e9))
    print("RATIO     : %.3fx" % man["ratio"])
    print("Store     : %s" % out_dir)
    return 0


# -- verify ------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> int:
    """Reads every blob in a store's weights.bin once and checks it
    against the manifest's stored CRC32s -- binstore.verify_store() had
    anticipated this command in its own docstring since before it existed
    ("meant to be run explicitly (a CLI `afterimage verify` command...)"),
    which this closes. The one time this matters most: right after
    `afterimage pull` fetches a store you didn't compress yourself, where
    silent corruption in transit would otherwise surface as a mysterious
    decode error deep into a run instead of a clear answer up front.
    """
    from afterimage.runtime.binstore import verify_store

    store_dir = pathlib.Path(args.store) if args.store else _store_dir_for(args.model)
    if not (store_dir / "manifest.json").exists():
        print("No compressed store at %s -- run `afterimage compress %s` or "
              "`afterimage pull %s` first" % (store_dir, args.model, args.model),
              file=sys.stderr)
        return 1

    print("Verifying %s ..." % store_dir)
    ok, bad_keys = verify_store(store_dir)
    if ok:
        print("OK -- every tensor's checksum matched.")
        return 0
    print("FAILED -- %d tensor(s) don't match their stored checksum:" % len(bad_keys),
         file=sys.stderr)
    for key in bad_keys[:20]:
        print("  %s" % key, file=sys.stderr)
    if len(bad_keys) > 20:
        print("  ... and %d more" % (len(bad_keys) - 20), file=sys.stderr)
    print("The store is corrupt. Re-compress or re-pull it -- do not run against it.",
         file=sys.stderr)
    return 1


# -- pull ----------------------------------------------------------------

def cmd_pull(args: argparse.Namespace) -> int:
    """Fetches an already-compressed store from a Hugging Face repo instead
    of downloading the original checkpoint and compressing it locally.

    A compressed store is nothing more than manifest.json + weights.bin
    (see compress_model_to_disk); nothing stops someone publishing one and
    everyone else just downloading it, which is strictly better than
    compressing locally on every axis: less to transfer (the compressed
    store is smaller than the original checkpoint), no compression pass at
    all, and the original checkpoint never has to exist on disk, so peak
    disk usage drops too. This is the client half of that; it does not
    publish anything, and no store is hosted anywhere by this project yet
    -- point --store-repo at one you (or someone) has actually published.
    """
    from huggingface_hub import hf_hub_download

    store_dir = pathlib.Path(args.store) if args.store else _store_dir_for(args.model)
    store_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching a pre-compressed store for %s from %s (%s repo) ..."
         % (args.model, args.store_repo, args.repo_type))
    try:
        manifest_path = hf_hub_download(args.store_repo, "manifest.json",
                                        repo_type=args.repo_type)
        weights_path = hf_hub_download(args.store_repo, "weights.bin",
                                       repo_type=args.repo_type)
    except Exception as e:
        print("Could not fetch manifest.json + weights.bin from %s: %s"
             % (args.store_repo, e), file=sys.stderr)
        print("This only works against a repo that actually hosts an Afterimage "
              "store (a manifest.json and weights.bin at its root) -- if you meant "
              "to compress %s yourself instead, use `afterimage compress %s`."
              % (args.model, args.model), file=sys.stderr)
        return 1

    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    if manifest.get("model_id") != args.model:
        print("NOTE: this store's manifest says model_id=%r, not %r -- using it "
              "anyway since you pointed --store-repo at it explicitly, but this is "
              "either a differently-named checkpoint or the wrong repo."
              % (manifest.get("model_id"), args.model), file=sys.stderr)

    shutil.copy(manifest_path, store_dir / "manifest.json")
    shutil.copy(weights_path, store_dir / "weights.bin")

    print("Store fetched: %.2f GB (%.3fx ratio, per the manifest)"
         % (manifest.get("total_comp_bytes", 0) / 1e9, manifest.get("ratio", 0.0)))
    if args.verify:
        from afterimage.runtime.binstore import verify_store
        print("Verifying checksums ...")
        ok, bad_keys = verify_store(store_dir)
        if not ok:
            print("FAILED -- %d tensor(s) don't match their checksum after download; "
                 "the transfer was corrupted. Delete %s and try again."
                 % (len(bad_keys), store_dir), file=sys.stderr)
            return 1
        print("OK -- every tensor's checksum matched.")
    print("Run: afterimage run %s \"...\" --auto" % args.model)
    return 0


# -- run -------------------------------------------------------------------

# Measured operating points from the README's benchmark table (RTX 3080
# Laptop, 8 GB, Qwen3-14B). vram_budget_gb of None means "no residency
# planning" -- the exact minimum-memory floor, ~1.7 GB on that model.
RUN_PROFILES = {
    "min-memory": {"vram_budget_gb": None, "draft_model": None},
    "balanced": {"vram_budget_gb": 4.0, "draft_model": None},
    "fast": {"vram_budget_gb": 4.0, "draft_model": "Qwen/Qwen3-0.6B"},
}


def _resolve_run_profile(args: argparse.Namespace) -> None:
    """Applies --profile or --auto onto args in place, in the core fields
    (vram_budget_gb, draft_model) only. An explicit flag the user actually
    passed always wins -- this only fills in values still at their argparse
    default (None)."""
    name = args.profile
    if args.auto and name is None:
        vram_gb = _detect_gpu().get("vram_gb")
        if vram_gb is None:
            name = "min-memory"
            reason = "no GPU detected"
        elif vram_gb >= 6.0:
            name = "fast"
            reason = "%.1f GB VRAM detected" % vram_gb
        elif vram_gb >= 3.0:
            name = "balanced"
            reason = "%.1f GB VRAM detected" % vram_gb
        else:
            name = "min-memory"
            reason = "%.1f GB VRAM detected -- too little to spend on residency" % vram_gb
        print("[auto] picked --profile %s (%s)" % (name, reason), file=sys.stderr)
    if name is None:
        return
    preset = RUN_PROFILES[name]
    if args.vram_budget_gb is None:
        args.vram_budget_gb = preset["vram_budget_gb"]
    if args.draft_model is None:
        args.draft_model = preset["draft_model"]
    print("[profile %s] vram_budget_gb=%s draft_model=%s (pass explicit flags to override)"
          % (name, args.vram_budget_gb, args.draft_model), file=sys.stderr)


def _render_prompt(tok, prompt: str, think: bool) -> str:
    """Applies the model's chat template, the way an instruct model expects
    to be talked to -- matching what the server (_build_prompt in
    server/app.py) and the benchmark suite (render_chat_prompt in
    bench/prompt_suite.py) already do. A raw completion string sent to an
    instruct model tends to ramble or ignore the requested format, which is
    exactly what a bare-prompt CLI run would produce.

    enable_thinking=False by default: Qwen3-family models reason inside
    <think> blocks before answering, and every one of those tokens costs
    this engine's full per-token price (9-30s at the profiles README
    documents), so a few hundred thinking tokens can turn a two-sentence
    answer into a half-hour run. --think opts back in. Older tokenizers
    that don't accept enable_thinking raise TypeError, caught the same way
    render_chat_prompt does.
    """
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if think:
        return tok.apply_chat_template(messages, **kwargs)
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def _make_stream_printer(tok):
    """Returns an on_token callback that prints newly-decoded text to
    stdout as soon as each token arrives, plus the running list of token
    ids it has seen.

    Decodes the WHOLE buffer every call, not just the new token, and prints
    only the suffix that's new: one character can span multiple tokens
    (BPE/SentencePiece byte-fallback, multi-byte UTF-8, leading-space
    markers), so decoding a token in isolation can print garbled text or
    drop a space that only resolves once the next token arrives.
    Re-decoding the whole buffer is milliseconds -- irrelevant against a
    9-30s/token engine -- and it can never drift from what a final,
    single decode() of the same ids would produce, since it always IS
    that decode, just called earlier and more often.
    """
    token_ids: list[int] = []
    state = {"printed": ""}

    def on_token(tok_id: int) -> None:
        token_ids.append(tok_id)
        text = tok.decode(token_ids, skip_special_tokens=True)
        if text.startswith(state["printed"]):
            new = text[len(state["printed"]):]
            if new:
                print(new, end="", flush=True)
            state["printed"] = text
        # else: this tokenizer's decode() isn't prefix-stable as more ids
        # are added (rare). Skip the increment rather than risk printing
        # something garbled or duplicated -- cmd_run reconciles against the
        # authoritative final decode once generation finishes.

    return on_token, state


def cmd_run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoTokenizer
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import (
        StreamingLosslessModel, load_draft_model,
    )

    _resolve_run_profile(args)

    store_dir = pathlib.Path(args.store) if args.store else _store_dir_for(args.model)
    if not (store_dir / "manifest.json").exists():
        print("No compressed store at %s -- run `afterimage compress %s` first"
              % (store_dir, args.model), file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(args.model)
    rendered = args.prompt if args.raw else _render_prompt(tok, args.prompt, args.think)
    ids = tok(rendered, return_tensors="pt").input_ids
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = ids.to(device)
    stop_token_ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()

    cfg = EngineConfig(vram_cap_gb=args.vram_cap_gb, vram_budget_gb=args.vram_budget_gb,
                       ram_budget_gb=args.ram_budget_gb, max_context=args.max_context,
                       progress=not args.quiet,
                       io_prefetch_depth=args.io_prefetch_depth,
                       storage_read_policy=args.storage_read_policy,
                       storage_extent_max_bytes=args.storage_extent_max_bytes,
                       storage_extent_max_gap_bytes=args.storage_extent_max_gap_bytes,
                       decode_slice_elems=args.decode_slice_elems,
                       ram_tier_format=args.ram_tier_format,
                       lm_head_slice_rows=args.lm_head_slice_rows,
                       placement_policy=args.placement_policy,
                       critical_path_profile=args.critical_path_profile,
                       replay_plan_state=args.replay_plan_state,
                       prefetch_policy=args.prefetch_policy,
                       io_prefetch_max_depth=args.io_prefetch_max_depth,
                       lm_head_policy=args.lm_head_policy,
                       require_pinned_ram=args.require_pinned_ram,
                       draft_mode=("model" if args.draft_model else "none"),
                       spec_k=args.spec_k,
                       spec_target_cache=args.spec_target_cache,
                       trace_events=bool(args.trace_output),
                       trace_output=args.trace_output)
    if not cfg.is_lossless:
        print("WARNING: %s" % cfg.describe(), file=sys.stderr)
    if not args.quiet and args.max_new_tokens > 0:
        eta_s_per_token = 30.0 if not cfg.vram_budget_gb else (
            9.5 if args.draft_model else 17.5)
        print("[eta] very rough estimate: ~%.0fs for %d tokens (varies a lot by "
              "hardware and prompt -- pass --stats to see the real numbers after)"
              % (eta_s_per_token * args.max_new_tokens, args.max_new_tokens),
              file=sys.stderr)
    on_token, stream_state = (None, None) if args.no_stream else _make_stream_printer(tok)
    with StreamingLosslessModel(args.model, store_dir, device=device, config=cfg) as sm:
        if args.draft_model:
            draft = load_draft_model(args.draft_model, device=device)
            seq, _policy = sm.generate_adaptive(
                ids, max_new_tokens=args.max_new_tokens, draft_model=draft,
                temperature=args.spec_temperature, on_token=on_token,
                stop_token_ids=stop_token_ids)
        else:
            seq = sm.generate_greedy(ids, max_new_tokens=args.max_new_tokens,
                                     use_cache=not args.no_kv_cache, on_token=on_token,
                                     stop_token_ids=stop_token_ids)
        text = tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True)
        if args.no_stream:
            print(text)
        elif stream_state["printed"] != text:
            # Rare: this tokenizer's decode() wasn't prefix-stable as tokens
            # accumulated (see _make_stream_printer), so what streamed to
            # the terminal doesn't quite match a single decode of the whole
            # sequence. Print the authoritative text so the visible output
            # is still correct, even though the stream stumbled getting there.
            print("\n[reconciled] %s" % text)
        else:
            print()
        if args.stats:
            print("\n--- stats ---", file=sys.stderr)
            print("io=%.2fs decode=%.2fs compute=%.2fs bytes_read=%.2fGB"
                  % (sm.stats.io_seconds, sm.stats.decode_seconds,
                     sm.stats.compute_seconds, sm.stats.bytes_read / 1e9),
                  file=sys.stderr)
            print("storage_read_calls=%d storage_extent_bytes=%.2fGB "
                  "prefetch_peak_inflight_bytes=%.2fGB"
                  % (sm.stats.storage_read_calls,
                     sm.stats.storage_extent_bytes / 1e9,
                     sm.stats.prefetch_peak_inflight_bytes / 1e9),
                  file=sys.stderr)
            if torch.cuda.is_available():
                print("peak_vram=%.2fGB" % (torch.cuda.max_memory_allocated() / 1e9),
                      file=sys.stderr)
    return 0


# -- research profiles ----------------------------------------------------

def cmd_experiments(args: argparse.Namespace) -> int:
    from afterimage.experiments import registry_payload

    payload = registry_payload()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for hypothesis in payload["hypotheses"]:
            print("%s  %s" % (hypothesis["id"], hypothesis["title"]))
            print("  candidate=%s  control=%s  metric=%s" % (
                hypothesis["candidate_profile"], hypothesis["control_profile"],
                hypothesis["primary_metric"]))
    return 0


def cmd_test_plan(args: argparse.Namespace) -> int:
    """Print the regulated evidence protocol for one or all hypotheses."""
    from afterimage.experiments import HYPOTHESES
    from afterimage.protocols import protocol_for, protocol_payload

    if args.hypothesis:
        query = args.hypothesis.lower()
        matches = [
            hypothesis_id for hypothesis_id in HYPOTHESES
            if hypothesis_id.lower() == query
            or hypothesis_id.split("-", 1)[0].lower() == query
        ]
        if len(matches) != 1:
            print("Unknown hypothesis %r" % args.hypothesis, file=sys.stderr)
            return 2
        hypothesis_id = matches[0]
        payload = {
            "hypothesis": dataclasses.asdict(HYPOTHESES[hypothesis_id]),
            "protocol": dataclasses.asdict(protocol_for(hypothesis_id)),
        }
    else:
        payload = protocol_payload()
    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.hypothesis:
        print("%s -- %s" % (hypothesis_id, payload["protocol"]["family"]))
        for stage in payload["protocol"]["stages"]:
            print("  %s %-3s %s" % (stage["id"], stage["level"], stage["purpose"]))
        print("advance: " + payload["protocol"]["advance_rule"])
        print("confirm: " + payload["protocol"]["confirmation_rule"])
    else:
        for hypothesis_id, protocol_id in sorted(
                payload["hypothesis_protocols"].items()):
            print("%s  %s" % (hypothesis_id, protocol_id))
    return 0


def cmd_pin_preflight(args: argparse.Namespace) -> int:
    """Fail-closed environment check for pinned-RAM experiments."""
    from afterimage.runtime.memory_preflight import pinned_memory_preflight

    report = pinned_memory_preflight(
        int(args.gigabytes * 1e9),
        attempt_allocation=not args.static_only)
    payload = dataclasses.asdict(report)
    if args.out:
        out = pathlib.Path(args.out)
        if out.exists():
            raise FileExistsError("refusing to overwrite immutable result: %s" % out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True),
                       encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("PASS" if report.success else "BLOCKED", report.reason)
        print("requested=%.3f GB hard_memlock=%s" % (
            report.requested_bytes / 1e9,
            ("unlimited/unknown" if report.memlock_hard_bytes is None
             else "%.3f GB" % (report.memlock_hard_bytes / 1e9))))
    return 0 if report.success else 3


def cmd_profile_trace(args: argparse.Namespace) -> int:
    from afterimage.runtime.critical_path import CriticalPathProfile, TraceRecorder

    traces = [TraceRecorder.load(path) for path in args.traces]
    profile = CriticalPathProfile.from_traces(traces)
    profile.save(args.out)
    print("Wrote %s tensors from %d traces to %s" % (
        len(profile.tensors), profile.trace_count, args.out))
    if args.manifest:
        manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
        eligible = {key for key, meta in manifest["tensors"].items()
                    if not meta.get("row_gather")}
        coverage = len(eligible & set(profile.tensors)) / max(len(eligible), 1)
        print("Placement-candidate coverage: %.1f%%" % (100.0 * coverage))
        if coverage < 0.90:
            print("Profile is below the runtime's 90% coverage gate", file=sys.stderr)
            return 2
    return 0


def cmd_optimize_residency(args: argparse.Namespace) -> int:
    """Fit a whole-set residency plan on separate calibration traces."""
    from afterimage.runtime.critical_path import TraceRecorder
    from afterimage.runtime.replay_planner import (
        optimize_extent_qubo_residency, optimize_qubo_residency,
        optimize_replay_residency,
    )

    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    traces = [TraceRecorder.load(path) for path in args.traces]
    if args.search_method == "qubo":
        plan = optimize_qubo_residency(
            manifest, traces, vram_budget_gb=args.vram_budget_gb,
            decode_slice_elems=args.decode_slice_elems,
            pairwise_candidates=args.pairwise_candidates,
            restarts=args.anneal_restarts, sweeps=args.anneal_sweeps,
            seed=args.seed)
    elif args.search_method == "extent-qubo":
        plan = optimize_extent_qubo_residency(
            manifest, traces, vram_budget_gb=args.vram_budget_gb,
            decode_slice_elems=args.decode_slice_elems,
            max_extent_bytes=args.max_extent_bytes,
            max_gap_bytes=args.max_extent_gap_bytes,
            max_tensors_per_extent=args.max_tensors_per_extent,
            pairwise_candidates=args.pairwise_candidates,
            restarts=args.anneal_restarts, sweeps=args.anneal_sweeps,
            seed=args.seed)
    else:
        plan = optimize_replay_residency(
            manifest, traces, vram_budget_gb=args.vram_budget_gb,
            decode_slice_elems=args.decode_slice_elems,
            iterations=args.iterations, population=args.population,
            elite_fraction=args.elite_fraction, seed=args.seed)
    plan.save(args.out)
    print("Wrote replay-%s plan with %d resident tensors to %s" %
          (plan.report.search_method, len(plan.vram_keys), args.out))
    print("Calibration replay: %.3fs -> %.3fs (%.3fx), %d evaluations" %
          (plan.report.baseline_s, plan.report.optimized_s,
           plan.report.predicted_speedup, plan.report.evaluations))
    return 0


# -- quickstart --------------------------------------------------------

QUICKSTART_MODEL = "Qwen/Qwen3-0.6B"
QUICKSTART_PROMPT = "The capital of France is"


def cmd_quickstart(args: argparse.Namespace) -> int:
    """Compress + run a small model end to end, so a new install proves
    itself in minutes and ~2 GB instead of the ~70 minutes and ~50 GB a
    14B-class model costs. Not a benchmark -- Qwen3-0.6B is small enough
    that streaming/compression overhead barely matters; it exists to prove
    the pipeline works on this machine before committing to a real model."""
    import time

    import torch
    from transformers import AutoTokenizer
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import (
        StreamingLosslessModel, compress_model_to_disk,
    )

    model = args.model or QUICKSTART_MODEL
    store_dir = _store_dir_for(model)
    t0 = time.perf_counter()

    if (store_dir / "manifest.json").exists():
        print("[quickstart] %s is already compressed at %s -- skipping compression"
              % (model, store_dir))
    else:
        if not _disk_preflight(model, store_dir, assume_yes=args.yes):
            print("Aborted.", file=sys.stderr)
            return 1
        print("[quickstart] compressing %s (small model -- this should take well "
              "under a minute of CPU work plus the download)" % model)
        man = compress_model_to_disk(model, store_dir, config=EngineConfig())
        print("[quickstart] compressed: %.3f GB -> %.3f GB (%.3fx)"
              % (man["total_orig_bytes"] / 1e9, man["total_comp_bytes"] / 1e9,
                 man["ratio"]))

    print("[quickstart] loading tokenizer and running a short generation...")
    tok = AutoTokenizer.from_pretrained(model)
    ids = tok(QUICKSTART_PROMPT, return_tensors="pt").input_ids
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = ids.to(device)

    cfg = EngineConfig(progress=False)
    gen_t0 = time.perf_counter()
    with StreamingLosslessModel(model, store_dir, device=device, config=cfg) as sm:
        seq = sm.generate_greedy(ids, max_new_tokens=24)
        text = tok.decode(seq[0, ids.shape[1]:])
        gen_s = time.perf_counter() - gen_t0
        n_tokens = seq.shape[1] - ids.shape[1]

        print()
        print("Prompt: %r" % QUICKSTART_PROMPT)
        print("Output: %r" % text)
        print()
        print("Measured on this machine, right now:")
        print("  %d tokens in %.1fs (%.2f s/token)" % (n_tokens, gen_s, gen_s / max(n_tokens, 1)))
        print("  peak I/O    : %.2fs   peak decode: %.2fs" %
              (sm.stats.io_seconds, sm.stats.decode_seconds))
        if torch.cuda.is_available():
            print("  peak VRAM   : %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    print()
    print("Total quickstart time: %.1fs" % (time.perf_counter() - t0))
    print()
    print("It works. For a real model:")
    print("  afterimage compress Qwen/Qwen3-14B   # ~30 min download, ~6 min compress")
    print("  afterimage run Qwen/Qwen3-14B \"...\" --auto")
    return 0


# -- serve -----------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    import logging

    import uvicorn
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run("afterimage.server.app:app", host=args.host, port=args.port,
               reload=False, log_level=args.log_level.lower())
    return 0


# -- bench -----------------------------------------------------------------

def cmd_bench(args: argparse.Namespace) -> int:
    try:
        import airllm  # noqa: F401
    except ImportError:
        print("The `airllm` package is not installed -- `pip install airllm` to "
              "enable the comparison, or use `afterimage run` to just run this "
              "engine without a baseline.", file=sys.stderr)
        return 1

    print("For the full cold-cache, drop-caches comparison protocol this project")
    print("uses, see scripts/run_headtohead.py (Linux/WSL2 only -- needs")
    print("/proc/sys/vm/drop_caches). This command runs the same comparison with")
    print("page caches left warm, which is faster to iterate on but not the")
    print("number to report.")
    from transformers import AutoTokenizer
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel
    import time

    store_dir = pathlib.Path(args.store) if args.store else _store_dir_for(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()

    cfg = EngineConfig(vram_cap_gb=args.vram_cap_gb, progress=True)
    with StreamingLosslessModel(args.model, store_dir, device="cuda", config=cfg) as sm:
        t0 = time.perf_counter()
        sm.generate_greedy(ids, max_new_tokens=args.n_tokens)
        wall = time.perf_counter() - t0
    print("afterimage: %.2f s/token   %.2f GB read/token"
          % (wall / args.n_tokens, sm.stats.bytes_read / 1e9 / args.n_tokens))

    model = airllm.AutoModel.from_pretrained(args.model)
    inputs = model.tokenizer(args.prompt, return_tensors="pt",
                             return_attention_mask=False, truncation=True)
    t0 = time.perf_counter()
    model.generate(inputs["input_ids"].cuda(), max_new_tokens=args.n_tokens, use_cache=True)
    wall_air = time.perf_counter() - t0
    print("airllm    : %.2f s/token" % (wall_air / args.n_tokens))
    print("speedup   : %.2fx" % ((wall_air / args.n_tokens) / (wall / args.n_tokens)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from afterimage import __version__

    p = argparse.ArgumentParser(prog="afterimage",
                                description="Lossless streaming inference for models larger than your GPU.")
    p.add_argument("--version", action="version", version="afterimage %s" % __version__)
    p.add_argument("--debug", action="store_true",
                   help="show the full traceback instead of a short diagnosis on error")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="hardware detection + install diagnosis")
    d.add_argument("--skip-disk-check", action="store_true",
                   help="skip the ~256 MB read/write disk-speed probe")
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("compress", help="build a compressed store for a model")
    c.add_argument("model", help="HuggingFace model id, e.g. Qwen/Qwen3-14B")
    c.add_argument("--out", default=None, help="output store directory (default: ~/.afterimage/stores/<model>)")
    c.add_argument("--chunk-size", type=int, default=1024)
    c.add_argument("--quantize", default=None, choices=[None, "q8"])
    c.add_argument("--progress-every", type=int, default=50)
    c.add_argument("--workers", type=int, default=None)
    c.add_argument("--dry-run", action="store_true",
                   help="print estimated download/store size and exit -- no download, "
                        "no compression")
    c.add_argument("--yes", action="store_true",
                   help="don't prompt if the disk-space preflight looks tight")
    c.set_defaults(func=cmd_compress)

    v = sub.add_parser("verify", help="check a compressed store's checksums")
    v.add_argument("model", help="HuggingFace model id whose store to verify")
    v.add_argument("--store", default=None, help="store directory (default: ~/.afterimage/stores/<model>)")
    v.set_defaults(func=cmd_verify)

    pl = sub.add_parser(
        "pull", help="fetch an already-compressed store from a Hugging Face repo")
    pl.add_argument("model", help="HuggingFace model id the store is for")
    pl.add_argument("--store-repo", required=True,
                    help="Hugging Face repo id hosting manifest.json + weights.bin "
                         "(e.g. someone/afterimage-qwen3-14b) -- this project doesn't "
                         "host any itself yet, so this must be a repo you know exists")
    pl.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    pl.add_argument("--store", default=None, help="store directory (default: ~/.afterimage/stores/<model>)")
    pl.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip the checksum pass after downloading (verified by default)")
    pl.set_defaults(func=cmd_pull)

    r = sub.add_parser("run", help="one-off generation from a compressed store")
    r.add_argument("model", help="HuggingFace model id (for the tokenizer + architecture)")
    r.add_argument("prompt")
    r_core = r.add_argument_group(
        "core", "the flags most runs need -- see docs/CONFIGURATION.md")
    r_core.add_argument("--store", default=None, help="store directory (default: ~/.afterimage/stores/<model>)")
    r_core.add_argument("--max-new-tokens", type=int, default=16,
                        help="default kept small on purpose: at the exact minimum-memory "
                             "profile this is roughly seconds-per-token x this number "
                             "before you see anything (default: 16)")
    r_core.add_argument("--profile", default=None,
                        choices=["min-memory", "balanced", "fast"],
                        help="apply a measured operating point (see README's benchmark "
                             "table): min-memory = lowest VRAM, exact, slowest; "
                             "balanced = +4GB residency, 1.66x; fast = +speculation, "
                             "3.15x. Explicit flags below still override it.")
    r_core.add_argument("--auto", action="store_true",
                        help="detect available VRAM and pick a profile automatically; "
                             "prints the decision before running. Explicit flags win.")
    r_core.add_argument("--vram-budget-gb", type=float, default=None,
                        help="spend this much VRAM on residency (more = faster, see "
                             "README); refused up front if infeasible, never approximated")
    r_core.add_argument("--ram-budget-gb", type=float, default=None,
                        help="a second, pinned-host-RAM tier below VRAM and above disk")
    r_core.add_argument("--max-context", type=int, default=None,
                        help="reserve VRAM for up to this many tokens of KV cache "
                             "(requires --vram-budget-gb / --profile / --auto, since "
                             "the reserve needs a VRAM budget to attach a refusal to). "
                             "Without this, an infeasible combination of budget and "
                             "conversation length OOMs mid-generation instead of being "
                             "refused up front -- see docs/CONFIGURATION.md")
    r_core.add_argument("--draft-model", default=None,
                        help="HuggingFace id of a small resident draft model (e.g. "
                             "Qwen/Qwen3-0.6B) -- enables speculative decoding, the "
                             "engine's largest lossless speedup (3.15x measured)")
    r_core.add_argument("--spec-k", type=int, default=8,
                        help="draft chain length when --draft-model is set")
    r_core.add_argument("--spec-temperature", type=float, default=0.0,
                        help="0 (default) makes speculation provably token-identical "
                             "to plain greedy decoding for any draft, at the cost of "
                             "deterministic output; >0 samples from the true distribution")
    r_core.add_argument("--quiet", action="store_true")
    r_core.add_argument("--stats", action="store_true",
                        help="print peak-VRAM/throughput/prefetch counters after generation")
    r_core.add_argument("--raw", action="store_true",
                        help="skip the chat template and feed the prompt to the model "
                             "exactly as typed -- for base/completion models, or to "
                             "reproduce the raw-prompt behaviour from before this flag "
                             "existed. Most instruct models (the default assumption) "
                             "want the template; without it they often ramble or ignore "
                             "the prompt's format.")
    r_core.add_argument("--think", action="store_true",
                        help="allow the model's native reasoning mode instead of "
                             "suppressing it (Qwen3's enable_thinking=False by default). "
                             "Thinking tokens are real generated tokens at this engine's "
                             "per-token price, so this can multiply wall time; off unless "
                             "you specifically want to see the reasoning trace.")
    r_core.add_argument("--no-stream", action="store_true",
                        help="print the full answer at once when generation finishes "
                             "instead of as each token arrives")

    r_adv = r.add_argument_group(
        "advanced / research",
        "opt-in mechanisms from the H0-H18 research layer -- see "
        "docs/RESEARCH_METHODS.md. Ordinary use never needs these.")
    r_adv.add_argument("--vram-cap-gb", type=float, default=None)
    r_adv.add_argument("--io-prefetch-depth", type=int, default=1)
    r_adv.add_argument("--storage-read-policy", default="per_blob",
                       choices=["per_blob", "coalesced_extents", "tensor_extents"])
    r_adv.add_argument("--storage-extent-max-bytes", type=int, default=1 << 28)
    r_adv.add_argument("--storage-extent-max-gap-bytes", type=int, default=0)
    r_adv.add_argument("--io-prefetch-max-depth", type=int, default=8)
    r_adv.add_argument("--prefetch-policy", default="fixed",
                       choices=["fixed", "pi", "mpc", "bayes_probit"])
    r_adv.add_argument("--placement-policy", default="traffic_density",
                       choices=["traffic_density", "profiled_knapsack", "critical_path",
                                "replay_cem", "replay_qubo", "replay_extent_qubo"])
    r_adv.add_argument("--critical-path-profile", default=None)
    r_adv.add_argument("--replay-plan-state", default=None,
                       help="frozen plan produced by `afterimage optimize-residency`")
    r_adv.add_argument("--lm-head-policy", default="full",
                       choices=["full", "certified_mips", "ram_overlay"])
    r_adv.add_argument("--require-pinned-ram", action="store_true",
                       help="fail closed instead of degrading to pageable RAM -- the "
                            "regulated H9 mechanism gate")
    r_adv.add_argument("--trace-output", default=None,
                       help="write an event-DAG trace after generation")
    r_adv.add_argument("--spec-target-cache", action="store_true",
                       help="H18: crop/reuse the exact target KV prefix between "
                            "speculative verification sweeps")
    r_adv.add_argument("--decode-slice-elems", type=int, default=1 << 25,
                       help="weights per bounded decode slice. Smaller values shrink "
                            "transient decode scratch, which is what lowers the floor "
                            "on --vram-budget-gb (1<<22 gets a 14B under 1.7 GB); the "
                            "cost is more kernel launches. Cannot change decoded values.")
    r_adv.add_argument("--no-kv-cache", action="store_true")
    r_adv.add_argument("--ram-tier-format", default="decoded", choices=["decoded", "compressed"],
                       help="'decoded' pins bf16 tensors (needs a real ulimit -l -- see "
                            "EngineConfig.ram_tier_format); 'compressed' caches raw bytes and "
                            "decodes each token instead, fitting ~1.45x more per --ram-budget-gb.")
    r_adv.add_argument("--lm-head-slice-rows", type=int, default=0,
                       help="compute logits in blocks of N vocabulary rows instead of "
                            "materializing lm_head whole. Lowers the VRAM floor by over "
                            "a gigabyte on a 14B (1.556 GB -> ~84 MB at N=8192), but is "
                            "NOT bit-exact: blocking changes the matmul reduction order "
                            "(measured up to 2.0 logit deviation at 14B dimensions). "
                            "0 (default) keeps the lossless whole-head path.")
    r.set_defaults(func=cmd_run)

    q = sub.add_parser(
        "quickstart",
        help="compress + run a small model end to end (~2 GB, minutes) to prove "
             "the install works before committing to a real model")
    q.add_argument("--model", default=None,
                   help="override the default quickstart model (%s)" % QUICKSTART_MODEL)
    q.add_argument("--yes", action="store_true",
                   help="don't prompt if the disk-space preflight looks tight")
    q.set_defaults(func=cmd_quickstart)

    s = sub.add_parser("serve", help="launch the FastAPI server + web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8420)
    s.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"])
    s.set_defaults(func=cmd_serve)

    b = sub.add_parser("bench", help="quick comparison vs AirLLM (warm-cache; see scripts/ for the rigorous protocol)")
    b.add_argument("model")
    b.add_argument("--store", default=None)
    b.add_argument("--n-tokens", type=int, default=3)
    b.add_argument("--vram-cap-gb", type=float, default=None)
    b.add_argument("--prompt", default="The capital of France is")
    b.set_defaults(func=cmd_bench)

    research = sub.add_parser(
        "research",
        help="the H0-H18 research layer -- opt-in, does not affect ordinary "
             "compress/run/serve. See docs/RESEARCH_METHODS.md.")
    research_sub = research.add_subparsers(dest="research_command", required=True)

    e = research_sub.add_parser(
        "experiments", help="list versioned H0-H18 experiment definitions")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_experiments)

    tp = research_sub.add_parser(
        "test-plan", help="show hypothesis-aware L0-L3 evidence requirements")
    tp.add_argument("hypothesis", nargs="?", default=None)
    tp.add_argument("--json", action="store_true")
    tp.set_defaults(func=cmd_test_plan)

    pin = research_sub.add_parser(
        "pin-preflight", help="prove a pinned-host-RAM experiment can allocate")
    pin.add_argument("--gigabytes", type=float, default=1.6)
    pin.add_argument("--static-only", action="store_true",
                     help="check limits without attempting the allocation")
    pin.add_argument("--json", action="store_true")
    pin.add_argument("--out", default=None,
                     help="write an immutable JSON preflight artifact")
    pin.set_defaults(func=cmd_pin_preflight)

    t = research_sub.add_parser(
        "profile-trace", help="build a measured critical-path profile from trace files")
    t.add_argument("traces", nargs="+")
    t.add_argument("--out", required=True)
    t.add_argument("--manifest", default=None,
                   help="optional store manifest used to enforce 90%% tensor coverage")
    t.set_defaults(func=cmd_profile_trace)

    o = research_sub.add_parser(
        "optimize-residency",
        help="learn a frozen whole-set residency plan from event-DAG traces")
    o.add_argument("traces", nargs="+")
    o.add_argument("--manifest", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--vram-budget-gb", type=float, required=True)
    o.add_argument("--decode-slice-elems", type=int, default=1 << 25)
    o.add_argument("--search-method", choices=["cem", "qubo", "extent-qubo"],
                   default="cem")
    o.add_argument("--iterations", type=int, default=12)
    o.add_argument("--population", type=int, default=64)
    o.add_argument("--elite-fraction", type=float, default=0.15)
    o.add_argument("--pairwise-candidates", type=int, default=24)
    o.add_argument("--anneal-restarts", type=int, default=8)
    o.add_argument("--anneal-sweeps", type=int, default=2000)
    o.add_argument("--max-extent-bytes", type=int, default=1 << 28)
    o.add_argument("--max-extent-gap-bytes", type=int, default=0)
    o.add_argument("--max-tensors-per-extent", type=int, default=8)
    o.add_argument("--seed", type=int, default=0)
    o.set_defaults(func=cmd_optimize_residency)

    return p


# (exception type or message substring, one-line diagnosis, fixing command)
# Checked in order; the first match wins. Keeps common, predictable failures
# from surfacing as a raw Python traceback -- --debug always shows the real
# one underneath.
_KNOWN_FAILURES: list[tuple[type[BaseException] | None, str, str]] = [
    (ModuleNotFoundError, "No module named 'transformers'",
     "the [server]/[gpu] extras aren't installed -- run: pip install -e \".[gpu,server]\""),
    (FileNotFoundError, "manifest.json",
     "no compressed store found -- run: afterimage compress MODEL_ID"),
    (RuntimeError, "out of memory",
     "ran out of VRAM -- try a smaller --vram-budget-gb, or --lm-head-slice-rows "
     "to shrink the output-head floor (not bit-exact, see docs/CONFIGURATION.md)"),
    (RuntimeError, "memlock",
     "the host's pinned-memory limit is too low for this -- see "
     "docs/TROUBLESHOOTING.md's WSL2 memlock entry"),
    (OSError, "No space left on device",
     "ran out of disk mid-pass -- afterimage compress --dry-run estimates space "
     "needed before it starts"),
]


def _diagnose(exc: BaseException) -> str | None:
    text = str(exc)
    for exc_type, needle, fix in _KNOWN_FAILURES:
        if (exc_type is None or isinstance(exc, exc_type)) and needle in text:
            return fix
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        fix = _diagnose(exc)
        print("Error: %s" % exc, file=sys.stderr)
        if fix:
            print("  -> %s" % fix, file=sys.stderr)
        print("  (re-run with `afterimage --debug ...` for the full traceback)",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
