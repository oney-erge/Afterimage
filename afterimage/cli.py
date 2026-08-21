"""Command-line entry point for Afterimage.

    afterimage doctor                     hardware detection + install diagnosis
    afterimage compress MODEL              build a compressed store, with progress
    afterimage run MODEL PROMPT            one-off generation
    afterimage serve                       launch the FastAPI server + web UI
    afterimage bench MODEL                 head-to-head vs AirLLM (needs airllm installed)

Registered as the `afterimage` console script (see pyproject.toml
[project.scripts]) so `pip install -e .` makes this runnable as a plain
command, not just `python -m afterimage.cli`.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

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
        print("      wants a different value -- see docs/MASTER_PLAN.md.")

    print()
    print("Compressed stores in %s:" % DEFAULT_STORE_ROOT)
    if DEFAULT_STORE_ROOT.exists():
        stores = sorted(p for p in DEFAULT_STORE_ROOT.iterdir() if (p / "manifest.json").exists())
        if not stores:
            print("  (none yet -- run `afterimage compress <model>`)")
        for s in stores:
            man = json.loads((s / "manifest.json").read_text())
            print("  %-40s %6.2f GB (%.2fx)" % (
                man.get("model_id", s.name), man["total_comp_bytes"] / 1e9, man["ratio"]))
    else:
        print("  (none yet -- run `afterimage compress <model>`)")

    ok = cuda_ok or gpu["vendor"] == "amd"
    print()
    print("Overall: %s" % ("ready" if ok else "no usable GPU found -- CPU fallback only, will be slow"))
    return 0 if ok else 1


# -- compress ------------------------------------------------------------

def cmd_compress(args: argparse.Namespace) -> int:
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import compress_model_to_disk

    out_dir = pathlib.Path(args.out) if args.out else _store_dir_for(args.model, )
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


# -- run -------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoTokenizer
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    store_dir = pathlib.Path(args.store) if args.store else _store_dir_for(args.model)
    if not (store_dir / "manifest.json").exists():
        print("No compressed store at %s -- run `afterimage compress %s` first"
              % (store_dir, args.model), file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = ids.to(device)

    cfg = EngineConfig(vram_cap_gb=args.vram_cap_gb, vram_budget_gb=args.vram_budget_gb,
                       ram_budget_gb=args.ram_budget_gb, progress=not args.quiet,
                       io_prefetch_depth=args.io_prefetch_depth,
                       decode_slice_elems=args.decode_slice_elems,
                       ram_tier_format=args.ram_tier_format,
                       lm_head_slice_rows=args.lm_head_slice_rows,
                       placement_policy=args.placement_policy,
                       critical_path_profile=args.critical_path_profile,
                       replay_plan_state=args.replay_plan_state,
                       prefetch_policy=args.prefetch_policy,
                       io_prefetch_max_depth=args.io_prefetch_max_depth,
                       lm_head_policy=args.lm_head_policy,
                       trace_events=bool(args.trace_output),
                       trace_output=args.trace_output)
    if not cfg.is_lossless:
        print("WARNING: %s" % cfg.describe(), file=sys.stderr)
    with StreamingLosslessModel(args.model, store_dir, device=device, config=cfg) as sm:
        seq = sm.generate_greedy(ids, max_new_tokens=args.max_new_tokens,
                                 use_cache=not args.no_kv_cache)
        text = tok.decode(seq[0, ids.shape[1]:])
        print(text)
        if args.stats:
            print("\n--- stats ---", file=sys.stderr)
            print("io=%.2fs decode=%.2fs compute=%.2fs bytes_read=%.2fGB"
                  % (sm.stats.io_seconds, sm.stats.decode_seconds,
                     sm.stats.compute_seconds, sm.stats.bytes_read / 1e9),
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
    """Fit a whole-set CEM residency plan on separate calibration traces."""
    from afterimage.runtime.critical_path import TraceRecorder
    from afterimage.runtime.replay_planner import optimize_replay_residency

    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    traces = [TraceRecorder.load(path) for path in args.traces]
    plan = optimize_replay_residency(
        manifest, traces, vram_budget_gb=args.vram_budget_gb,
        decode_slice_elems=args.decode_slice_elems,
        iterations=args.iterations, population=args.population,
        elite_fraction=args.elite_fraction, seed=args.seed)
    plan.save(args.out)
    print("Wrote replay-CEM plan with %d resident tensors to %s" %
          (len(plan.vram_keys), args.out))
    print("Calibration replay: %.3fs -> %.3fs (%.3fx), %d evaluations" %
          (plan.report.baseline_s, plan.report.optimized_s,
           plan.report.predicted_speedup, plan.report.evaluations))
    return 0


# -- serve -----------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    uvicorn.run("afterimage.server.app:app", host=args.host, port=args.port,
               reload=False)
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
    import torch
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
        seq = sm.generate_greedy(ids, max_new_tokens=args.n_tokens)
        wall = time.perf_counter() - t0
    print("afterimage: %.2f s/token   %.2f GB read/token"
          % (wall / args.n_tokens, sm.stats.bytes_read / 1e9 / args.n_tokens))

    from airllm import AutoModel
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
    p = argparse.ArgumentParser(prog="afterimage",
                                description="Lossless streaming inference for models larger than your GPU.")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="hardware detection + install diagnosis")
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("compress", help="build a compressed store for a model")
    c.add_argument("model", help="HuggingFace model id, e.g. Qwen/Qwen3-14B")
    c.add_argument("--out", default=None, help="output store directory (default: ~/.afterimage/stores/<model>)")
    c.add_argument("--chunk-size", type=int, default=1024)
    c.add_argument("--quantize", default=None, choices=[None, "q8"])
    c.add_argument("--progress-every", type=int, default=50)
    c.add_argument("--workers", type=int, default=None)
    c.set_defaults(func=cmd_compress)

    r = sub.add_parser("run", help="one-off generation from a compressed store")
    r.add_argument("model", help="HuggingFace model id (for the tokenizer + architecture)")
    r.add_argument("prompt")
    r.add_argument("--store", default=None, help="store directory (default: ~/.afterimage/stores/<model>)")
    r.add_argument("--max-new-tokens", type=int, default=64)
    r.add_argument("--vram-cap-gb", type=float, default=None)
    r.add_argument("--vram-budget-gb", type=float, default=None)
    r.add_argument("--ram-budget-gb", type=float, default=None)
    r.add_argument("--io-prefetch-depth", type=int, default=1)
    r.add_argument("--io-prefetch-max-depth", type=int, default=8)
    r.add_argument("--prefetch-policy", default="fixed", choices=["fixed", "pi", "mpc"])
    r.add_argument("--placement-policy", default="traffic_density",
                   choices=["traffic_density", "profiled_knapsack", "critical_path",
                            "replay_cem"])
    r.add_argument("--critical-path-profile", default=None)
    r.add_argument("--replay-plan-state", default=None,
                   help="frozen plan produced by `afterimage optimize-residency`")
    r.add_argument("--lm-head-policy", default="full",
                   choices=["full", "certified_mips", "ram_overlay"])
    r.add_argument("--trace-output", default=None,
                   help="write an event-DAG trace after generation")
    r.add_argument("--decode-slice-elems", type=int, default=1 << 25,
                   help="weights per bounded decode slice. Smaller values shrink "
                        "transient decode scratch, which is what lowers the floor "
                        "on --vram-budget-gb (1<<22 gets a 14B under 1.7 GB); the "
                        "cost is more kernel launches. Cannot change decoded values.")
    r.add_argument("--no-kv-cache", action="store_true")
    r.add_argument("--ram-tier-format", default="decoded", choices=["decoded", "compressed"],
                   help="'decoded' pins bf16 tensors (needs a real ulimit -l -- see "
                        "EngineConfig.ram_tier_format); 'compressed' caches raw bytes and "
                        "decodes each token instead, fitting ~1.45x more per --ram-budget-gb.")
    r.add_argument("--lm-head-slice-rows", type=int, default=0,
                   help="compute logits in blocks of N vocabulary rows instead of "
                        "materializing lm_head whole. Lowers the VRAM floor by over "
                        "a gigabyte on a 14B (1.556 GB -> ~84 MB at N=8192), but is "
                        "NOT bit-exact: blocking changes the matmul reduction order "
                        "(measured up to 2.0 logit deviation at 14B dimensions). "
                        "0 (default) keeps the lossless whole-head path.")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--stats", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("serve", help="launch the FastAPI server + web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8420)
    s.set_defaults(func=cmd_serve)

    b = sub.add_parser("bench", help="quick comparison vs AirLLM (warm-cache; see scripts/ for the rigorous protocol)")
    b.add_argument("model")
    b.add_argument("--store", default=None)
    b.add_argument("--n-tokens", type=int, default=3)
    b.add_argument("--vram-cap-gb", type=float, default=None)
    b.add_argument("--prompt", default="The capital of France is")
    b.set_defaults(func=cmd_bench)

    e = sub.add_parser("experiments", help="list versioned H0-H11 experiment definitions")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_experiments)

    t = sub.add_parser("profile-trace",
                       help="build a measured critical-path profile from trace files")
    t.add_argument("traces", nargs="+")
    t.add_argument("--out", required=True)
    t.add_argument("--manifest", default=None,
                   help="optional store manifest used to enforce 90%% tensor coverage")
    t.set_defaults(func=cmd_profile_trace)

    o = sub.add_parser(
        "optimize-residency",
        help="learn a frozen whole-set residency plan from event-DAG traces")
    o.add_argument("traces", nargs="+")
    o.add_argument("--manifest", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--vram-budget-gb", type=float, required=True)
    o.add_argument("--decode-slice-elems", type=int, default=1 << 25)
    o.add_argument("--iterations", type=int, default=12)
    o.add_argument("--population", type=int, default=64)
    o.add_argument("--elite-fraction", type=float, default=0.15)
    o.add_argument("--seed", type=int, default=0)
    o.set_defaults(func=cmd_optimize_residency)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
