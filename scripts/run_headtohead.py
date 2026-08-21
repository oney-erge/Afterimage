#!/usr/bin/env python3
"""Head-to-head on a model that fits in NEITHER VRAM nor RAM.

Both systems solve the same problem the same way -- stream the model from
disk one layer at a time -- and both are bit-exact. The only difference is
how many bytes cross the bus. This measures whether that difference actually
shows up in wall-clock tokens/sec, or whether decode overhead eats it.

Run:  python -u scripts/run_headtohead.py --n-tokens 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

MODEL = "Qwen/Qwen3-14B"
STORE = "/root/afterimage/store_14b"
PROMPT = "The capital of France is"


def log(m: str) -> None:
    print(m, flush=True)


def drop_caches() -> bool:
    """Both systems must read from DISK, not the page cache, or the
    comparison measures RAM bandwidth and is meaningless (this machine has
    19 GB of RAM against a 29.5 GB model, so the model cannot be fully
    cached -- but partial caching would still skew a short run)."""
    import subprocess
    try:
        subprocess.run(["sync"], check=True, timeout=60)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except Exception as e:
        log("  WARNING: could not drop caches (%s) -- timings may be optimistic" % e)
        return False


def run_ours(n_tokens: int, vram_cap_gb=None, empty_cache_every: int = 0,
            io_prefetch_depth: int = 1, vram_budget_gb=None, ram_budget_gb=None) -> dict:
    from transformers import AutoTokenizer
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()

    t0 = time.perf_counter()
    cfg = EngineConfig(vram_cap_gb=vram_cap_gb, empty_cache_every=empty_cache_every,
                       progress=True, io_prefetch_depth=io_prefetch_depth,
                       vram_budget_gb=vram_budget_gb, ram_budget_gb=ram_budget_gb)
    sm = StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg)
    load_s = time.perf_counter() - t0
    log("  engine init (resident weights): %.1fs" % load_s)

    drop_caches()
    sm.stats.reset()
    t0 = time.perf_counter()
    seq = sm.generate_greedy(ids, max_new_tokens=n_tokens)
    wall = time.perf_counter() - t0

    text = tok.decode(seq[0, ids.shape[1]:])
    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_resv = torch.cuda.max_memory_reserved() / 1e9
    log("  peak VRAM: %.2f GB live / %.2f GB reserved" % (peak_alloc, peak_resv))
    result = {
        "system": "afterimage-lossless",
        "peak_vram_live_gb": peak_alloc,
        "peak_vram_reserved_gb": peak_resv,
        "wall_s": wall,
        "tokens": n_tokens,
        "tok_per_s": n_tokens / wall,
        "s_per_tok": wall / n_tokens,
        "bytes_read": sm.stats.bytes_read,
        "gb_per_token": sm.stats.bytes_read / 1e9 / n_tokens,
        "decode_s": sm.stats.decode_seconds,
        "io_s": sm.stats.io_seconds,
        "compute_s": sm.stats.compute_seconds,
        "layer_loads": sm.stats.layer_loads,
        "prefetch": sm.prefetch,
        "io_prefetch_depth": sm.io_prefetch_depth,
        "vram_tier_count": sum(1 for t in sm._tier.values() if t == "vram"),
        "ram_tier_count": sum(1 for t in sm._tier.values() if t == "ram"),
        "disk_tier_count": sum(1 for t in sm._tier.values() if t == "disk"),
        "text": text,
        "token_ids": seq[0, ids.shape[1]:].tolist(),
    }
    sm.close()
    return result


def run_airllm(n_tokens: int) -> dict:
    from airllm import AutoModel

    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(MODEL)
    load_s = time.perf_counter() - t0
    log("  airllm init/split: %.1fs" % load_s)

    inputs = model.tokenizer(PROMPT, return_tensors="pt",
                             return_attention_mask=False, truncation=True)
    drop_caches()
    t0 = time.perf_counter()
    out = model.generate(inputs["input_ids"].cuda(), max_new_tokens=n_tokens,
                         use_cache=True, return_dict_in_generate=True)
    wall = time.perf_counter() - t0

    seq = out.sequences if hasattr(out, "sequences") else out
    gen = seq[0, inputs["input_ids"].shape[1]:]
    text = model.tokenizer.decode(gen)
    return {
        "system": "airllm",
        "wall_s": wall,
        "tokens": n_tokens,
        "tok_per_s": n_tokens / wall,
        "s_per_tok": wall / n_tokens,
        "text": text,
        "token_ids": gen.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=3)
    ap.add_argument("--skip-airllm", action="store_true")
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--vram-cap-gb", type=float, default=None,
                    help="hard cap on GPU memory for this process")
    ap.add_argument("--empty-cache-every", type=int, default=0,
                    help="release cached GPU blocks every N layer frees")
    ap.add_argument("--io-prefetch-depth", type=int, default=1,
                    help="0 disables I/O prefetch overlap; >1 prefetches further ahead")
    ap.add_argument("--vram-budget-gb", type=float, default=None,
                    help="hand residency to the three-tier planner instead of the legacy fixed policy")
    ap.add_argument("--ram-budget-gb", type=float, default=None,
                    help="pinned-host-RAM tier size; requires --vram-budget-gb")
    args = ap.parse_args()

    man = json.loads((pathlib.Path(STORE) / "manifest.json").read_text())
    log("=" * 68)
    log("MODEL: %s" % MODEL)
    log("=" * 68)
    log("  original (bf16)  : %.2f GB" % (man["total_orig_bytes"] / 1e9))
    log("  compressed       : %.2f GB  (%.3fx smaller, %.1f%% of original)"
        % (man["total_comp_bytes"] / 1e9, man["ratio"], 100 / man["ratio"]))
    free, total = torch.cuda.mem_get_info()
    log("  GPU VRAM total   : %.2f GB   (model is %.1fx larger than VRAM)"
        % (total / 1e9, man["total_orig_bytes"] / total))
    log("")

    results = []
    if not args.skip_ours:
        log("--- AFTERIMAGE (lossless compressed streaming) ---")
        try:
            r = run_ours(args.n_tokens, args.vram_cap_gb, args.empty_cache_every,
                        args.io_prefetch_depth, args.vram_budget_gb, args.ram_budget_gb)
            results.append(r)
            log("  %.1f s/token   %.4f tok/s   %.2f GB read/token"
                % (r["s_per_tok"], r["tok_per_s"], r["gb_per_token"]))
            log("  output: %r" % r["text"])
        except Exception as e:
            log("  FAILED: %r" % e)
            import traceback; traceback.print_exc()

    if not args.skip_airllm:
        log("")
        log("--- AIRLLM (uncompressed layer streaming) ---")
        try:
            r = run_airllm(args.n_tokens)
            results.append(r)
            log("  %.1f s/token   %.4f tok/s" % (r["s_per_tok"], r["tok_per_s"]))
            log("  output: %r" % r["text"])
        except Exception as e:
            log("  FAILED: %r" % e)
            import traceback; traceback.print_exc()

    log("")
    log("=" * 68)
    log("RESULT")
    log("=" * 68)
    by = {r["system"]: r for r in results}
    if "afterimage-lossless" in by and "airllm" in by:
        a, b = by["afterimage-lossless"], by["airllm"]
        log("  afterimage : %8.2f s/token   %.4f tok/s" % (a["s_per_tok"], a["tok_per_s"]))
        log("  airllm     : %8.2f s/token   %.4f tok/s" % (b["s_per_tok"], b["tok_per_s"]))
        log("  SPEEDUP    : %.2fx %s" % (b["s_per_tok"] / a["s_per_tok"],
                                          "(we are FASTER)" if a["s_per_tok"] < b["s_per_tok"]
                                          else "(we are SLOWER)"))
        same = a["token_ids"] == b["token_ids"]
        log("  identical output tokens: %s" % same)

    out = pathlib.Path("/root/afterimage/results/headtohead_14b.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"manifest": {k: man[k] for k in
                    ("total_orig_bytes", "total_comp_bytes", "ratio")},
                   "results": results}, indent=2))
    log("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
