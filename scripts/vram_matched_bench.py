#!/usr/bin/env python3
"""VRAM-MATCHED head-to-head, plus worked examples.

The previous comparison was not controlled: Afterimage held 2.66 GB peak
while AirLLM held ~1.57 GB, because Afterimage kept lm_head (1.56 GB, the
largest single tensor) permanently VRAM-resident and AirLLM streams it.
Faster-while-using-more-memory is not a like-for-like result.

This sweeps Afterimage across VRAM budgets -- including the minimum-VRAM
plan where lm_head streams too, which is the configuration directly
comparable to AirLLM -- and measures AirLLM in the same process with the
same peak-memory instrumentation, so both numbers come from the same
counter rather than from two differently-instrumented runs.

Also records the actual prompt/answer/time for each run, so the results
can be shown as worked examples rather than only as aggregate rates.

Run:  python -u scripts/vram_matched_bench.py --n-tokens 12
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

MODEL = "Qwen/Qwen3-14B"
STORE = "/root/afterimage/store_14b"
PROMPT = "What is the capital of France?"


def log(m: str) -> None:
    print(m, flush=True)


def drop_caches() -> None:
    """Both systems must read from DISK, not the page cache, or the
    comparison measures RAM bandwidth instead of the thing being compared."""
    import subprocess
    try:
        subprocess.run(["sync"], check=True, timeout=60)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception as e:
        log("  WARNING: could not drop caches (%s) -- timings may be optimistic" % e)


def _reset_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_airllm(prompt: str, n_tokens: int) -> dict:
    from airllm import AutoModel

    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(MODEL)
    init_s = time.perf_counter() - t0

    inputs = model.tokenizer(prompt, return_tensors="pt",
                             return_attention_mask=False, truncation=True)
    drop_caches()
    _reset_peak()
    t0 = time.perf_counter()
    out = model.generate(inputs["input_ids"].cuda(), max_new_tokens=n_tokens,
                         use_cache=True, return_dict_in_generate=True)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    seq = out.sequences if hasattr(out, "sequences") else out
    gen = seq[0, inputs["input_ids"].shape[1]:]
    text = model.tokenizer.decode(gen)

    del model, out, seq
    gc.collect()
    torch.cuda.empty_cache()

    return {"system": "airllm", "config": "default (streams everything)",
            "peak_vram_gb": peak, "wall_s": wall, "s_per_tok": wall / n_tokens,
            "tok_per_s": n_tokens / wall, "init_s": init_s,
            "prompt": prompt, "answer": text, "token_ids": gen.tolist(),
            "gb_per_token": None}


def run_ours(prompt: str, n_tokens: int, vram_budget_gb, label: str,
             decode_slice_elems: int = 1 << 25) -> dict:
    from transformers import AutoTokenizer
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()

    cfg = EngineConfig(vram_budget_gb=vram_budget_gb, io_prefetch_depth=2,
                       empty_cache_every=1, progress=False,
                       decode_slice_elems=decode_slice_elems)
    t0 = time.perf_counter()
    sm = StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg)
    init_s = time.perf_counter() - t0

    tiers = {}
    for t in sm._tier.values():
        tiers[t] = tiers.get(t, 0) + 1

    drop_caches()
    sm.stats.reset()
    _reset_peak()
    t0 = time.perf_counter()
    seq = sm.generate_greedy(ids, max_new_tokens=n_tokens, use_cache=True)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    text = tok.decode(seq[0, ids.shape[1]:])
    result = {"system": "afterimage", "config": label,
              "vram_budget_gb": vram_budget_gb,
              "peak_vram_gb": peak, "wall_s": wall, "s_per_tok": wall / n_tokens,
              "tok_per_s": n_tokens / wall, "init_s": init_s,
              "bytes_read": sm.stats.bytes_read,
              "gb_per_token": sm.stats.bytes_read / 1e9 / n_tokens,
              "io_s": sm.stats.io_seconds, "decode_s": sm.stats.decode_seconds,
              "compute_s": sm.stats.compute_seconds,
              "tiers": tiers,
              "decode_slice_elems": decode_slice_elems,
              "lm_head_tier": sm._tier.get("lm_head.weight"),
              "prompt": prompt, "answer": text,
              "token_ids": seq[0, ids.shape[1]:].tolist()}
    sm.close()
    del sm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=6)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--skip-airllm", action="store_true")
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--budgets", default="2.1,4.0,6.0",
                    help="comma-separated vram_budget_gb values to sweep")
    ap.add_argument("--decode-slice-elems", type=int, default=1 << 25,
                    help="weights per bounded decode slice; smaller = less "
                         "transient decode scratch, more kernel launches")
    ap.add_argument("--out", default="/root/afterimage/results/vram_matched_14b.json")
    args = ap.parse_args()

    man = json.loads((pathlib.Path(STORE) / "manifest.json").read_text())
    log("=" * 76)
    log("VRAM-MATCHED COMPARISON -- %s" % MODEL)
    log("=" * 76)
    log("  original (bf16) : %.2f GB" % (man["total_orig_bytes"] / 1e9))
    log("  compressed      : %.2f GB (%.3fx)" % (man["total_comp_bytes"] / 1e9, man["ratio"]))
    log("  prompt          : %r" % args.prompt)
    log("  tokens          : %d" % args.n_tokens)
    log("")

    results = []

    if not args.skip_airllm:
        log("--- AirLLM (baseline) ---")
        try:
            r = run_airllm(args.prompt, args.n_tokens)
            results.append(r)
            log("  peak VRAM %.2f GB   %.2f s/token   answer=%r"
                % (r["peak_vram_gb"], r["s_per_tok"], r["answer"]))
        except Exception as e:
            log("  FAILED: %r" % e)
            import traceback; traceback.print_exc()
        log("")

    # 2.1 GB is the smallest feasible budget on this model: the planner
    # reserves (largest tensor + decode scratch) as working headroom, and
    # lm_head is 1.56 GB. At this budget nothing at all stays resident --
    # lm_head streams too, which is exactly what AirLLM does, so this is
    # the directly comparable configuration.
    _labels = {2.1: "minimum VRAM -- everything streams (AirLLM-matched)",
               3.0: "small budget", 4.0: "medium budget",
               6.0: "large budget -- lm_head resident"}
    sweep = [] if args.skip_ours else [
        (float(b), _labels.get(float(b), "budget %.1f GB" % float(b)))
        for b in args.budgets.split(",") if b.strip()]

    for budget, label in sweep:
        log("--- Afterimage @ vram_budget_gb=%.1f (%s) ---" % (budget, label))
        try:
            r = run_ours(args.prompt, args.n_tokens, budget, label,
                         decode_slice_elems=args.decode_slice_elems)
            results.append(r)
            log("  peak VRAM %.2f GB   %.2f s/token   %.2f GB/token   lm_head=%s"
                % (r["peak_vram_gb"], r["s_per_tok"], r["gb_per_token"], r["lm_head_tier"]))
            log("  answer=%r" % r["answer"])
        except Exception as e:
            log("  FAILED: %r" % e)
            import traceback; traceback.print_exc()
        log("")

    log("=" * 76)
    log("SUMMARY")
    log("=" * 76)
    log("%-46s %10s %12s %11s" % ("configuration", "peak VRAM", "s/token", "GB/token"))
    for r in results:
        log("%-46s %9.2fG %11.2f %11s" % (
            ("AirLLM" if r["system"] == "airllm" else "Afterimage: " + r["config"])[:46],
            r["peak_vram_gb"], r["s_per_tok"],
            ("%.2f" % r["gb_per_token"]) if r["gb_per_token"] else "-"))

    air = next((r for r in results if r["system"] == "airllm"), None)
    if air:
        log("")
        log("Speedup at comparable peak VRAM:")
        for r in results:
            if r["system"] != "afterimage":
                continue
            log("  %-44s %.2fx  (%.2f GB vs AirLLM %.2f GB)"
                % (r["config"][:44], air["s_per_tok"] / r["s_per_tok"],
                   r["peak_vram_gb"], air["peak_vram_gb"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL, "prompt": args.prompt,
                               "n_tokens": args.n_tokens,
                               "manifest": {k: man[k] for k in
                                            ("total_orig_bytes", "total_comp_bytes", "ratio")},
                               "results": results}, indent=2))
    log("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
