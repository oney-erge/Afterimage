#!/usr/bin/env python3
"""One table: every method, side by side with AirLLM, on identical work.

Same prompt, same token count, cold page cache before every timed run, and
peak VRAM read from the SAME counter (torch.cuda.max_memory_allocated) for
every row including AirLLM's. Each row also records whether it is lossless,
because a faster row that is not bit-exact is not comparable to one that is
and must never be presented as if it were.

AirLLM is run with do_sample=False. Without it, HF generate() honours
Qwen3's generation_config (which samples), so the baseline produced
different text than our greedy path and the transcripts were not
comparable -- a presentation bug found in an earlier run of this comparison.

Also checks TOKEN AGREEMENT between the lossless greedy path and the
chunked-head path. The chunked head is not bit-exact (blocking the output
projection changes the matmul's reduction order; see RESULTS_LOG.md), so
the practically useful question is not "are the logits identical" -- they
are not -- but "does it ever change the token that comes out."
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

MODEL = "Qwen/Qwen3-14B"
DRAFT = "Qwen/Qwen3-0.6B"
STORE = "/root/afterimage/store_14b"
PROMPT = "What is the capital of France?"


def log(m: str) -> None:
    print(m, flush=True)


def drop_caches() -> None:
    subprocess.run(["sync"], check=False)
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except OSError as e:
        log("  WARNING: drop_caches failed (%s) -- timings may be optimistic" % e)


def _reset_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _tok_ids(prompt):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    return tok, tok(prompt, return_tensors="pt").input_ids.cuda()


def run_airllm(prompt, n):
    from airllm import AutoModel
    model = AutoModel.from_pretrained(MODEL)
    enc = model.tokenizer(prompt, return_tensors="pt", truncation=True)
    ids = enc["input_ids"].cuda()
    kw = {}
    if enc.get("attention_mask") is not None:
        kw["attention_mask"] = enc["attention_mask"].cuda()

    drop_caches(); _reset_peak()
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=n, use_cache=True,
                         do_sample=False, return_dict_in_generate=True, **kw)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    seq = out.sequences if hasattr(out, "sequences") else out
    gen = seq[0, ids.shape[1]:]
    text = model.tokenizer.decode(gen)
    del model, out, seq
    gc.collect(); torch.cuda.empty_cache()
    return dict(method="AirLLM (baseline)", vram_gb=peak, s_per_tok=wall / n,
                lossless=True, answer=text, token_ids=gen.tolist(), notes="streams raw weights")


def run_engine(prompt, n, label, budget, slice_elems=1 << 20, head_rows=0,
               spec=False, spec_k=8, notes=""):
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel, load_draft_model

    tok, ids = _tok_ids(prompt)
    draft = None
    draft_gb = 0.0
    if spec:
        _reset_peak()
        draft = load_draft_model(DRAFT, device="cuda")
        draft_gb = torch.cuda.max_memory_allocated() / 1e9

    cfg = EngineConfig(vram_budget_gb=budget, io_prefetch_depth=2,
                       decode_slice_elems=slice_elems,
                       lm_head_slice_rows=head_rows,
                       draft_mode="model" if spec else "none", spec_k=spec_k)
    sm = StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg)

    drop_caches(); sm.stats.reset(); _reset_peak()
    t0 = time.perf_counter()
    if spec:
        gen = torch.Generator(device="cuda").manual_seed(0)
        seq, _ = sm.generate_adaptive(ids, max_new_tokens=n, draft_model=draft,
                                      temperature=0.0, generator=gen)
    else:
        seq = sm.generate_greedy(ids, max_new_tokens=n, use_cache=True)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    n_out = seq.shape[1] - ids.shape[1]
    out = dict(method=label, vram_gb=peak, s_per_tok=wall / n_out,
               lossless=cfg.is_lossless, answer=tok.decode(seq[0, ids.shape[1]:]),
               token_ids=seq[0, ids.shape[1]:].tolist(),
               gb_per_tok=sm.stats.bytes_read / 1e9 / n_out,
               io_s=sm.stats.io_seconds, decode_s=sm.stats.decode_seconds,
               draft_model_gb=draft_gb, notes=notes)
    if spec:
        out["tok_per_sweep"] = n_out / max(1, sm.stats.spec_sweeps)
    sm.close()
    del sm, draft
    gc.collect(); torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=8)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--out", default="/root/afterimage/results/methods_vs_airllm.json")
    ap.add_argument("--skip-airllm", action="store_true")
    args = ap.parse_args()

    results = []
    log("=" * 92)
    log("ALL METHODS vs AirLLM -- %s, %d tokens, cold cache, same VRAM counter"
        % (MODEL, args.n_tokens))
    log("=" * 92)

    plan = []
    if not args.skip_airllm:
        plan.append(("airllm", None))
    plan += [
        ("engine", dict(label="1. Compression only (lossless)", budget=1.80,
                        slice_elems=1 << 20,
                        notes="closest lossless match to AirLLM's VRAM")),
        ("engine", dict(label="2. + residency, 2.1GB (lossless)", budget=2.10,
                        slice_elems=1 << 22, notes="spends spare VRAM")),
        ("engine", dict(label="2. + residency, 4.0GB (lossless)", budget=4.00,
                        slice_elems=1 << 22, notes="spends more VRAM")),
        ("engine", dict(label="3. + chunked head (LOSSY)", budget=0.50,
                        slice_elems=1 << 20, head_rows=2048,
                        notes="head never materialized")),
        ("engine", dict(label="4. + speculation, 4GB (lossless)", budget=2.70,
                        slice_elems=1 << 22, spec=True,
                        notes="draft model costs ~1.3GB on top")),
        ("engine", dict(label="5. chunked head + speculation (LOSSY)", budget=0.50,
                        slice_elems=1 << 20, head_rows=2048, spec=True,
                        notes="both levers together -- lowest VRAM at speed")),
    ]

    for kind, kw in plan:
        try:
            if kind == "airllm":
                log("\n--- AirLLM ---")
                r = run_airllm(args.prompt, args.n_tokens)
            else:
                log("\n--- %s ---" % kw["label"])
                r = run_engine(args.prompt, args.n_tokens, **kw)
            results.append(r)
            log("   VRAM %.3f GB   %.2f s/tok   lossless=%s" %
                (r["vram_gb"], r["s_per_tok"], r["lossless"]))
            log("   answer: %r" % r["answer"])
        except Exception as e:
            log("   FAILED: %r" % e)
            results.append({"method": (kw or {}).get("label", "airllm"),
                            "FAILED": repr(e)})

    ok = [r for r in results if "FAILED" not in r]
    air = next((r for r in ok if r["method"].startswith("AirLLM")), None)

    log("\n" + "=" * 92)
    log("%-40s %9s %10s %10s %9s" % ("method", "VRAM", "s/token", "lossless", "vs AirLLM"))
    log("-" * 92)
    for r in ok:
        sp = ("%.2fx" % (air["s_per_tok"] / r["s_per_tok"])) if air else "-"
        log("%-40s %8.3fG %10.2f %10s %9s"
            % (r["method"][:40], r["vram_gb"], r["s_per_tok"],
               "yes" if r["lossless"] else "NO", sp))
    for r in results:
        if "FAILED" in r:
            log("%-40s %8s %10s %10s %9s" % (r["method"][:40], "FAILED", "-", "-", "-"))

    # Token agreement: does the non-bit-exact path ever change the output?
    base = next((r for r in ok if r["method"].startswith("1.")), None)
    if base:
        log("\nToken agreement vs the lossless greedy path (method 1):")
        for r in ok:
            if r is base or r["method"].startswith("AirLLM"):
                continue
            same = r["token_ids"] == base["token_ids"]
            log("  %-40s %s" % (r["method"][:40],
                                "IDENTICAL" if same else "DIFFERS: %r" % r["answer"]))

    if air:
        log("\nVRAM vs AirLLM (%.3f GB):" % air["vram_gb"])
        for r in ok:
            if r["method"].startswith("AirLLM"):
                continue
            log("  %-40s %+.0f%%" % (r["method"][:40],
                                     100 * (r["vram_gb"] / air["vram_gb"] - 1)))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
