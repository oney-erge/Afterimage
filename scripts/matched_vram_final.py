#!/usr/bin/env python3
"""THE apples-to-apples number: both systems at the SAME peak VRAM, s/token.

Everything else this project measures is secondary to this one table. The
rule is simple and it is the rule that was previously broken: a speed claim
against AirLLM is only valid if the measured peak VRAM matches. "Faster
while holding 2.4x the memory" is not a result, it is a bigger machine.

AirLLM cannot be told to use MORE memory (it streams one layer at a time;
its footprint is whatever its largest tensor forces, ~1.57 GB on this
model). So matching means bringing Afterimage DOWN to AirLLM's floor, not
meeting in the middle. That floor is set by lm_head (1.556 GB), which BOTH
systems must materialize to produce logits -- so both are bounded by the
same tensor, which is what makes the comparison fair.

Sweeps Afterimage from its minimum feasible budget upward and reports every
configuration's MEASURED peak next to AirLLM's, so the matched pair is
identifiable by inspection rather than by assertion.

Both systems are lossless here. AirLLM also ships a lossy block-quantized
mode (4/8-bit) that is faster; it is deliberately not used, because this
engine's entire premise is bit-exact output and comparing against a lossy
configuration would measure two different things.
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
    import subprocess
    try:
        subprocess.run(["sync"], check=True, timeout=60)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception as e:
        log("  WARNING: could not drop caches (%s)" % e)


def disk_read_bytes() -> int:
    """Actual block-device bytes this process has read. With caches dropped
    this is the real I/O volume -- the quantity compression exists to shrink,
    measured rather than inferred from the manifest."""
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _reset_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_airllm(prompt: str, n_tokens: int) -> dict:
    from airllm import AutoModel

    model = AutoModel.from_pretrained(MODEL)
    # Pass a real attention mask. The previous baseline call used
    # return_attention_mask=False, which made transformers warn
    # ("attention mask is not set ... you may observe unexpected behavior")
    # and produced visibly off-topic text. Timing/VRAM were unaffected, but
    # a worked-example transcript that shows the baseline answering the
    # wrong question is not a fair presentation of the baseline.
    enc = model.tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].cuda()
    attn = enc.get("attention_mask")
    kw = {"attention_mask": attn.cuda()} if attn is not None else {}

    drop_caches()
    _reset_peak()
    io0 = disk_read_bytes()
    t0 = time.perf_counter()
    out = model.generate(input_ids, max_new_tokens=n_tokens, use_cache=True,
                         return_dict_in_generate=True, **kw)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    read_gb = (disk_read_bytes() - io0) / 1e9

    seq = out.sequences if hasattr(out, "sequences") else out
    gen = seq[0, input_ids.shape[1]:]
    text = model.tokenizer.decode(gen)

    del model, out, seq
    gc.collect()
    torch.cuda.empty_cache()

    return dict(system="AirLLM", config="default (streams every layer)",
                peak_vram_gb=peak, s_per_tok=wall / n_tokens, wall_s=wall,
                disk_gb_per_tok=read_gb / n_tokens, answer=text,
                token_ids=gen.tolist())


def run_ours(prompt: str, n_tokens: int, budget_gb: float, slice_elems: int) -> dict:
    from transformers import AutoTokenizer
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()

    cfg = EngineConfig(vram_budget_gb=budget_gb, io_prefetch_depth=2,
                       decode_slice_elems=slice_elems, empty_cache_every=1)
    sm = StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg)

    drop_caches()
    sm.stats.reset()
    _reset_peak()
    io0 = disk_read_bytes()
    t0 = time.perf_counter()
    seq = sm.generate_greedy(ids, max_new_tokens=n_tokens, use_cache=True)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    read_gb = (disk_read_bytes() - io0) / 1e9

    text = tok.decode(seq[0, ids.shape[1]:])
    n_vram = sum(1 for v in sm._tier.values() if v == "vram")
    result = dict(system="Afterimage", config="budget=%.2fGB slice=2^%d"
                  % (budget_gb, slice_elems.bit_length() - 1),
                  budget_gb=budget_gb, slice_elems=slice_elems,
                  peak_vram_gb=peak, s_per_tok=wall / n_tokens, wall_s=wall,
                  disk_gb_per_tok=read_gb / n_tokens,
                  manifest_gb_per_tok=sm.stats.bytes_read / 1e9 / n_tokens,
                  io_s=sm.stats.io_seconds, decode_s=sm.stats.decode_seconds,
                  compute_s=sm.stats.compute_seconds, vram_resident_tensors=n_vram,
                  answer=text, token_ids=seq[0, ids.shape[1]:].tolist())
    sm.close()
    del sm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=8)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--configs", default="1.70:20,1.75:20,1.90:20,2.10:22",
                    help="comma-separated budget_gb:log2(slice_elems)")
    ap.add_argument("--out", default="/root/afterimage/results/matched_vram_final.json")
    args = ap.parse_args()

    results = []
    log("=" * 78)
    log("MATCHED-VRAM COMPARISON -- both systems, same peak VRAM, s/token")
    log("=" * 78)
    log("prompt: %r   tokens: %d   lossless both sides" % (args.prompt, args.n_tokens))
    log("")

    log("--- AirLLM (sets the floor: cannot be given more VRAM) ---")
    try:
        air = run_ours if False else run_airllm(args.prompt, args.n_tokens)
        results.append(air)
        log("  peak %.2f GB   %.2f s/tok   %.1f GB read/tok" %
            (air["peak_vram_gb"], air["s_per_tok"], air["disk_gb_per_tok"]))
        log("  answer: %r" % air["answer"])
    except Exception as e:
        air = None
        log("  FAILED: %r" % e)
        import traceback; traceback.print_exc()
    log("")

    for spec in args.configs.split(","):
        if not spec.strip():
            continue
        b, s = spec.split(":")
        budget, slice_elems = float(b), 1 << int(s)
        log("--- Afterimage @ budget %.2f GB, slice 2^%s ---" % (budget, s))
        try:
            r = run_ours(args.prompt, args.n_tokens, budget, slice_elems)
            results.append(r)
            log("  peak %.2f GB   %.2f s/tok   %.1f GB read/tok   %d tensors resident"
                % (r["peak_vram_gb"], r["s_per_tok"], r["disk_gb_per_tok"],
                   r["vram_resident_tensors"]))
            log("  answer: %r" % r["answer"])
        except Exception as e:
            log("  FAILED: %r" % e)
            results.append({"system": "Afterimage", "budget_gb": budget,
                            "slice_elems": slice_elems, "FAILED": repr(e)})
        log("")

    log("=" * 78)
    log("%-34s %11s %11s %13s" % ("configuration", "peak VRAM", "s/token", "GB read/tok"))
    log("-" * 78)
    for r in results:
        if "FAILED" in r:
            log("%-34s %11s %11s %13s" % (r["config"] if "config" in r else
                                          "budget=%.2f" % r["budget_gb"],
                                          "INFEASIBLE", "-", "-"))
            continue
        log("%-34s %10.2fG %11.2f %13.1f" % (
            (r["system"] + ": " + r["config"])[:34], r["peak_vram_gb"],
            r["s_per_tok"], r["disk_gb_per_tok"]))

    if air:
        log("")
        log("MATCHED-VRAM VERDICT (only configs at or below AirLLM's %.2f GB count):"
            % air["peak_vram_gb"])
        matched = [r for r in results if r.get("system") == "Afterimage"
                   and "FAILED" not in r and r["peak_vram_gb"] <= air["peak_vram_gb"]]
        near = [r for r in results if r.get("system") == "Afterimage"
                and "FAILED" not in r
                and air["peak_vram_gb"] < r["peak_vram_gb"] <= air["peak_vram_gb"] * 1.05]
        if matched:
            best = min(matched, key=lambda r: r["s_per_tok"])
            log("  AT OR BELOW AirLLM's VRAM: %.2f GB, %.2f s/tok = %.2fx faster"
                % (best["peak_vram_gb"], best["s_per_tok"],
                   air["s_per_tok"] / best["s_per_tok"]))
        for r in near:
            log("  within 5%% of AirLLM's VRAM: %.2f GB (%+.1f%%), %.2f s/tok = %.2fx"
                % (r["peak_vram_gb"],
                   100 * (r["peak_vram_gb"] / air["peak_vram_gb"] - 1),
                   r["s_per_tok"], air["s_per_tok"] / r["s_per_tok"]))
        if not matched and not near:
            log("  NONE -- every Afterimage config used more VRAM than AirLLM. "
                "No matched-VRAM speed claim can be made from this run.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
