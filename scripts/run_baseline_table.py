#!/usr/bin/env python3
"""Fills the reference/baseline rows of VALIDATION_PLAN.md #2 with MEASURED
numbers -- checkpoint size, peak VRAM, perplexity, and greedy output tokens
for the token-identity comparison.

This needs no Afterimage runtime: it establishes the bar that any method must
beat, on the actual target hardware. Run it per (model, dtype) config.

Usage (inside the WSL venv):
    python -u scripts/run_baseline_table.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype fp16
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from afterimage.bench.accuracy import perplexity_from_logits, token_identity_rate
from afterimage.bench.memory import MemoryProbe, checkpoint_bytes
from afterimage.probe.workloads import FOCUSED_CODE, LONG_FORM_PROSE, MULTI_TURN_CHAT


def log(msg: str) -> None:
    print(msg, flush=True)


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_snapshot_dir(model_id: str) -> pathlib.Path | None:
    """Locates the on-disk snapshot so checkpoint size is measured, not
    assumed from a parameter count."""
    try:
        from huggingface_hub import snapshot_download
        return pathlib.Path(snapshot_download(model_id))
    except Exception:
        return None


def greedy_generate(model, tokenizer, prompts: list[str], max_new_tokens: int,
                     device: str) -> list[list[int]]:
    """Deterministic greedy decode, one prompt at a time.

    One at a time rather than batched on purpose: batched generation with
    right-padding can change results at the margins depending on how a given
    model handles pad positions, and this output feeds the token-identity
    comparison where a single differing token is treated as a failure. The
    comparison must not be contaminated by batching artifacts.
    """
    out = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        out.append(gen[0, ids.shape[1]:].tolist())
    return out


def measure_perplexity(model, tokenizer, texts: list[str], max_len: int, device: str) -> float:
    enc = tokenizer(texts, padding="longest", truncation=True, max_length=max_len,
                     return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    return perplexity_from_logits(logits, ids, mask)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--n-gen-prompts", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPES[args.dtype]
    log(f"model={args.model}  dtype={args.dtype}  device={device}")

    snapshot = resolve_snapshot_dir(args.model)
    ckpt_bytes = checkpoint_bytes(snapshot) if snapshot else None
    if ckpt_bytes:
        log(f"checkpoint on disk: {ckpt_bytes/1e9:.2f} GB  ({snapshot})")
    else:
        log("checkpoint on disk: UNKNOWN (snapshot dir not resolved)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = (FOCUSED_CODE + MULTI_TURN_CHAT + LONG_FORM_PROSE)[: args.n_gen_prompts]
    ppl_texts = (LONG_FORM_PROSE + MULTI_TURN_CHAT)[:8]

    with MemoryProbe(interval_s=0.05) as probe:
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
        model.eval()
        if device == "cuda":
            torch.cuda.synchronize()
        load_s = time.perf_counter() - t0
        log(f"model ready in {load_s:.1f}s")

        t0 = time.perf_counter()
        ppl = measure_perplexity(model, tokenizer, ppl_texts, args.max_len, device)
        log(f"perplexity: {ppl:.4f}  ({time.perf_counter()-t0:.1f}s)")

        t0 = time.perf_counter()
        gen_tokens = greedy_generate(model, tokenizer, prompts, args.max_new_tokens, device)
        gen_s = time.perf_counter() - t0
        n_tokens = sum(len(g) for g in gen_tokens)
        log(f"generated {n_tokens} tokens across {len(prompts)} prompts in {gen_s:.1f}s "
            f"({n_tokens/gen_s:.1f} tok/s)")

    mem = probe.report()
    log("")
    log("--- MEMORY ---")
    log(f"  checkpoint on disk : {ckpt_bytes/1e9:.2f} GB" if ckpt_bytes else "  checkpoint on disk : n/a")
    log(f"  torch peak VRAM    : {mem.torch_peak_vram_gb:.2f} GB" if mem.torch_peak_vram_gb else "  torch peak VRAM    : n/a")
    log(f"  nvidia-smi baseline: {mem.smi_baseline_used_mb} MiB")
    log(f"  nvidia-smi peak    : {mem.smi_peak_used_mb} MiB  ({mem.n_samples} samples)")
    log(f"  smi delta (this run): {mem.smi_delta_gb:.2f} GB" if mem.smi_delta_gb is not None else "")
    log(f"  host RSS peak      : {mem.host_rss_peak_gb:.2f} GB" if mem.host_rss_peak_gb else "")

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "device": device,
        "checkpoint_bytes": ckpt_bytes,
        "torch_peak_vram_bytes": mem.torch_peak_vram_bytes,
        "smi_baseline_used_mb": mem.smi_baseline_used_mb,
        "smi_peak_used_mb": mem.smi_peak_used_mb,
        "smi_delta_gb": mem.smi_delta_gb,
        "host_rss_peak_bytes": mem.host_rss_peak_bytes,
        "perplexity": ppl,
        "load_seconds": load_s,
        "generation_tok_per_s": n_tokens / gen_s if gen_s else None,
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "generated_token_ids": gen_tokens,
    }

    stem = args.model.replace("/", "__")
    out_path = pathlib.Path(args.out) if args.out else (
        pathlib.Path.home() / "afterimage" / "results" / f"baseline_{stem}_{args.dtype}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
