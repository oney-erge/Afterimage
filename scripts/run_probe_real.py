#!/usr/bin/env python3
"""Phase 0 probe against a REAL transformer (IMPLEMENTATION_PLAN.md #2).

This is the actual decision-gate measurement, not a toy-model stand-in:
does within-session activation rank get low enough, at a real model's real
layers, for the Afterimage cache to have anything to work with.

Requires `transformers`, `accelerate`, `torch` with (ideally) CUDA. Falls
back to CPU automatically -- slower, but correct, for a model this size.

Usage (inside the WSL venv, after `pip install -e .[models]`):
    python3 scripts/run_probe_real.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from afterimage.probe.closed_loop import calibrate_bases, closed_loop_error, open_loop_error
from afterimage.probe.hooks import ActivationCapture
from afterimage.probe.spectra import layer_rank_report
from afterimage.probe.workloads import FOCUSED_CODE, LONG_FORM_PROSE, MULTI_TURN_CHAT, topic_switch_prompts


def log(msg: str) -> None:
    print(msg, flush=True)


class LogitsOnly(nn.Module):
    """Wraps an HF CausalLM so `forward(input_ids) -> (N, vocab)` -- last-
    REAL-token logits only, matching the (N, d) shape closed_loop.py's error
    functions expect (one output vector per one input row).

    Sequences are right-padded to a common length for batching, so index -1
    is the position after however many pad tokens follow the real content --
    for most rows that is NOT the position that actually predicts the next
    real token. This class carries the per-row real length (from the
    tokenizer's attention_mask) and gathers each row's logits at its own
    last real position instead of a fixed index."""

    def __init__(self, hf_model: nn.Module, attention_mask: torch.Tensor):
        super().__init__()
        self.hf_model = hf_model
        self.last_real_idx = attention_mask.sum(dim=1) - 1  # (N,)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.hf_model(input_ids=input_ids, use_cache=False)
        logits = out.logits  # (N, seq_len, vocab)
        idx = self.last_real_idx.to(logits.device)
        return logits[torch.arange(logits.shape[0], device=logits.device), idx, :]


def pick_target_layers(model: nn.Module, n_depths: int = 6) -> list[str]:
    """Picks one MLP down_proj per layer at n_depths evenly spaced depths --
    down_proj is where the FFN's dimensionality expansion collapses back
    down, a natural place to look for compressibility. Falls back to any
    Linear if the expected Qwen2-style naming isn't found."""
    layers = model.model.layers
    n_layers = len(layers)
    depths = sorted(set(int(round(i * (n_layers - 1) / (n_depths - 1))) for i in range(n_depths)))
    names = []
    for d in depths:
        candidate = f"model.layers.{d}.mlp.down_proj"
        try:
            mod = model.get_submodule(candidate)
            if isinstance(mod, nn.Linear):
                names.append(candidate)
                continue
        except AttributeError:
            pass
        # fallback: first Linear found in that layer block
        for name, mod in layers[d].named_modules():
            if isinstance(mod, nn.Linear):
                names.append(f"model.layers.{d}.{name}")
                break
    return names


def tokenize_batch(tokenizer, texts: list[str], max_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic (longest-in-batch) right-padding, not a fixed max_length --
    minimizes the fraction of pad tokens rather than padding every short
    example out to a fixed budget. Returns (input_ids, attention_mask); the
    mask is what lets callers exclude pad-position activations."""
    enc = tokenizer(texts, padding="longest", truncation=True, max_length=max_len,
                     return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def measure_workload(model, tokenizer, target_layers: list[str], texts: list[str],
                      max_len: int, device: str, ranks: list[int]) -> dict:
    """Right-padding means pad positions still produce forward-pass
    activations for the pad token embedding -- a repeated, near-constant
    vector that would artificially deflate any rank measurement if included
    (caught during development; see hooks.py::stacked_masked). Causal
    attention makes this safe on the input side: pad tokens strictly after
    all real tokens cannot affect any real token's own activations, so the
    only contamination risk is capturing the pad positions themselves, which
    stacked_masked filters out."""
    ids, mask = tokenize_batch(tokenizer, texts, max_len, device)
    log(f"    tokenized: {ids.shape}, real tokens: {int(mask.sum().item())}")

    with ActivationCapture(model, layer_names=target_layers) as cap:
        with torch.no_grad():
            t0 = time.perf_counter()
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            log(f"    forward pass: {time.perf_counter()-t0:.2f}s")

    per_layer = {}
    for name in target_layers:
        t0 = time.perf_counter()
        X = cap.stacked_masked(name, mask).float()
        W = model.get_submodule(name).weight.float()
        valid_ranks = [r for r in ranks if r < min(X.shape[0], X.shape[1])]
        if not valid_ranks:
            log(f"    {name}: skipped (only {X.shape[0]} activation rows)")
            continue
        report = layer_rank_report(X, W, valid_ranks)
        per_layer[name] = report
        log(f"    {name}: N={X.shape[0]} d_in={X.shape[1]} "
            f"eff_rank={report['effective_rank']:.1f}  ({time.perf_counter()-t0:.2f}s)")
    return per_layer


def measure_closed_loop(model, tokenizer, target_layers: list[str],
                         calib_texts: list[str], eval_texts: list[str],
                         max_len: int, device: str, ranks: list[int]) -> dict:
    """target_layers are dotted paths on the raw HF model (e.g.
    "model.layers.0.mlp.down_proj"), used as-is by measure_workload, which
    hooks the raw model directly. Here the model actually passed to
    calibrate_bases/open_loop_error/closed_loop_error is LogitsOnly, which
    stores the real model one level deeper as `self.hf_model` -- so
    LogitsOnly.named_modules() yields "hf_model.model.layers.0..." paths,
    not "model.layers.0...". Passing the unprefixed names silently matched
    nothing: ActivationCapture.attach() never created a dict entry for them,
    and the first read crashed with KeyError instead of quietly returning
    empty (caught by an actual run against a real model; a toy-model test
    with a flat, unwrapped module tree could not have exposed this)."""
    wrapped_layers = [f"hf_model.{name}" for name in target_layers]

    calib_ids, calib_mask = tokenize_batch(tokenizer, calib_texts, max_len, device)
    eval_ids, eval_mask = tokenize_batch(tokenizer, eval_texts, max_len, device)

    calib_wrapped = LogitsOnly(model, calib_mask)
    eval_wrapped = LogitsOnly(model, eval_mask)

    out = {}
    for r in ranks:
        t0 = time.perf_counter()
        bases = calibrate_bases(calib_wrapped, calib_ids, wrapped_layers, rank=r,
                                 attention_mask=calib_mask)
        ol = open_loop_error(eval_wrapped, bases, eval_ids, wrapped_layers,
                              attention_mask=eval_mask)
        cl = closed_loop_error(eval_wrapped, bases, eval_ids, wrapped_layers)
        out[r] = {"open_loop_mean": sum(ol.values()) / len(ol), "closed_loop": cl}
        log(f"    rank={r}: closed_loop={cl:.4f}  ({time.perf_counter()-t0:.2f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--n-depths", type=int, default=6)
    ap.add_argument("--ranks", default="4,8,16,32,64,128,256")
    ap.add_argument("--closed-loop-ranks", default="8,32,128")
    ap.add_argument("--workloads", default="focused_code,multi_turn_chat,long_form_prose,adversarial_topic_switch")
    ap.add_argument("--skip-closed-loop", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ranks_arg = [int(x) for x in args.ranks.split(",")]
    cl_ranks_arg = [int(x) for x in args.closed_loop_ranks.split(",")]
    workloads_arg = set(args.workloads.split(","))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")
    log(f"loading {args.model} ...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    log(f"from_pretrained().to(device) call returned in {time.perf_counter()-t0:.1f}s "
        f"(CUDA ops are async -- this call returning is NOT proof the transfer finished)")
    if device == "cuda":
        t_sync = time.perf_counter()
        torch.cuda.synchronize()
        log(f"torch.cuda.synchronize() took an additional {time.perf_counter()-t_sync:.1f}s "
            f"-- this is when the weight transfer actually finished")
    log(f"total model-ready time: {time.perf_counter()-t0:.1f}s. "
        f"n_layers={len(model.model.layers)} d_model={model.config.hidden_size}")

    target_layers = pick_target_layers(model, n_depths=args.n_depths)
    log(f"target layers ({len(target_layers)}): {target_layers}")

    results = {"model": args.model, "target_layers": target_layers, "workloads": {}}

    all_workloads = {
        "focused_code": FOCUSED_CODE,
        "multi_turn_chat": MULTI_TURN_CHAT,
        "long_form_prose": LONG_FORM_PROSE,
        "adversarial_topic_switch": topic_switch_prompts(),
    }
    workloads = {k: v for k, v in all_workloads.items() if k in workloads_arg}

    for wl_name, texts in workloads.items():
        log(f"\n--- {wl_name} ({len(texts)} examples) ---")
        t0 = time.perf_counter()
        per_layer = measure_workload(model, tokenizer, target_layers, texts, args.max_len, device, ranks_arg)
        results["workloads"][wl_name] = {"rank_curves": per_layer}
        for name, r in per_layer.items():
            log(f"  {name}: eff_rank={r['effective_rank']:.1f}/{r['d_in']}  "
                f"var@rank0={r['variance_captured'][0]:.3f}  "
                f"func_err@rank0={r['functional_error'][0]:.3f}")
        log(f"  ({time.perf_counter()-t0:.1f}s)")

    if not args.skip_closed_loop and "focused_code" in workloads_arg:
        log("\n--- closed-loop: narrow (focused_code) vs adversarial (topic-switch) ---")
        n = len(FOCUSED_CODE)
        split = n // 2
        cl_narrow = measure_closed_loop(model, tokenizer, target_layers,
                                         FOCUSED_CODE[:split], FOCUSED_CODE[split:],
                                         args.max_len, device, cl_ranks_arg)
        adv = topic_switch_prompts()
        asplit = len(adv) // 2
        cl_adv = measure_closed_loop(model, tokenizer, target_layers,
                                      adv[:asplit], adv[asplit:],
                                      args.max_len, device, cl_ranks_arg)
        results["closed_loop"] = {"narrow_focused_code": cl_narrow, "adversarial_topic_switch": cl_adv}
        for r in cl_ranks_arg:
            log(f"  rank={r:>4}  narrow closed_loop={cl_narrow[r]['closed_loop']:.4f}   "
                f"adversarial closed_loop={cl_adv[r]['closed_loop']:.4f}")

    out_path = pathlib.Path(args.out) if args.out else pathlib.Path.home() / "afterimage" / "results" / "phase0_real.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
