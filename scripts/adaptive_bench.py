#!/usr/bin/env python3
"""Real-hardware harness for docs/archive/ADAPTIVE_TEST_PLAN.md / docs/archive/PROPOSAL_ADAPTIVE.md.

Enforces the test plan's Rule 1 (every arm gets the SAME total VRAM budget,
they differ only in how they spend it) mechanically:

  Arm A (greedy)        engine vram_budget_gb = --total-vram-gb
  Arm B (small draft)   engine vram_budget_gb = --total-vram-gb - measured
                         draft-model resident footprint (~1.2 GB, Qwen3-0.6B)
  Arm C (self-draft)    engine vram_budget_gb = --total-vram-gb  (unchanged --
                         self-drafting has no cost OUTSIDE the planner's own
                         budget: the draft layers are ordinary decoder layers
                         the SAME vram_budget_gb already governs. pin_draft_
                         layers only changes how that one pool is RANKED, not
                         its size. This is a real, load-bearing difference
                         from arm B, not an oversight.)

Peak VRAM is always MEASURED (torch.cuda.max_memory_allocated), never
assumed -- reported alongside every timing so a reader can check Rule 1 was
actually honoured, not just requested.

Correctness for every arm that uses generate_adaptive / generate_speculative
is checked by running at temperature=0.0: verify.temperature_probs makes
speculative decoding provably reproduce generate_greedy's argmax sequence
token-for-token regardless of draft quality/k/policy (see its docstring and
docs/archive/ADAPTIVE_TEST_PLAN.md Sec 3) -- callers of this script that want that
guarantee checked should pass --temperature 0 and diff token_ids against a
run_greedy() result themselves; this harness does not assert it inline so it
can also be used for the realistic temperature>0 regime.

Subcommands:
  t0   coupling check    -- k x vram_budget grid, existing generate_speculative
  t1   matched-VRAM arms -- A / B / C at one total budget
  t3   pinning ablation  -- C with pin_draft_layers on vs off
  t2   policy comparison -- C with spec_k_policy fixed/gamma/threshold
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
DRAFT_MODEL_ID = "Qwen/Qwen3-0.6B"
STORE = "/root/afterimage/store_14b"
PROMPT = "What is the capital of France?"
N_LAYERS = 40


def log(m: str) -> None:
    print(m, flush=True)


def drop_caches() -> None:
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


def _tokenize(prompt: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    return tok, tok(prompt, return_tensors="pt").input_ids.cuda()


def run_greedy(prompt: str, n_tokens: int, vram_budget_gb: float) -> dict:
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    tok, ids = _tokenize(prompt)
    cfg = EngineConfig(vram_budget_gb=vram_budget_gb, io_prefetch_depth=2,
                       decode_slice_elems=1 << 22)
    drop_caches()
    _reset_peak()
    t0 = time.perf_counter()
    sm = StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg)
    seq = sm.generate_greedy(ids, max_new_tokens=n_tokens, use_cache=False)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    text = tok.decode(seq[0, ids.shape[1]:])
    result = dict(arm="A_greedy", vram_budget_gb=vram_budget_gb, peak_vram_gb=peak,
                 wall_s=wall, s_per_tok=wall / n_tokens, tok_per_s=n_tokens / wall,
                 bytes_read=sm.stats.bytes_read, gb_per_token=sm.stats.bytes_read / 1e9 / n_tokens,
                 io_s=sm.stats.io_seconds, decode_s=sm.stats.decode_seconds,
                 compute_s=sm.stats.compute_seconds, prompt=prompt, answer=text,
                 token_ids=seq[0, ids.shape[1]:].tolist())
    sm.close()
    del sm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_speculative_model(prompt: str, n_tokens: int, vram_budget_gb: float,
                          k: int = 8, temperature: float = 1.0,
                          spec_k_policy: str = "fixed") -> dict:
    """Arm B. Uses generate_adaptive(draft_mode="model"), NOT the older
    generate_speculative -- generate_speculative's softmax(logits/temperature)
    was never built to support temperature=0 (divides by zero; confirmed the
    hard way, see docs/RESULTS_LOG.md's adaptive-bench entry) and this
    harness needs temperature=0 for the token-identical-to-greedy
    correctness check shared by every arm. generate_adaptive's
    verify.temperature_probs handles it safely, and draft_mode="model"
    reduces to plain small-model speculative decoding either way -- same
    mechanism, different entry point."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel, load_draft_model

    tok, ids = _tokenize(prompt)
    _reset_peak()
    draft = load_draft_model(DRAFT_MODEL_ID, device="cuda")
    draft_peak_gb = torch.cuda.max_memory_allocated() / 1e9

    cfg = EngineConfig(vram_budget_gb=vram_budget_gb, io_prefetch_depth=2,
                       decode_slice_elems=1 << 22, draft_mode="model", spec_k=k,
                       spec_k_policy=spec_k_policy)
    drop_caches()
    _reset_peak()
    t0 = time.perf_counter()
    sm = StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg)
    gen = torch.Generator(device="cuda").manual_seed(0)
    seq, policy = sm.generate_adaptive(ids, max_new_tokens=n_tokens, draft_model=draft,
                                       temperature=temperature, generator=gen)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    text = tok.decode(seq[0, ids.shape[1]:])
    n_out = seq.shape[1] - ids.shape[1]
    result = dict(arm="B_model_draft", vram_budget_gb=vram_budget_gb,
                 draft_peak_vram_gb=draft_peak_gb, peak_vram_gb=peak,
                 wall_s=wall, s_per_tok=wall / n_out, tok_per_s=n_out / wall,
                 spec_sweeps=sm.stats.spec_sweeps, spec_accepted=sm.stats.spec_accepted_tokens,
                 tok_per_sweep=n_out / max(1, sm.stats.spec_sweeps), k=k,
                 spec_k_policy=spec_k_policy, final_k=policy.choose_k(),
                 policy_state=policy.state_dict(),
                 acceptance=sm.stats.spec_accepted_tokens / max(1, sm.stats.spec_sweeps * k),
                 bytes_read=sm.stats.bytes_read, gb_per_token=sm.stats.bytes_read / 1e9 / n_out,
                 io_s=sm.stats.io_seconds, decode_s=sm.stats.decode_seconds,
                 prompt=prompt, answer=text, token_ids=seq[0, ids.shape[1]:].tolist())
    sm.close()
    del sm, draft
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_self_draft(prompt: str, n_tokens: int, vram_budget_gb: float, exit_layer: int = 4,
                   spec_k: int = 8, spec_k_policy: str = "fixed",
                   pin_draft_layers: bool = True, temperature: float = 1.0) -> dict:
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    tok, ids = _tokenize(prompt)
    cfg = EngineConfig(vram_budget_gb=vram_budget_gb, io_prefetch_depth=2,
                       decode_slice_elems=1 << 22, draft_mode="self",
                       draft_exit_layer=exit_layer, spec_k=spec_k,
                       spec_k_policy=spec_k_policy, pin_draft_layers=pin_draft_layers)
    drop_caches()
    _reset_peak()
    t0 = time.perf_counter()
    sm = StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg)
    gen = torch.Generator(device="cuda").manual_seed(0)
    seq, policy = sm.generate_adaptive(ids, max_new_tokens=n_tokens,
                                       temperature=temperature, generator=gen)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    text = tok.decode(seq[0, ids.shape[1]:])
    n_out = seq.shape[1] - ids.shape[1]
    result = dict(arm="C_self_draft", vram_budget_gb=vram_budget_gb, exit_layer=exit_layer,
                 spec_k=spec_k, spec_k_policy=spec_k_policy, pin_draft_layers=pin_draft_layers,
                 peak_vram_gb=peak, wall_s=wall, s_per_tok=wall / n_out, tok_per_s=n_out / wall,
                 spec_sweeps=sm.stats.spec_sweeps, spec_accepted=sm.stats.spec_accepted_tokens,
                 tok_per_sweep=n_out / max(1, sm.stats.spec_sweeps),
                 acceptance=sm.stats.spec_accepted_tokens / max(1, sm.stats.spec_sweeps * spec_k),
                 final_k=policy.choose_k(), policy_state=policy.state_dict(),
                 bytes_read=sm.stats.bytes_read, gb_per_token=sm.stats.bytes_read / 1e9 / n_out,
                 io_s=sm.stats.io_seconds, decode_s=sm.stats.decode_seconds,
                 prompt=prompt, answer=text, token_ids=seq[0, ids.shape[1]:].tolist())
    sm.close()
    del sm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _write(results: list, out: str) -> None:
    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2))
    log("\nwrote %s" % p)


def cmd_t0(args) -> int:
    log("=" * 76)
    log("T0 -- coupling check (does best-k shift with VRAM budget?)")
    log("=" * 76)
    results = []
    ks = [int(x) for x in args.ks.split(",")]
    budgets = [float(x) for x in args.budgets.split(",")]
    for budget in budgets:
        for k in ks:
            log("--- budget=%.1fGB k=%d ---" % (budget, k))
            try:
                r = run_speculative_model(args.prompt, args.n_tokens, budget, k=k,
                                          temperature=args.temperature)
                results.append(r)
                log("  %.2f s/tok  tok/sweep=%.2f  accept=%.1f%%  peak=%.2fGB"
                    % (r["s_per_tok"], r["tok_per_sweep"], 100 * r["acceptance"], r["peak_vram_gb"]))
            except Exception as e:
                log("  FAILED: %r" % e)
                import traceback
                traceback.print_exc()
                results.append({"arm": "B_model_draft", "vram_budget_gb": budget, "k": k,
                               "FAILED": repr(e)})
    log("\nbest k per budget:")
    for budget in budgets:
        rows = [r for r in results if r.get("vram_budget_gb") == budget and "FAILED" not in r]
        if rows:
            best = min(rows, key=lambda r: r["s_per_tok"])
            log("  %.1fGB -> k=%d (%.2f s/tok)" % (budget, best["k"], best["s_per_tok"]))
    _write(results, args.out)
    return 0


def cmd_t1(args) -> int:
    log("=" * 76)
    log("T1 -- matched-VRAM arm comparison @ total=%.1fGB" % args.total_vram_gb)
    log("=" * 76)
    results = []

    log("--- Arm A: greedy ---")
    try:
        r = run_greedy(args.prompt, args.n_tokens, args.total_vram_gb)
        results.append(r)
        log("  peak=%.2fGB  %.2f s/tok  answer=%r" % (r["peak_vram_gb"], r["s_per_tok"], r["answer"]))
    except Exception as e:
        log("  FAILED: %r" % e)
        import traceback
        traceback.print_exc()

    log("--- Arm B: small draft model (Qwen3-0.6B) ---")
    try:
        b_budget = args.total_vram_gb - args.draft_model_gb
        r = run_speculative_model(args.prompt, args.n_tokens, b_budget, k=args.spec_k,
                                  temperature=args.temperature)
        results.append(r)
        log("  peak=%.2fGB (engine budget %.2fGB + ~%.2fGB draft model)  %.2f s/tok  tok/sweep=%.2f  answer=%r"
            % (r["peak_vram_gb"], b_budget, args.draft_model_gb, r["s_per_tok"], r["tok_per_sweep"], r["answer"]))
    except Exception as e:
        log("  FAILED: %r" % e)
        import traceback
        traceback.print_exc()

    log("--- Arm C: self-draft (exit_layer=%d, pinned) ---" % args.exit_layer)
    try:
        r = run_self_draft(args.prompt, args.n_tokens, args.total_vram_gb,
                           exit_layer=args.exit_layer, spec_k=args.spec_k,
                           pin_draft_layers=True, temperature=args.temperature)
        results.append(r)
        log("  peak=%.2fGB  %.2f s/tok  tok/sweep=%.2f  answer=%r"
            % (r["peak_vram_gb"], r["s_per_tok"], r["tok_per_sweep"], r["answer"]))
    except Exception as e:
        log("  FAILED: %r" % e)
        import traceback
        traceback.print_exc()

    log("\n" + "=" * 76)
    log("SUMMARY (total budget %.1fGB)" % args.total_vram_gb)
    log("=" * 76)
    log("%-16s %10s %10s %12s" % ("arm", "peak VRAM", "s/token", "tok/sweep"))
    for r in results:
        log("%-16s %9.2fG %10.2f %12s" % (
            r.get("arm", "?"), r.get("peak_vram_gb", -1), r.get("s_per_tok", -1),
            ("%.2f" % r["tok_per_sweep"]) if "tok_per_sweep" in r else "-"))

    _write(results, args.out)
    return 0


def cmd_t3(args) -> int:
    log("=" * 76)
    log("T3 -- pinning ablation @ total=%.1fGB, exit_layer=%d" % (args.total_vram_gb, args.exit_layer))
    log("=" * 76)
    results = []
    for pin in (True, False):
        log("--- pin_draft_layers=%s ---" % pin)
        try:
            r = run_self_draft(args.prompt, args.n_tokens, args.total_vram_gb,
                               exit_layer=args.exit_layer, spec_k=args.spec_k,
                               pin_draft_layers=pin, temperature=args.temperature)
            results.append(r)
            log("  peak=%.2fGB  %.2f s/tok  tok/sweep=%.2f"
                % (r["peak_vram_gb"], r["s_per_tok"], r["tok_per_sweep"]))
        except Exception as e:
            log("  FAILED: %r" % e)
            import traceback
            traceback.print_exc()
            results.append({"arm": "C_self_draft", "pin_draft_layers": pin, "FAILED": repr(e)})
    if len(results) == 2 and "FAILED" not in results[0] and "FAILED" not in results[1]:
        on, off = results[0], results[1]
        log("\npinned  : %.2f s/tok" % on["s_per_tok"])
        log("unpinned: %.2f s/tok" % off["s_per_tok"])
        log("pinning helps: %s (%.2fx)" % (on["s_per_tok"] < off["s_per_tok"],
                                           off["s_per_tok"] / on["s_per_tok"]))
    _write(results, args.out)
    return 0


def cmd_t2(args) -> int:
    log("=" * 76)
    log("T2 -- policy comparison @ total=%.1fGB, draft_mode=%s" % (args.total_vram_gb, args.draft_mode))
    log("=" * 76)
    results = []
    for policy_name in ("fixed", "gamma", "threshold"):
        log("--- spec_k_policy=%s ---" % policy_name)
        try:
            if args.draft_mode == "self":
                r = run_self_draft(args.prompt, args.n_tokens, args.total_vram_gb,
                                   exit_layer=args.exit_layer, spec_k=args.spec_k,
                                   spec_k_policy=policy_name, pin_draft_layers=True,
                                   temperature=args.temperature)
            else:
                r = run_speculative_model(args.prompt, args.n_tokens,
                                          args.total_vram_gb - args.draft_model_gb,
                                          k=args.spec_k, temperature=args.temperature,
                                          spec_k_policy=policy_name)
            results.append(r)
            log("  peak=%.2fGB  %.2f s/tok  tok/sweep=%.2f  sweeps=%d  final_k=%s"
                % (r["peak_vram_gb"], r["s_per_tok"], r["tok_per_sweep"],
                   r["spec_sweeps"], r.get("final_k")))
        except Exception as e:
            log("  FAILED: %r" % e)
            import traceback
            traceback.print_exc()
            results.append({"arm": args.draft_mode, "spec_k_policy": policy_name, "FAILED": repr(e)})
    log("\n%-10s %10s %8s %8s" % ("policy", "s/token", "sweeps", "final_k"))
    for r in results:
        if "FAILED" not in r:
            log("%-10s %10.2f %8d %8s" % (r["spec_k_policy"], r["s_per_tok"],
                                          r["spec_sweeps"], r.get("final_k")))
    _write(results, args.out)
    return 0


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

    return dict(arm="airllm", peak_vram_gb=peak, wall_s=wall, s_per_tok=wall / n_tokens,
               tok_per_s=n_tokens / wall, init_s=init_s, prompt=prompt, answer=text,
               token_ids=gen.tolist())


def cmd_t4(args) -> int:
    log("=" * 76)
    log("T4 -- final head-to-head vs AirLLM, swept across VRAM budgets")
    log("=" * 76)
    results = []

    log("--- AirLLM (reference, sets the VRAM floor everything else is matched against) ---")
    try:
        air = run_airllm(args.prompt, args.n_tokens)
        results.append(air)
        log("  peak=%.2fGB  %.2f s/tok  answer=%r" % (air["peak_vram_gb"], air["s_per_tok"], air["answer"]))
    except Exception as e:
        air = None
        log("  FAILED: %r" % e)
        import traceback
        traceback.print_exc()

    budgets = [float(x) for x in args.budgets.split(",")]
    for budget in budgets:
        log("--- Afterimage greedy @ total=%.2fGB ---" % budget)
        try:
            r = run_greedy(args.prompt, args.n_tokens, budget)
            results.append(r)
            log("  peak=%.2fGB  %.2f s/tok  answer=%r" % (r["peak_vram_gb"], r["s_per_tok"], r["answer"]))
        except Exception as e:
            log("  FAILED (infeasible at this budget?): %r" % e)
            results.append({"arm": "A_greedy", "vram_budget_gb": budget, "FAILED": repr(e)})

        log("--- Afterimage speculative (small draft model) @ total=%.2fGB "
            "(engine gets %.2fGB after the draft model's own ~%.2fGB) ---"
            % (budget, budget - args.draft_model_gb, args.draft_model_gb))
        try:
            b_budget = budget - args.draft_model_gb
            r = run_speculative_model(args.prompt, args.n_tokens, b_budget, k=args.spec_k,
                                      temperature=args.temperature)
            results.append(r)
            log("  peak=%.2fGB  %.2f s/tok  tok/sweep=%.2f  answer=%r"
                % (r["peak_vram_gb"], r["s_per_tok"], r["tok_per_sweep"], r["answer"]))
        except Exception as e:
            log("  FAILED (infeasible at this budget -- draft model's own footprint may leave "
                "too little for the target model's tiering): %r" % e)
            results.append({"arm": "B_model_draft", "vram_budget_gb": budget, "FAILED": repr(e)})

    log("\n" + "=" * 76)
    log("SUMMARY")
    log("=" * 76)
    log("%-16s %10s %10s %10s" % ("arm", "budget", "peak VRAM", "s/token"))
    if air:
        log("%-16s %10s %9.2fG %10.2f" % ("airllm", "-", air["peak_vram_gb"], air["s_per_tok"]))
    for r in results:
        if r.get("arm") == "airllm" or "FAILED" in r:
            continue
        log("%-16s %10.2f %9.2fG %10.2f" % (r["arm"], r["vram_budget_gb"], r["peak_vram_gb"], r["s_per_tok"]))
    for r in results:
        if "FAILED" in r:
            log("%-16s %10.2f %9s %10s" % (r["arm"], r["vram_budget_gb"], "INFEASIBLE", "-"))

    if air:
        log("\nSpeedup at comparable peak VRAM:")
        for r in results:
            if r.get("arm") == "airllm" or "FAILED" in r:
                continue
            log("  %-14s @ %.2fGB budget: %.2fx  (%.2f GB actual vs AirLLM %.2f GB)"
                % (r["arm"], r["vram_budget_gb"], air["s_per_tok"] / r["s_per_tok"],
                   r["peak_vram_gb"], air["peak_vram_gb"]))

    _write(results, args.out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = dict()

    p0 = sub.add_parser("t0")
    p0.add_argument("--prompt", default=PROMPT)
    p0.add_argument("--n-tokens", type=int, default=3)
    p0.add_argument("--ks", default="2,4,8,12")
    p0.add_argument("--budgets", default="2.5,4.0,6.0")
    p0.add_argument("--temperature", type=float, default=1.0)
    p0.add_argument("--out", default="/root/afterimage/results/adaptive_t0.json")
    p0.set_defaults(func=cmd_t0)

    p1 = sub.add_parser("t1")
    p1.add_argument("--prompt", default=PROMPT)
    p1.add_argument("--n-tokens", type=int, default=4)
    p1.add_argument("--total-vram-gb", type=float, default=6.0)
    p1.add_argument("--draft-model-gb", type=float, default=1.3)
    p1.add_argument("--exit-layer", type=int, default=4)
    p1.add_argument("--spec-k", type=int, default=8)
    p1.add_argument("--temperature", type=float, default=0.0)
    p1.add_argument("--out", default="/root/afterimage/results/adaptive_t1.json")
    p1.set_defaults(func=cmd_t1)

    p3 = sub.add_parser("t3")
    p3.add_argument("--prompt", default=PROMPT)
    p3.add_argument("--n-tokens", type=int, default=4)
    p3.add_argument("--total-vram-gb", type=float, default=6.0)
    p3.add_argument("--exit-layer", type=int, default=4)
    p3.add_argument("--spec-k", type=int, default=8)
    p3.add_argument("--temperature", type=float, default=0.0)
    p3.add_argument("--out", default="/root/afterimage/results/adaptive_t3.json")
    p3.set_defaults(func=cmd_t3)

    p2 = sub.add_parser("t2")
    p2.add_argument("--prompt", default=PROMPT)
    p2.add_argument("--n-tokens", type=int, default=6)
    p2.add_argument("--total-vram-gb", type=float, default=6.0)
    p2.add_argument("--draft-mode", default="model", choices=["model", "self"])
    p2.add_argument("--draft-model-gb", type=float, default=1.3)
    p2.add_argument("--exit-layer", type=int, default=4)
    p2.add_argument("--spec-k", type=int, default=8)
    p2.add_argument("--temperature", type=float, default=1.0)
    p2.add_argument("--out", default="/root/afterimage/results/adaptive_t2.json")
    p2.set_defaults(func=cmd_t2)

    p4 = sub.add_parser("t4")
    p4.add_argument("--prompt", default=PROMPT)
    p4.add_argument("--n-tokens", type=int, default=6)
    p4.add_argument("--budgets", default="2.1,3.0,4.0,6.0")
    p4.add_argument("--draft-model-gb", type=float, default=1.3)
    p4.add_argument("--spec-k", type=int, default=8)
    p4.add_argument("--temperature", type=float, default=0.0)
    p4.add_argument("--out", default="/root/afterimage/results/adaptive_t4.json")
    p4.set_defaults(func=cmd_t4)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
