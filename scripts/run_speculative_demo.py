#!/usr/bin/env python3
"""Lever 5 demo: speculative decoding vs. plain greedy streaming, both on
the compressed Qwen3-14B store, with Qwen3-0.6B (same tokenizer/vocabulary,
fully resident) as the draft model.

This is a DIFFERENT decoding mode from the AirLLM head-to-head, which uses
generate_greedy specifically because its correctness check is token
identity. generate_speculative samples from the target's exact distribution
at temperature=1.0 -- see its docstring for why that cannot, and should not,
be judged by matching any other run's token sequence. The correctness
claim for THIS script is exact-distribution preservation, which
runtime/verify.py's tests already establish statistically; this script
measures speed and acceptance rate, not token identity.

Run:  python -u scripts/run_speculative_demo.py --n-tokens 20 --k 8
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

TARGET = "Qwen/Qwen3-14B"
DRAFT = "Qwen/Qwen3-0.6B"
STORE = "/root/afterimage/store_14b"
PROMPT = "The capital of France is"


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=20)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--vram-cap-gb", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from afterimage.runtime.streaming_engine import StreamingLosslessModel, load_draft_model

    tok = AutoTokenizer.from_pretrained(TARGET)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()

    log("--- loading draft model (%s, fully resident) ---" % DRAFT)
    t0 = time.perf_counter()
    draft = load_draft_model(DRAFT, device="cuda")
    log("  draft load: %.1fs" % (time.perf_counter() - t0))

    log("--- loading target engine (%s, streamed) ---" % TARGET)
    t0 = time.perf_counter()
    sm = StreamingLosslessModel(TARGET, STORE, device="cuda",
                                vram_cap_gb=args.vram_cap_gb, progress=True)
    log("  engine init: %.1fs" % (time.perf_counter() - t0))

    # Must match the device of the probability tensors sample_categorical
    # draws from (draft/target logits are both on cuda) -- torch.multinomial
    # requires the generator and the tensor to agree.
    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    drop_caches()
    sm.stats.reset()
    t0 = time.perf_counter()
    seq = sm.generate_speculative(ids, max_new_tokens=args.n_tokens, draft_model=draft,
                                  k=args.k, temperature=1.0, generator=gen)
    wall = time.perf_counter() - t0

    n_out = seq.shape[1] - ids.shape[1]
    text = tok.decode(seq[0, ids.shape[1]:])
    accept_rate = sm.stats.spec_accepted_tokens / max(1, sm.stats.spec_sweeps * args.k)
    peak_alloc = torch.cuda.max_memory_allocated() / 1e9

    log("")
    log("=" * 68)
    log("SPECULATIVE DECODING RESULT (k=%d)" % args.k)
    log("=" * 68)
    log("  tokens generated  : %d in %.2fs  (%.3f s/token, %.3f tok/s)"
        % (n_out, wall, wall / n_out, n_out / wall))
    log("  target sweeps     : %d  (%.2f tokens/sweep)"
        % (sm.stats.spec_sweeps, n_out / max(1, sm.stats.spec_sweeps)))
    log("  draft accept rate : %.1f%%" % (100 * accept_rate))
    log("  bytes read        : %.2f GB  (%.2f GB/sweep)"
        % (sm.stats.bytes_read / 1e9, sm.stats.bytes_read / 1e9 / max(1, sm.stats.spec_sweeps)))
    log("  peak VRAM live    : %.2f GB" % peak_alloc)
    log("  output            : %r" % text)
    log("")
    log("  NOTE: this is SAMPLED output (temperature=1.0), not the greedy")
    log("  argmax sequence generate_greedy/AirLLM comparisons use -- see")
    log("  generate_speculative's docstring. Correctness here means the")
    log("  target's exact distribution is preserved (proven in")
    log("  runtime/verify.py's tests), not token-identity with any other run.")

    out = pathlib.Path("/root/afterimage/results/speculative_14b.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "target": TARGET, "draft": DRAFT, "k": args.k, "seed": args.seed,
        "n_tokens": n_out, "wall_s": wall, "s_per_tok": wall / n_out,
        "tok_per_s": n_out / wall, "sweeps": sm.stats.spec_sweeps,
        "tokens_per_sweep": n_out / max(1, sm.stats.spec_sweeps),
        "accept_rate": accept_rate, "bytes_read": sm.stats.bytes_read,
        "peak_vram_live_gb": peak_alloc,
        "text": text, "token_ids": seq[0, ids.shape[1]:].tolist(),
    }, indent=2))
    log("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
