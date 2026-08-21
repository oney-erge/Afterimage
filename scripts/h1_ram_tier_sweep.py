import sys, pathlib, time, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from afterimage.runtime.config import EngineConfig
from afterimage.runtime.streaming_engine import StreamingLosslessModel

MODEL = "Qwen/Qwen3-14B"
STORE = "/root/afterimage/store_14b"
PROMPT = "What is the capital of France?"
N_TOKENS = 4


def drop_caches():
    import subprocess
    try:
        subprocess.run(["sync"], check=True, timeout=60)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception as e:
        print("  WARNING: drop_caches failed: %r" % e, flush=True)


def run(ram_tier_format: str) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()

    import os
    ram_budget = float(os.environ.get("H1_RAM_BUDGET_GB", "4.0"))
    cfg = EngineConfig(vram_budget_gb=2.0, ram_budget_gb=ram_budget,
                       decode_slice_elems=1 << 22, io_prefetch_depth=2,
                       ram_tier_format=ram_tier_format)
    sm = StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg)

    tiers = {}
    for t in sm._tier.values():
        tiers[t] = tiers.get(t, 0) + 1
    ram_bytes_orig = sum(sm.manifest["tensors"][k]["orig_bytes"]
                         for k, t in sm._tier.items() if t == "ram")

    drop_caches()
    sm.stats.reset()
    t0 = time.perf_counter()
    seq = sm.generate_greedy(ids, max_new_tokens=N_TOKENS, use_cache=True)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    text = tok.decode(seq[0, ids.shape[1]:])
    result = {
        "ram_tier_format": ram_tier_format, "wall_s": wall, "s_per_tok": wall / N_TOKENS,
        "peak_vram_gb": peak, "answer": text, "tiers": tiers,
        "ram_tensor_count": tiers.get("ram", 0),
        "ram_resident_orig_gb": ram_bytes_orig / 1e9,
        "disk_tensor_count": tiers.get("disk", 0),
        "bytes_read": sm.stats.bytes_read,
        "gb_per_token": sm.stats.bytes_read / 1e9 / N_TOKENS,
        "io_s": sm.stats.io_seconds, "decode_s": sm.stats.decode_seconds,
    }
    sm.close()
    del sm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def main() -> int:
    print("=" * 72)
    print("H1 sweep: ram_tier_format, Qwen3-14B, vram=2.0GB ram=4.0GB, N=%d" % N_TOKENS)
    print("=" * 72)
    results = []
    for fmt in ["decoded", "compressed"]:
        print("\n--- ram_tier_format=%s ---" % fmt, flush=True)
        try:
            r = run(fmt)
            results.append(r)
            print("  s/tok=%.2f  peak=%.2fGB  ram_tensors=%d (%.2fGB orig)  disk_tensors=%d  "
                  "GB/tok=%.2f  answer=%r"
                  % (r["s_per_tok"], r["peak_vram_gb"], r["ram_tensor_count"],
                     r["ram_resident_orig_gb"], r["disk_tensor_count"], r["gb_per_token"],
                     r["answer"]))
        except Exception as e:
            print("  FAILED: %r" % e)
            import traceback; traceback.print_exc()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("%-14s %10s %10s %14s %14s" % ("format", "s/tok", "GB/tok", "ram_tensors", "disk_tensors"))
    for r in results:
        print("%-14s %10.2f %10.2f %14d %14d"
              % (r["ram_tier_format"], r["s_per_tok"], r["gb_per_token"],
                 r["ram_tensor_count"], r["disk_tensor_count"]))

    import json
    out = pathlib.Path("/root/afterimage/results/h1_ram_tier_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
