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


def run(fraction: float) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()

    cfg = EngineConfig(vram_budget_gb=2.0, decode_slice_elems=1 << 22,
                       io_prefetch_depth=2, cpu_decode_fraction=fraction)
    sm = StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg)

    drop_caches()
    sm.stats.reset()
    t0 = time.perf_counter()
    seq = sm.generate_greedy(ids, max_new_tokens=N_TOKENS, use_cache=True)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    text = tok.decode(seq[0, ids.shape[1]:])
    result = {
        "cpu_decode_fraction": fraction, "wall_s": wall, "s_per_tok": wall / N_TOKENS,
        "peak_vram_gb": peak, "answer": text,
        "io_s": sm.stats.io_seconds, "gpu_decode_s": sm.stats.decode_seconds,
        "cpu_decode_s": sm.stats.cpu_decode_seconds, "compute_s": sm.stats.compute_seconds,
        "cpu_decoded_tensors": sm.stats.cpu_decoded_tensors,
        "gpu_decoded_tensors": sm.stats.gpu_decoded_tensors,
        "bytes_read": sm.stats.bytes_read,
    }
    sm.close()
    del sm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def main() -> int:
    print("=" * 72)
    print("H2 sweep: cpu_decode_fraction, Qwen3-14B, vram_budget_gb=2.0, N=%d" % N_TOKENS)
    print("=" * 72)
    results = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print("\n--- cpu_decode_fraction=%.2f ---" % frac, flush=True)
        try:
            r = run(frac)
            results.append(r)
            print("  s/tok=%.2f  peak=%.2fGB  io=%.1fs  gpu_decode=%.1fs  cpu_decode=%.1fs  "
                  "cpu_tensors=%d  gpu_tensors=%d  answer=%r"
                  % (r["s_per_tok"], r["peak_vram_gb"], r["io_s"], r["gpu_decode_s"],
                     r["cpu_decode_s"], r["cpu_decoded_tensors"], r["gpu_decoded_tensors"],
                     r["answer"]))
        except Exception as e:
            print("  FAILED: %r" % e)
            import traceback; traceback.print_exc()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("%-10s %10s %10s %10s %10s" % ("fraction", "s/tok", "peak GB", "gpu_dec_s", "cpu_dec_s"))
    base = results[0]["s_per_tok"] if results else None
    for r in results:
        print("%-10.2f %10.2f %10.2f %10.2f %10.2f  (%.2fx vs baseline)"
              % (r["cpu_decode_fraction"], r["s_per_tok"], r["peak_vram_gb"],
                 r["gpu_decode_s"], r["cpu_decode_s"], base / r["s_per_tok"] if base else 1.0))

    import json
    out = pathlib.Path("/root/afterimage/results/h2_cpu_decode_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
