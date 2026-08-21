import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from afterimage.runtime.config import EngineConfig
from afterimage.runtime.streaming_engine import StreamingLosslessModel

MODEL = "Qwen/Qwen3-14B"
STORE = "/root/afterimage/store_14b"
PROMPT = "The capital of France is"
N = 10


def run(use_cache: bool):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    cfg = EngineConfig(vram_cap_gb=6.0, empty_cache_every=1, io_prefetch_depth=2)
    sm = StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg)
    t0 = time.perf_counter()
    seq = sm.generate_greedy(ids, max_new_tokens=N, use_cache=use_cache)
    wall = time.perf_counter() - t0
    text = tok.decode(seq[0, ids.shape[1]:])
    sm.close()
    return seq[0, ids.shape[1]:].tolist(), text, wall


def main() -> int:
    print("=== use_cache=True ===", flush=True)
    cached_tokens, cached_text, cached_wall = run(True)
    print("tokens=%r text=%r wall=%.1fs" % (cached_tokens, cached_text, cached_wall))

    print("\n=== use_cache=False ===", flush=True)
    nocache_tokens, nocache_text, nocache_wall = run(False)
    print("tokens=%r text=%r wall=%.1fs" % (nocache_tokens, nocache_text, nocache_wall))

    match = cached_tokens == nocache_tokens
    print("\nMATCH: %s" % match)
    print("speedup from caching: %.2fx" % (nocache_wall / cached_wall))
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
