import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from afterimage.runtime.config import EngineConfig
from afterimage.runtime.streaming_engine import StreamingLosslessModel

MODEL = "Qwen/Qwen3-14B"
STORE = "/root/afterimage/store_14b"
PROMPT = "The capital of France is"
EXPECTED_TOKENS = [12095, 13]  # known-correct: " Paris."


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()

    t0 = time.perf_counter()
    cfg = EngineConfig(vram_cap_gb=6.0, empty_cache_every=1, progress=True,
                       io_prefetch_depth=2)
    sm = StreamingLosslessModel(MODEL, STORE, device="cuda", config=cfg)
    print("engine init: %.1fs" % (time.perf_counter() - t0), flush=True)

    t0 = time.perf_counter()
    seq = sm.generate_greedy(ids, max_new_tokens=2, use_cache=True)
    wall = time.perf_counter() - t0

    got = seq[0, ids.shape[1]:].tolist()
    text = tok.decode(seq[0, ids.shape[1]:])
    print("wall=%.1fs  tokens=%r  text=%r" % (wall, got, text))
    print("io=%.2fs decode=%.2fs compute=%.2fs bytes_read=%.2fGB" % (
        sm.stats.io_seconds, sm.stats.decode_seconds, sm.stats.compute_seconds,
        sm.stats.bytes_read / 1e9))
    match = got == EXPECTED_TOKENS
    print("MATCHES KNOWN-CORRECT OUTPUT %r: %s" % (EXPECTED_TOKENS, match))
    sm.close()
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
