import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from afterimage.runtime.config import EngineConfig
from afterimage.runtime.streaming_engine import compress_model_to_disk

MODEL = "Qwen/Qwen3-14B"
STORE = "/root/afterimage/store_14b"


def main() -> int:
    t0 = time.perf_counter()
    man = compress_model_to_disk(MODEL, STORE, config=EngineConfig(chunk_size=1024),
                                 progress_every=50)
    print("=" * 60)
    print("compress wall time: %.1fs" % (time.perf_counter() - t0))
    print("ORIGINAL  : %.3f GB" % (man["total_orig_bytes"]/1e9))
    print("COMPRESSED: %.3f GB" % (man["total_comp_bytes"]/1e9))
    print("RATIO     : %.3fx  (size %.1f%%)" % (man["ratio"], 100/man["ratio"]))
    return 0


if __name__ == "__main__":
    # See validate_streaming.py's __main__ guard comment: compress_model_to_
    # disk spawns worker processes that re-import this file, so nothing at
    # module level may run unguarded.
    raise SystemExit(main())
