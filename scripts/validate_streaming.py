import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from afterimage.runtime.streaming_engine import compress_model_to_disk, StreamingLosslessModel

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
STORE = "/root/afterimage/store_1.5b"


def main() -> int:
    print("=== compressing model to disk ===", flush=True)
    t0 = time.perf_counter()
    man = compress_model_to_disk(MODEL, STORE, chunk_size=1024, quantize=None)
    print("compress time: %.1fs" % (time.perf_counter() - t0))
    print("orig %.3f GB -> comp %.3f GB  (%.3fx)" % (
        man["total_orig_bytes"]/1e9, man["total_comp_bytes"]/1e9, man["ratio"]))

    print("\n=== streaming engine forward vs reference ===", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()

    sm = StreamingLosslessModel(MODEL, STORE, device="cuda")
    t0 = time.perf_counter()
    logits_stream = sm.forward_logits(ids)
    t_stream = time.perf_counter() - t0
    print("streaming forward: %.2fs  bytes_read=%.1f MB  layer_loads=%d" % (
        t_stream, sm.stats.bytes_read/1e6, sm.stats.layer_loads))
    print("  io=%.2fs decode=%.2fs compute=%.2fs" % (
        sm.stats.io_seconds, sm.stats.decode_seconds, sm.stats.compute_seconds))
    sm.close()

    ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        logits_ref = ref(input_ids=ids, use_cache=False).logits

    exact = torch.equal(logits_stream.view(torch.int16), logits_ref.view(torch.int16))
    maxdiff = (logits_stream.float() - logits_ref.float()).abs().max().item()
    print("\nLOGITS BIT-EXACT vs reference: %s   max_abs_diff=%s" % (exact, maxdiff))
    print("top token stream=%d ref=%d" % (
        logits_stream[0,-1].argmax().item(), logits_ref[0,-1].argmax().item()))
    return 0


if __name__ == "__main__":
    # compress_model_to_disk pools tensor compression across worker
    # processes started with the "spawn" method (fork was tried first and
    # deadlocked -- forking a process that has already touched torch's
    # OpenMP-backed CPU thread pool hangs every worker at 0% CPU). spawn
    # re-imports this file in each worker to bootstrap it, so anything at
    # module level that shouldn't re-run in a worker -- all of it, here --
    # must sit behind this guard; without it every worker re-ran the whole
    # script recursively, which is what actually happened before this was
    # added.
    raise SystemExit(main())
