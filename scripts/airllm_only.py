import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch

MODEL = "Qwen/Qwen3-14B"
PROMPT = "The capital of France is"
N = 2

def drop_caches():
    import subprocess
    try:
        subprocess.run(["sync"], check=True, timeout=60)
        open("/proc/sys/vm/drop_caches","w").write("3\n")
        print("  caches dropped", flush=True)
    except Exception as e:
        print("  WARNING: cache drop failed: %r" % e, flush=True)

print("=== AIRLLM on %s ===" % MODEL, flush=True)
from airllm import AutoModel
t0 = time.perf_counter()
model = AutoModel.from_pretrained(MODEL)
print("  init/split: %.1fs" % (time.perf_counter()-t0), flush=True)

inp = model.tokenizer(PROMPT, return_tensors="pt", return_attention_mask=False, truncation=True)
drop_caches()
t0 = time.perf_counter()
out = model.generate(inp["input_ids"].cuda(), max_new_tokens=N, use_cache=True,
                     return_dict_in_generate=True)
wall = time.perf_counter()-t0
seq = out.sequences if hasattr(out,"sequences") else out
gen = seq[0, inp["input_ids"].shape[1]:]
peak = torch.cuda.max_memory_allocated()/1e9
print("  %.2f s/token   %.4f tok/s   peak VRAM %.2f GB" % (wall/N, N/wall, peak), flush=True)
print("  tokens: %s" % gen.tolist(), flush=True)
print("  text  : %r" % model.tokenizer.decode(gen), flush=True)
json.dump({"system":"airllm","wall_s":wall,"tokens":N,"s_per_tok":wall/N,
           "tok_per_s":N/wall,"peak_vram_gb":peak,"token_ids":gen.tolist()},
          open("/root/afterimage/results/airllm_14b.json","w"), indent=2)
