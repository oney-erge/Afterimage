import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
print("tie_word_embeddings:", getattr(cfg, "tie_word_embeddings", None))
man = json.loads(pathlib.Path("/root/afterimage/store_1.5b/manifest.json").read_text())
keys = list(man["tensors"].keys())
print("n tensors in store:", len(keys))
print("has lm_head.weight:", "lm_head.weight" in man["tensors"])
print("has model.embed_tokens.weight:", "model.embed_tokens.weight" in man["tensors"])
print("sample keys:", [k for k in keys if "embed" in k or "lm_head" in k or "norm" in k][:6])
