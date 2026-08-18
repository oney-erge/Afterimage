import pathlib, json, subprocess
from transformers import AutoConfig
d = pathlib.Path("/root/afterimage/store_14b")
done = sum(f.stat().st_size for f in d.glob("*.npz"))
n = len(list(d.glob("*.npz")))
cfg = AutoConfig.from_pretrained("Qwen/Qwen3-14B")
h, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
tie = getattr(cfg, "tie_word_embeddings", False)
inter = cfg.intermediate_size
print("Qwen3-14B: hidden=%d layers=%d vocab=%d ffn=%d tied=%s" % (h, L, V, inter, tie))
embed = V*h*2/1e9
per_layer = (4*h*h + 3*h*inter)*2/1e9
print("  embed_tokens : %.2f GB" % embed)
print("  lm_head      : %.2f GB %s" % (0 if tie else embed, "(tied->free)" if tie else ""))
print("  ONE layer    : %.2f GB" % per_layer)
peak = embed + (0 if tie else embed) + per_layer
print("  => PEAK VRAM (streaming) ~ %.2f GB  + activations" % peak)
print()
est_total = 20.0
print("COMPRESSION PROGRESS: %.1f GB written, %d tensors  (~%.0f%% of ~%.0f GB est)"
      % (done/1e9, n, 100*done/1e9/est_total, est_total))
