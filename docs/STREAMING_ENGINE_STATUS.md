# Streaming Engine — Status

Built to answer one question directly: **can a model that fits in neither
VRAM nor RAM be run losslessly, and does compressing the streamed bytes beat
AirLLM?**

## What is built and verified

| Piece | Status |
|---|---|
| Entropy codec (canonical Huffman, exponent field) | working, bit-exact |
| GPU decoder (Triton, warp-vectorized) | working, 32/32 tests |
| Full weight reconstruction (sign+mantissa+exponent) | working, bit-exact |
| Offline model compression (lazy, one tensor at a time) | working |
| **Layer-streaming inference engine** | **working, bit-exact** |
| Q8 as opt-in config (lossless default) | working, 6/6 tests |
| Speculative decoding | **not built** |

**121 tests passing**, 32 GPU tests passing under CUDA.

## Verified lossless, end to end, on a real model

Qwen2.5-1.5B, streamed layer-by-layer from the compressed store:

```
LOGITS BIT-EXACT vs reference: True   max_abs_diff=0.0
top token stream=12095  ref=12095
```

Every decoder layer was materialized from entropy-coded bytes on the GPU,
run, and freed — and the result is *identical* to the untouched model, not
approximately equal.

Compression on that model: **3.087 GB -> 2.117 GB (1.458x)**.

## Three real bugs found by building it

1. **Broken weight tying.** Qwen sets `tie_word_embeddings=True`, so the
   checkpoint contains no `lm_head.weight` — it aliases `embed_tokens`.
   Calling `to_empty()` per module allocated a fresh tensor and silently
   broke that alias, leaving `lm_head` on uninitialized memory. Nothing
   crashed; the model returned confident nonsense (max logit diff 22.1,
   argmax 100628 vs the correct 12095). Now re-tied, and any parameter with
   no source raises instead of running on garbage.
2. **Computed buffers treated as checkpoint weights.** `rotary_emb.inv_freq`
   is non-persistent — derived from config, never stored. It must be
   *recomputed*, not looked up.
3. **Hand-rolled forward omitted the causal mask.** Replaced by forward
   hooks so transformers drives masking/rotary/cache and this engine only
   manages weight residency. Less brittle and correct by delegation.

## Performance finding that matters

Isolated, the decode kernel hits **16.87 GB/s**. In situ it delivers
**1.84 GB/s** — a 9x gap. The kernel is ~1.6 ms of a ~15 ms per-layer path;
the rest is `.npz` container parsing (~1 GB/s ceiling), host-to-device
transfer, and per-tensor Python/allocation overhead.

**This is the deciding factor for the AirLLM comparison.** Compression
removes ~31% of the bytes, but if reconstructing them costs more time than
the removed bytes would have taken to read, the net is a loss. Identified
fixes, none yet applied: raw binary store instead of `.npz`, pinned-memory
staging, and batching per-layer transfers. Changing the store format means
recompressing, so it is deferred until the current measurement lands.

## Configuration

Lossless is the default and Q8 is explicitly opt-in (`EngineConfig`), because
AirLLM does not quantize either — quantizing by default would make the
head-to-head measure two different things.

```python
EngineConfig()                  # lossless, bit-exact  (default)
EngineConfig(quantize="q8")     # ~2.0x, 0.55% error   (opt-in, declares itself lossy)
```
