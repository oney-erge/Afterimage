"""AirLLM baseline -- the headline external comparator (LITERATURE.md #2).

NOT EXECUTED OR VERIFIED IN THIS DEVELOPMENT ENVIRONMENT: requires the
`airllm` package, CUDA, and a downloaded checkpoint, none of which are
present here (IMPLEMENTATION_STATUS.md). This wraps AirLLM's documented
`AutoModel.from_pretrained(...).generate(...)` interface; it has not been
run. IMPLEMENTATION_PLAN.md #1 requires ALSO reporting
baselines/b3_sequential.py (this codebase's own AirLLM-equivalent) alongside
this, precisely because AirLLM is a different codebase with different
kernels -- comparing only against this wrapper would confound method with
engine.
"""
from __future__ import annotations

import time


def run_airllm_baseline(model_name: str, prompt: str, max_new_tokens: int,
                         compression: str | None = "4bit") -> dict:
    try:
        from airllm import AutoModel
    except ImportError as e:
        raise ImportError(
            "b1_airllm requires the `airllm` package, not installed in this "
            "environment (see IMPLEMENTATION_STATUS.md)."
        ) from e

    model = AutoModel.from_pretrained(model_name, compression=compression)
    input_ids = model.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)["input_ids"]

    t0 = time.perf_counter()
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, use_cache=True)
    dt = time.perf_counter() - t0

    n_generated = out.shape[1] - input_ids.shape[1] if hasattr(out, "shape") else max_new_tokens
    return {
        "tokens_generated": n_generated,
        "wall_seconds": dt,
        "tokens_per_second": n_generated / dt if dt > 0 else 0.0,
    }
