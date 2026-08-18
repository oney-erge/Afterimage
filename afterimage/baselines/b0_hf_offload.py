"""HuggingFace `accelerate` disk-offload baseline.

NOT EXECUTED OR VERIFIED IN THIS DEVELOPMENT ENVIRONMENT: requires
`transformers` + `accelerate` (not installed here, no CUDA available -- see
IMPLEMENTATION_STATUS.md) and a downloaded checkpoint. This is real,
reasonable integration code against accelerate's documented
`infer_auto_device_map` / `dispatch_model` API, but it has not been run.
Treat it as a starting point to validate on the actual benchmarking rig, not
as a tested component.
"""
from __future__ import annotations

import time


def run_hf_offload_baseline(model_name: str, offload_dir: str, prompt: str,
                             max_new_tokens: int, device_map: str = "auto") -> dict:
    try:
        from accelerate import infer_auto_device_map, dispatch_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "b0_hf_offload requires `transformers` and `accelerate`, not "
            "installed in this environment (see IMPLEMENTATION_STATUS.md). "
            "Install with `pip install -e .[models]` on the real benchmarking rig."
        ) from e

    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    device_map_result = infer_auto_device_map(
        model, max_memory={0: "7GiB", "cpu": "14GiB"}, no_split_module_classes=["DecoderLayer"]
    )
    model = dispatch_model(model, device_map=device_map_result, offload_dir=offload_dir)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    dt = time.perf_counter() - t0

    n_generated = out.shape[1] - inputs["input_ids"].shape[1]
    return {
        "tokens_generated": n_generated,
        "wall_seconds": dt,
        "tokens_per_second": n_generated / dt if dt > 0 else 0.0,
    }
