"""Hugging Face Accelerate big-model disk-offload baseline.

This comparator keeps the original BF16 checkpoint and lets Transformers /
Accelerate place complete modules across GPU, CPU and memory-mapped disk.  It
is not expected to be fast; its purpose is a second external, non-quantized
answer to the same "model exceeds VRAM" problem as AirLLM and Afterimage.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import time

import torch


@dataclasses.dataclass
class HFDiskOffloadBaseline:
    model: object
    tokenizer: object
    model_name: str
    offload_dir: str
    initialization_seconds: float
    device_map: dict

    def generate(self, prompt: str, max_new_tokens: int) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_device = self.model.device
        if input_device.type == "meta":
            mapped = {str(value) for value in self.device_map.values()}
            input_device = torch.device("cuda:0" if any(
                value == "0" or value.startswith("cuda") for value in mapped)
                else "cpu")
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, eos_token_id=None, pad_token_id=None)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        generated = output[0, inputs["input_ids"].shape[1]:].detach().cpu()
        return {
            "tokens_generated": int(generated.numel()),
            "output_token_ids": generated.tolist(),
            "text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "wall_seconds": wall,
            "seconds_per_token": wall / max(int(generated.numel()), 1),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        }


def load_hf_offload_baseline(model_name: str, offload_dir: str, *,
                             gpu_memory: str = "1500MB",
                             cpu_memory: str = "8GB") -> HFDiskOffloadBaseline:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Install the model baseline dependencies with "
            "`pip install -e .[bench]`.") from exc
    if importlib.util.find_spec("accelerate") is None:
        raise ImportError(
            "Hugging Face disk offload requires Accelerate; install the "
            "benchmark extras with `pip install -e .[bench]`.")

    offload_path = pathlib.Path(offload_dir)
    offload_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={0: gpu_memory, "cpu": cpu_memory},
        offload_folder=str(offload_path),
        offload_state_dict=True,
    )
    return HFDiskOffloadBaseline(
        model=model, tokenizer=tokenizer, model_name=model_name,
        offload_dir=str(offload_path),
        initialization_seconds=time.perf_counter() - started,
        device_map=dict(getattr(model, "hf_device_map", {})))


def run_hf_offload_baseline(model_name: str, offload_dir: str, prompt: str,
                            max_new_tokens: int, *, gpu_memory: str = "1500MB",
                            cpu_memory: str = "8GB") -> dict:
    """Compatibility wrapper for callers that want a one-shot baseline."""
    baseline = load_hf_offload_baseline(
        model_name, offload_dir, gpu_memory=gpu_memory, cpu_memory=cpu_memory)
    result = baseline.generate(prompt, max_new_tokens)
    result["initialization_seconds"] = baseline.initialization_seconds
    return result
