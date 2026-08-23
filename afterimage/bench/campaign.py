"""Durable helpers for staged cross-model hardware campaigns."""
from __future__ import annotations

import json
import pathlib
import shutil
import time
from typing import Any


def atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_campaign_config(path: pathlib.Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("cross-model campaign schema_version must be 1")
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("cross-model campaign must declare models")
    model_ids = [model.get("id") for model in models]
    if any(not value for value in model_ids) or len(model_ids) != len(set(model_ids)):
        raise ValueError("campaign model ids must be non-empty and unique")
    for model in models:
        stage_ids = [stage.get("id") for stage in model.get("benchmarks", [])]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("benchmark ids must be unique within %s" % model["id"])
        if not model.get("model_id") or not model.get("store"):
            raise ValueError("%s needs model_id and store" % model["id"])
    return config


def model_preflight(model: dict) -> dict:
    """Resolve access, exact Transformers shards, layout, and host capacity."""
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    model_id = model["model_id"]
    info = HfApi().model_info(model_id, files_metadata=True)
    sizes = {entry.rfilename: int(entry.size or 0) for entry in info.siblings}
    try:
        index_path = pathlib.Path(hf_hub_download(
            model_id, "model.safetensors.index.json"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_files = sorted(set(index["weight_map"].values()))
    except Exception:
        weight_files = ["model.safetensors"]
    missing_metadata = [name for name in weight_files if name not in sizes]
    if missing_metadata:
        raise RuntimeError("Hub metadata is missing indexed weights: %s" % missing_metadata)

    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        candidate = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    inner = getattr(candidate, "model", None)
    layout = {
        "model_class": type(candidate).__name__,
        "has_model": inner is not None,
        "has_layers": hasattr(inner, "layers"),
        "has_embed_tokens": hasattr(inner, "embed_tokens"),
        "has_lm_head": hasattr(candidate, "lm_head"),
    }
    layout["afterimage_compatible"] = all(layout[key] for key in (
        "has_model", "has_layers", "has_embed_tokens", "has_lm_head"))
    del candidate

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, fix_mistral_regex=True)
    store_parent = pathlib.Path(model["store"]).parent
    store_parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(store_parent)
    weight_bytes = sum(sizes[name] for name in weight_files)
    hidden = int(getattr(config, "hidden_size", 0) or 0)
    vocab = int(getattr(config, "vocab_size", 0) or 0)
    return {
        "status": "passed" if layout["afterimage_compatible"] else "failed",
        "model_id": model_id,
        "revision": info.sha,
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", []) or []),
        "dimensions": {
            "hidden_size": hidden,
            "intermediate_size": getattr(config, "intermediate_size", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "num_key_value_heads": getattr(config, "num_key_value_heads", None),
            "vocab_size": vocab,
            "tie_word_embeddings": bool(
                getattr(config, "tie_word_embeddings", False)),
        },
        "estimated_bf16_head_bytes": hidden * vocab * 2,
        "transformers_weight_files": weight_files,
        "transformers_weight_bytes": weight_bytes,
        "declared_expected_weight_bytes": model.get("expected_bf16_weight_bytes"),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": len(tokenizer),
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "layout": layout,
        "storage": {
            "path": str(store_parent),
            "free_bytes": disk.free,
            "required_preflight_bytes": int(weight_bytes * 1.9),
            "capacity_passed": disk.free >= int(weight_bytes * 1.9),
        },
        "completed_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
    }


def summarize_result(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "artifact": str(path),
        "status": payload.get("status"),
        "failure_count": len(payload.get("failures", [])),
    }
    if "methods" in payload:
        summary["methods"] = [{
            "method": entry.get("method_id"),
            "status": "failed" if entry.get("error") else "measured",
            "completed_cases": entry.get("summary", {}).get("completed_cases", 0),
            "seconds_per_token": entry.get("summary", {}).get("seconds_per_token"),
            "peak_vram_gb": entry.get("summary", {}).get("peak_vram_gb"),
            "expected_match_rate": entry.get("summary", {}).get(
                "expected_match_rate"),
            "error": entry.get("error"),
        } for entry in payload["methods"]]
    elif payload.get("method"):
        row = payload.get("summary", {})
        summary["methods"] = [{
            "method": payload["method"],
            "status": "measured",
            "completed_cases": len(payload.get("rows", [])),
            "seconds_per_token": row.get("seconds_per_token"),
            "peak_vram_gb": row.get("peak_vram_gb"),
            "expected_match_rate": row.get("expected_match_rate"),
        }]
    elif "analysis" in payload:
        analysis = payload["analysis"]
        summary["hypothesis_id"] = payload.get("hypothesis_id")
        summary["completed_pairs"] = analysis.get("completed_pairs")
        summary["paired_token_exact"] = analysis.get("paired_token_exact")
        summary["paired_effect"] = analysis.get("paired_effect")
        summary["mechanism_gate"] = analysis.get("mechanism_gate")
        summary["advance_to_l3"] = analysis.get("advance_to_l3")
    return summary


def render_campaign_markdown(campaign: dict) -> str:
    lines = [
        "# Cross-family scale benchmark: interim results",
        "",
        "This file is regenerated after every campaign stage. A row marked `running`",
        "or `failed` is an interim operational fact, not a research conclusion.",
        "",
        "Campaign status: **%s**" % campaign.get("status", "unknown"),
        "",
        "| Model | Role | Stage | Status | Result |",
        "|---|---|---|---|---|",
    ]
    for model in campaign.get("models", []):
        stages = model.get("stages", [])
        if not stages:
            lines.append("| %s | %s | preflight | pending | — |" % (
                model.get("model_id"), model.get("role")))
            continue
        for stage in stages:
            detail = stage.get("error") or stage.get("artifact") or stage.get("log") or "—"
            lines.append("| %s | %s | %s | %s | `%s` |" % (
                model.get("model_id"), model.get("role"), stage.get("id"),
                stage.get("status"), str(detail).replace("|", "\\|")))

    lines.extend(["", "## Measured method rows", "",
                  "| Model | Stage | Method | s/token | Peak VRAM GB | Cases |",
                  "|---|---|---|---:|---:|---:|"])
    any_rows = False
    for model in campaign.get("models", []):
        for stage in model.get("stages", []):
            for method in stage.get("summary", {}).get("methods", []):
                any_rows = True
                lines.append("| %s | %s | %s | %s | %s | %s |" % (
                    model.get("id"), stage.get("id"), method.get("method"),
                    _format_number(method.get("seconds_per_token")),
                    _format_number(method.get("peak_vram_gb")),
                    method.get("completed_cases", 0)))
    if not any_rows:
        lines.append("| — | — | — | — | — | — |")

    lines.extend(["", "## Deferred and excluded strata", ""])
    for item in campaign.get("deferred_models", []):
        lines.append("- `%s`: %s" % (item["model_id"], item["reason"]))
    for item in campaign.get("excluded_hypothesis_families", []):
        lines.append("- `%s`: %s" % (", ".join(item["ids"]), item["reason"]))
    lines.append("")
    return "\n".join(lines)


def _format_number(value: Any) -> str:
    return "—" if value is None else "%.4f" % float(value)
