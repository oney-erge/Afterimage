"""Read-only discovery of model files already present on this computer."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def _matches(query: str, value: str) -> bool:
    return not query or query.casefold() in value.casefold()


def huggingface_cache_models(query: str = "") -> list[dict[str, Any]]:
    """Return cached Hub model snapshots without making a network request."""

    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
    except Exception:  # noqa: BLE001 - a corrupt/missing cache is not fatal
        return []

    models: list[dict[str, Any]] = []
    for repo in cache.repos:
        if repo.repo_type != "model" or not _matches(query, repo.repo_id):
            continue
        revisions = sorted(
            repo.revisions,
            key=lambda item: float(getattr(item, "last_modified", 0) or 0),
            reverse=True,
        )
        if not revisions:
            continue
        revision = revisions[0]
        names = {item.file_name for item in revision.files}
        safetensors = any(name.endswith(".safetensors") for name in names)
        has_config = "config.json" in names
        models.append({
            "model_id": repo.repo_id,
            "source": "huggingface-cache",
            "source_label": "Hugging Face cache",
            "revision": revision.commit_hash,
            "snapshot_path": str(revision.snapshot_path),
            "size_bytes": int(repo.size_on_disk),
            "format": "safetensors" if safetensors else "other",
            "can_prepare": bool(safetensors and has_config),
            "message": (
                "Cached source is ready for Afterimage inspection."
                if safetensors and has_config
                else "Cached files are not a complete safetensors language-model snapshot."
            ),
        })
    return models


def ollama_models(query: str = "", *, timeout: float = 0.35) -> list[dict[str, Any]]:
    """Inspect an already-running local Ollama service; never start it."""

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/tags",
        headers={"Accept": "application/json", "User-Agent": "Afterimage/local-discovery"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return []

    models = []
    for item in payload.get("models", []):
        model_id = str(item.get("name") or item.get("model") or "").strip()
        if not model_id or not _matches(query, model_id):
            continue
        details = item.get("details") or {}
        models.append({
            "model_id": model_id,
            "source": "ollama",
            "source_label": "Ollama",
            "revision": item.get("digest"),
            "size_bytes": int(item.get("size") or 0),
            "format": str(details.get("quantization_level") or "GGUF"),
            "family": details.get("family"),
            "can_prepare": False,
            "external_url": "http://127.0.0.1:8000/ui",
            "message": (
                "This quantized Ollama model is already on disk. Afterimage needs the "
                "original Hugging Face safetensors checkpoint for lossless preparation."
            ),
        })
    return models


def discover_local_models(query: str = "") -> dict[str, Any]:
    rows = huggingface_cache_models(query) + ollama_models(query)
    rows.sort(key=lambda row: (row["source"] != "huggingface-cache", row["model_id"].casefold()))
    return {
        "models": rows,
        "sources": {
            "huggingface_cache": sum(row["source"] == "huggingface-cache" for row in rows),
            "ollama": sum(row["source"] == "ollama" for row in rows),
        },
    }
