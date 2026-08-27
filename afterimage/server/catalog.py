"""Hugging Face model discovery with an opaque cursor and honest metadata."""
from __future__ import annotations

import base64
import json
from itertools import islice
from typing import Any

from afterimage.reference import MEASURED_REFERENCE
from afterimage.runtime.adapters import classify_config

_SORTS = {
    "downloads": "downloads",
    "recent": "last_modified",
    "likes": "likes",
    "trending": "trending_score",
}


def _cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"offset": offset}).encode()).decode().rstrip("=")


def _offset(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded).decode())
        return max(0, int(value["offset"]))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid catalog cursor") from exc


def _model_row(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None) or {}
    compatibility = classify_config(config)
    safetensors = getattr(model, "safetensors", None)
    params = getattr(safetensors, "total", None)
    params_b = params / 1e9 if params else None
    estimated_source = params_b * MEASURED_REFERENCE["bf16_gb_per_b_params"] if params_b else None
    estimated_store = params_b * MEASURED_REFERENCE["compressed_gb_per_b_params"] if params_b else None
    gated = getattr(model, "gated", False)
    return {
        "model_id": model.id,
        "revision": getattr(model, "sha", None),
        "downloads": getattr(model, "downloads", None),
        "likes": getattr(model, "likes", None),
        "last_modified": (
            getattr(model, "last_modified", None).isoformat()
            if getattr(model, "last_modified", None)
            else None
        ),
        "pipeline_tag": getattr(model, "pipeline_tag", None),
        "gated": bool(gated),
        "private": bool(getattr(model, "private", False)),
        "disabled": bool(getattr(model, "disabled", False)),
        "format": "safetensors" if safetensors else "other-or-unknown",
        "params_b": round(params_b, 2) if params_b else None,
        # These are decimal GB because the measured reference uses decimal
        # checkpoint sizes. Keeping the unit explicit prevents the UI from
        # presenting a rough estimate as exact filesystem capacity.
        "estimated_source_gb": round(estimated_source, 1) if estimated_source else None,
        "estimated_store_gb": round(estimated_store, 1) if estimated_store else None,
        "availability": "remote",
        "action": "authenticate" if gated else "get",
        **compatibility,
    }


def search_catalog(
    *,
    query: str,
    cursor: str | None,
    page_size: int,
    sort: str,
    task: str | None,
    parameter_range: str | None,
    page: int | None = None,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    size = max(1, min(page_size, 50))
    requested_page = max(1, min(int(page or 1), 100))
    offset = _offset(cursor) if cursor else (requested_page - 1) * size
    api = HfApi()
    iterator = api.list_models(
        search=query or None,
        pipeline_tag=task or None,
        num_parameters=parameter_range or None,
        sort=_SORTS.get(sort, "downloads"),
        expand=[
            "config", "disabled", "downloads", "gated", "lastModified",
            "likes", "pipeline_tag", "private", "safetensors", "sha", "tags",
        ],
    )
    rows = list(islice(iterator, offset, offset + size + 1))
    has_more = len(rows) > size
    rows = rows[:size]
    return {
        "models": [_model_row(model) for model in rows],
        "cursor": cursor,
        "next_cursor": _cursor(offset + size) if has_more else None,
        "previous_cursor": _cursor(max(0, offset - size)) if offset else None,
        "page": offset // size + 1,
        "page_window": list(range(
            max(1, offset // size - 1),
            offset // size + (3 if has_more else 2),
        )),
        "page_size": size,
        "sort": sort if sort in _SORTS else "downloads",
        "exhausted": not has_more,
    }
