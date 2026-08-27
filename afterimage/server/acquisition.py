"""Idempotent model acquisition and preparation pipeline."""
from __future__ import annotations

import pathlib
import shutil
import time
from typing import Any

from afterimage.cli import DEFAULT_STORE_ROOT, _store_dir_for
from afterimage.runtime.adapters import classify_config, resolve_model_adapter
from afterimage.server.model_registry import ModelRegistry, model_registry


def _selected_files(info: Any) -> list[Any]:
    files = []
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if name.startswith(".git/") or (
            pathlib.PurePosixPath(name).name.startswith("consolidated")
            and name.endswith(".safetensors")
        ):
            continue
        files.append(sibling)
    return files


def _file_size(value: Any) -> int:
    return int(getattr(value, "size", None) or 0)


def _disk_preflight(source_bytes: int, store_root: pathlib.Path) -> None:
    store_root.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store_root.parent).free
    # A conservative first pass: source cache, final compressed store, and
    # temporary compression headroom.  The dry-run endpoint can refine this
    # after the config and tensor index are available.
    required = int(source_bytes * 1.9) + 2 * 1024**3
    if source_bytes and free < required:
        raise OSError(
            "insufficient disk space: acquisition needs about %.1f GiB free, "
            "but %.1f GiB is available"
            % (required / 1024**3, free / 1024**3)
        )


def inspect_snapshot(snapshot: pathlib.Path) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    import torch

    try:
        config = AutoConfig.from_pretrained(snapshot, trust_remote_code=False)
        classified = classify_config(config)
        model_class = (
            AutoModelForImageTextToText
            if classified["modality"] == "vision-text"
            else AutoModelForCausalLM
        )
        with torch.device("meta"):
            model = model_class.from_config(config, dtype=torch.bfloat16)
        adapter = resolve_model_adapter(model)
        classified.update(
            executable=True,
            adapter_layout=adapter.capabilities.layout,
            mixture_of_experts=adapter.capabilities.mixture_of_experts,
        )
    except Exception as exc:  # noqa: BLE001 - inspection failure preserves the download
        classified = locals().get("classified", {
            "architectures": [], "model_type": None, "modality": "unknown",
            "mixture_of_experts": False,
        })
        classified.update(executable=False, adapter_layout=None)
        classified["execution"] = "download-only"
        classified["execution_reason"] = str(exc)
    return classified


class _DownloadProgress:
    def __init__(self, *, done: int, total: int, control, filename: str):
        self.done = done
        self.total = total
        self.control = control
        self.filename = filename
        self.started = time.monotonic()
        self.current = 0

    def tqdm_class(self):
        owner = self
        from tqdm.auto import tqdm

        class ReportingTqdm(tqdm):
            def update(self, n=1):
                # Hugging Face calls tqdm.update while it receives chunks.
                # Checking here makes pause/cancel responsive within a large
                # shard instead of waiting until the entire file completes.
                owner.control.checkpoint()
                result = super().update(n)
                owner.current = int(self.n)
                elapsed = max(time.monotonic() - owner.started, 0.001)
                completed = owner.done + owner.current
                owner.control.report(
                    stage="downloading", file=owner.filename,
                    bytes_done=completed, bytes_total=owner.total,
                    progress=completed / max(owner.total, 1),
                    bytes_per_second=owner.current / elapsed,
                    message="Downloading %s" % owner.filename,
                )
                return result

        return ReportingTqdm


def download_snapshot(model_id: str, revision: str | None, control) -> tuple[pathlib.Path, dict]:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    info = api.model_info(
        model_id, revision=revision, files_metadata=True
    )
    resolved_revision = info.sha
    files = _selected_files(info)
    total = sum(_file_size(value) for value in files)
    _disk_preflight(total, DEFAULT_STORE_ROOT)
    model_registry.upsert_model(
        model_id, revision=resolved_revision, state="downloading",
        stage="downloading", bytes_total=total, error=None,
    )
    done = 0
    names: list[str] = []
    for index, sibling in enumerate(files):
        control.checkpoint()
        name = sibling.rfilename
        names.append(name)
        progress = _DownloadProgress(
            done=done, total=total, control=control, filename=name
        )
        from afterimage.runtime.control import JobCancelled

        for attempt in range(1, 4):
            try:
                hf_hub_download(
                    model_id, name, revision=resolved_revision,
                    tqdm_class=progress.tqdm_class(),
                )
                break
            except JobCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - bounded transient retry
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == 3 or status in {400, 401, 403, 404}:
                    raise
                control.report(
                    stage="downloading", file=name, retry=attempt,
                    message="Transfer interrupted. Retrying %s (%d/3)" % (name, attempt + 1),
                )
                time.sleep(2 ** (attempt - 1))
        done += _file_size(sibling)
        control.report(
            stage="downloading", file=name, file_index=index + 1,
            files_total=len(files), bytes_done=done, bytes_total=total,
            progress=done / max(total, 1), message="Downloaded %s" % name,
        )
        model_registry.upsert_model(
            model_id, bytes_done=done, bytes_total=total,
            state="downloading", stage="downloading",
        )
    snapshot = pathlib.Path(
        snapshot_download(
            model_id, revision=resolved_revision,
            allow_patterns=names, local_files_only=True,
        )
    )
    metadata = {
        "files": len(files), "source_bytes": total,
        "has_safetensors": any(name.endswith(".safetensors") for name in names),
    }
    return snapshot, {"revision": resolved_revision, **metadata}


def acquire_model(
    model_id: str,
    *,
    revision: str | None,
    prepare: bool,
    control,
    registry: ModelRegistry = model_registry,
) -> dict[str, Any]:
    """Download, inspect, prepare, and verify a model using durable states."""

    current = registry.get_model(model_id)
    snapshot = pathlib.Path(current["local_snapshot"]) if current and current.get(
        "local_snapshot"
    ) and pathlib.Path(current["local_snapshot"]).exists() else None
    metadata = dict(current.get("metadata", {})) if current else {}
    if snapshot is None:
        snapshot, downloaded = download_snapshot(model_id, revision, control)
        metadata.update(downloaded)
        registry.upsert_model(
            model_id, revision=downloaded["revision"], state="downloaded",
            stage="downloaded", local_snapshot=str(snapshot), metadata=metadata,
            error=None,
        )
    if not metadata.get("has_safetensors"):
        reason = "Downloaded successfully, but no safetensors checkpoint is available to prepare."
        registry.upsert_model(
            model_id, state="downloaded", stage="downloaded",
            compatibility="download-only", metadata=metadata, error=reason,
        )
        return {"model_id": model_id, "state": "downloaded", "message": reason}

    compatibility = inspect_snapshot(snapshot)
    metadata["compatibility"] = compatibility
    registry.upsert_model(
        model_id, compatibility=compatibility["execution"], metadata=metadata,
    )
    if not prepare or not compatibility["executable"]:
        state = "downloaded"
        registry.upsert_model(
            model_id, state=state, stage=state,
            error=None if compatibility["executable"] else compatibility["execution_reason"],
        )
        return {"model_id": model_id, "state": state, "compatibility": compatibility}

    from afterimage.runtime.binstore import verify_store
    from afterimage.runtime.streaming_engine import compress_model_to_disk

    store_dir = _store_dir_for(model_id)
    registry.upsert_model(model_id, state="preparing", stage="compressing", error=None)
    control.report(stage="preparing", progress=0.0, message="Preparing the lossless store")
    manifest = compress_model_to_disk(
        model_id, store_dir, source_dir=snapshot,
        revision=metadata.get("revision"), control=control,
    )
    registry.upsert_model(model_id, state="verifying", stage="verifying")
    control.report(stage="verifying", progress=0.98, message="Verifying the prepared store")
    valid, bad_keys = verify_store(store_dir)
    if not valid:
        raise RuntimeError("prepared store checksum mismatch: %s" % ", ".join(bad_keys[:5]))
    metadata["manifest"] = {
        "total_orig_bytes": manifest["total_orig_bytes"],
        "total_comp_bytes": manifest["total_comp_bytes"],
        "ratio": manifest["ratio"],
    }
    registry.upsert_model(
        model_id, state="ready", stage="ready", store_path=str(store_dir),
        compatibility=compatibility["execution"], metadata=metadata, error=None,
    )
    control.report(stage="ready", progress=1.0, message="Ready")
    return {
        "model_id": model_id, "state": "ready", "store": str(store_dir),
        "compatibility": compatibility,
    }
