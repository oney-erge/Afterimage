"""FastAPI server: an OpenAI-compatible chat endpoint plus native job
control (compress/plan/pause/resume/cancel) and a minimal static web UI,
all driving the same StreamingLosslessModel engine used everywhere else in
this project -- the server is a thin control layer, not a second
implementation of anything.

Run via `afterimage serve` (afterimage/cli.py) or directly:
    uvicorn afterimage.server.app:app --port 8420
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from afterimage.cli import DEFAULT_STORE_ROOT, _detect_gpu, _detect_ram_gb, _store_dir_for
from afterimage.runtime.config import EngineConfig
from afterimage.server.jobs import registry

app = FastAPI(title="Afterimage", description="Lossless streaming inference control API")

_STATIC_DIR = pathlib.Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


# -- hardware / models -------------------------------------------------

@app.get("/api/hardware")
def hardware() -> dict:
    import torch
    gpu = _detect_gpu()
    total = free = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        free, total = free / 1e9, total / 1e9
    return {"gpu": gpu, "ram_gb": _detect_ram_gb(), "cuda_available": torch.cuda.is_available(),
            "vram_free_gb": free, "vram_total_gb": total}


@app.get("/api/models")
def list_models() -> dict:
    out = []
    if DEFAULT_STORE_ROOT.exists():
        for p in sorted(DEFAULT_STORE_ROOT.iterdir()):
            man_path = p / "manifest.json"
            if man_path.exists():
                man = json.loads(man_path.read_text())
                out.append({"model_id": man.get("model_id", p.name), "store": str(p),
                           "orig_gb": man["total_orig_bytes"] / 1e9,
                           "comp_gb": man["total_comp_bytes"] / 1e9, "ratio": man["ratio"]})
    return {"models": out}


# -- compression job ------------------------------------------------------

class CompressRequest(BaseModel):
    model_id: str
    chunk_size: int = 1024
    quantize: str | None = None


@app.post("/api/compress")
def compress(req: CompressRequest) -> dict:
    from afterimage.runtime.streaming_engine import compress_model_to_disk

    out_dir = _store_dir_for(req.model_id)
    cfg = EngineConfig(chunk_size=req.chunk_size, quantize=req.quantize)

    def work(control):
        return compress_model_to_disk(req.model_id, out_dir, config=cfg, control=control)

    job = registry.create("compress", work)
    return {"job_id": job.id}


class PlanRequest(BaseModel):
    model_id: str
    vram_budget_gb: float
    ram_budget_gb: float = 0.0


@app.post("/api/plan")
def plan(req: PlanRequest) -> dict:
    from afterimage.runtime.vram_planner import plan_from_manifest

    man_path = _store_dir_for(req.model_id) / "manifest.json"
    if not man_path.exists():
        raise HTTPException(404, "no compressed store for %r -- POST /api/compress first" % req.model_id)
    man = json.loads(man_path.read_text())
    p = plan_from_manifest(man, vram_budget_gb=req.vram_budget_gb, ram_budget_gb=req.ram_budget_gb)
    return {"feasible": p.feasible, "reason": p.reason, "vram_gb": p.vram_gb, "ram_gb": p.ram_gb,
            "disk_gb_per_token": p.disk_gb_per_token, "vram_tensors": len(p.vram_keys),
            "ram_tensors": len(p.ram_keys), "disk_tensors": len(p.disk_keys)}


# -- job control ------------------------------------------------------

def _get_job(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job: %r" % job_id)
    return job


@app.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": [{"id": j.id, "kind": j.kind, "status": j.status} for j in registry.list()]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _get_job(job_id)
    return {"id": job.id, "kind": job.kind, "status": job.status,
            "progress": job.progress, "error": job.error}


@app.post("/api/jobs/{job_id}/pause")
def job_pause(job_id: str) -> dict:
    _get_job(job_id).control.pause()
    return {"status": "paused"}


@app.post("/api/jobs/{job_id}/resume")
def job_resume(job_id: str) -> dict:
    _get_job(job_id).control.resume()
    return {"status": "resumed"}


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    _get_job(job_id).control.cancel()
    return {"status": "cancelling"}


@app.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    """Pushes progress whenever it changes, polling the job at 2 Hz -- the
    job itself is updated synchronously from a background thread (see
    jobs.py), so this just watches for changes rather than blocking on them.
    """
    await websocket.accept()
    job = registry.get(job_id)
    if job is None:
        await websocket.close(code=4004)
        return
    last = None
    try:
        while True:
            snapshot = {"status": job.status, "progress": job.progress, "error": job.error}
            if snapshot != last:
                await websocket.send_json(snapshot)
                last = snapshot
            if job.status in ("done", "error", "cancelled"):
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


# -- inference: engine cache -------------------------------------------

class _EngineCache:
    """Holds at most one loaded StreamingLosslessModel: this project
    targets one consumer GPU, which cannot hold two large models resident
    at once anyway, so a single-slot cache (evict-and-reload on model
    change) is the right shape rather than an LRU of many."""

    def __init__(self):
        self._lock = threading.Lock()
        self._key = None
        self._sm = None
        self._tok = None

    def get(self, model_id: str, cfg: EngineConfig):
        key = (model_id, cfg.vram_cap_gb, cfg.vram_budget_gb, cfg.ram_budget_gb, cfg.io_prefetch_depth)
        with self._lock:
            if self._key != key:
                if self._sm is not None:
                    self._sm.close()
                    self._sm = None
                from transformers import AutoTokenizer
                from afterimage.runtime.streaming_engine import StreamingLosslessModel

                store_dir = _store_dir_for(model_id)
                if not (store_dir / "manifest.json").exists():
                    raise HTTPException(
                        404, "no compressed store for %r -- POST /api/compress first" % model_id)
                self._tok = AutoTokenizer.from_pretrained(model_id)
                self._sm = StreamingLosslessModel(model_id, store_dir, device="cuda", config=cfg)
                self._key = key
            return self._sm, self._tok


_engine_cache = _EngineCache()


# -- OpenAI-compatible chat completions ------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False
    vram_cap_gb: float | None = None
    vram_budget_gb: float | None = None
    ram_budget_gb: float | None = None


@app.get("/v1/models")
def openai_models() -> dict:
    models = list_models()["models"]
    return {"object": "list",
            "data": [{"id": m["model_id"], "object": "model", "owned_by": "afterimage"}
                     for m in models]}


def _build_prompt(tok, messages: list[ChatMessage]) -> str:
    return tok.apply_chat_template([m.model_dump() for m in messages],
                                   tokenize=False, add_generation_prompt=True)


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    cfg = EngineConfig(vram_cap_gb=req.vram_cap_gb, vram_budget_gb=req.vram_budget_gb,
                       ram_budget_gb=req.ram_budget_gb, progress=False)
    sm, tok = _engine_cache.get(req.model, cfg)
    prompt = _build_prompt(tok, req.messages)
    ids = tok(prompt, return_tensors="pt").input_ids.to(sm.device)
    stop_ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()

    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    if req.stream:
        # Token-by-token SSE needs each chunk emitted as it's produced, not
        # collected after the fact -- _stream_chat bridges generate_greedy's
        # synchronous on_token callback (running in a background thread) to
        # this generator via a queue.
        return StreamingResponse(_stream_chat(sm, tok, ids, req, cid, created, stop_ids),
                                 media_type="text/event-stream")

    import torch
    with torch.no_grad():
        seq = sm.generate_greedy(ids, max_new_tokens=req.max_tokens, stop_token_ids=stop_ids)
    text = tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True)
    return {"id": cid, "object": "chat.completion", "created": created, "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ids.shape[1],
                     "completion_tokens": seq.shape[1] - ids.shape[1],
                     "total_tokens": seq.shape[1]}}


def _stream_chat(sm, tok, ids, req: ChatCompletionRequest, cid: str, created: int, stop_ids: set):
    """Runs generation in a background thread, pushing each token's decoded
    text piece into a queue that this generator drains -- the standard
    bridge for turning a synchronous, blocking token loop into an SSE
    stream without blocking the async event loop on GPU work."""
    import queue
    import threading

    import torch

    q: queue.Queue = queue.Queue()
    SENTINEL = object()

    def on_token(tok_id: int) -> None:
        piece = tok.decode([tok_id], skip_special_tokens=True)
        q.put(piece)

    def run():
        try:
            with torch.no_grad():
                sm.generate_greedy(ids, max_new_tokens=req.max_tokens,
                                   on_token=on_token, stop_token_ids=stop_ids)
        finally:
            q.put(SENTINEL)

    threading.Thread(target=run, daemon=True).start()

    def chunk(delta: dict, finish_reason=None) -> str:
        payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                  "model": req.model,
                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
        return "data: " + json.dumps(payload) + "\n\n"

    yield chunk({"role": "assistant", "content": ""})
    while True:
        piece = q.get()
        if piece is SENTINEL:
            break
        yield chunk({"content": piece})
    yield chunk({}, finish_reason="stop")
    yield "data: [DONE]\n\n"
