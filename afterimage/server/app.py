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
import dataclasses
import json
import logging
import pathlib
import statistics
import threading
import time
import uuid

import numpy as np

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from afterimage.cli import DEFAULT_STORE_ROOT, _detect_gpu, _detect_ram_gb, _store_dir_for
from afterimage.experiments import (
    HYPOTHESES, PROFILES, ExperimentRun, ResultStore, oracle_gap,
    environment_manifest, registry_payload, run_paired,
)
from afterimage.reference import MEASURED_REFERENCE
from afterimage.runtime.config import EngineConfig
from afterimage.server.jobs import registry

logger = logging.getLogger(__name__)

app = FastAPI(title="Afterimage", description="Lossless streaming inference control API")

_STATIC_DIR = pathlib.Path(__file__).parent / "static"
_EXPERIMENT_RESULTS = ResultStore(DEFAULT_STORE_ROOT / "_experiment_results")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


# -- operability: health, version, last-run stats -----------------------

@app.get("/health")
def health() -> dict:
    """Cheap and CUDA-safe: never touches the engine cache lock, so it stays
    responsive while a request is mid-generation -- exactly the case a
    container orchestrator's health probe needs to distinguish from a
    genuinely hung process."""
    import torch
    model_loaded = _engine_cache._sm is not None
    return {"status": "ok", "model_loaded": model_loaded,
            "loaded_model": _engine_cache._key[0] if _engine_cache._key else None,
            "cuda_available": torch.cuda.is_available()}


@app.get("/api/version")
def version() -> dict:
    from afterimage import __version__
    return {"version": __version__}


@app.get("/api/stats")
def last_stats() -> dict:
    """The counters from the most recently completed /v1/chat/completions
    call on the currently loaded engine -- StreamStats.reset() runs at the
    start of each request, so this is that request's numbers, not a
    lifetime total."""
    if _engine_cache._sm is None:
        raise HTTPException(404, "no model loaded yet")
    sm = _engine_cache._sm
    completion_len = _engine_cache._last_completion_len
    if completion_len is None:
        raise HTTPException(404, "no completed generation on the loaded model yet")
    return _stats_usage(sm, 0, completion_len)["afterimage"]


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


# MEASURED_REFERENCE (imported above): Qwen3-14B on the RTX 3080 Laptop
# (README's benchmark table / docs/ALL_HYPOTHESES_AND_BASELINES.md).
# Extrapolating to other sizes assumes the same architecture family and
# roughly linear scaling of store size and streamed-read time with
# parameter count -- true to first order for same-precision dense
# transformers, not a promise for any specific checkpoint. Every number
# this produces is an ESTIMATE; only a real compress + run on the actual
# model is a measurement.


def _capability_estimate(params_b: float) -> dict:
    ref = MEASURED_REFERENCE
    return {
        "params_b": round(params_b, 2),
        "bf16_gb": round(params_b * ref["bf16_gb_per_b_params"], 1),
        "compressed_store_gb": round(params_b * ref["compressed_gb_per_b_params"], 1),
        "min_memory_s_per_token": round(params_b * ref["min_memory_s_per_token_per_b"], 1),
        "fast_s_per_token": round(params_b * ref["fast_s_per_token_per_b"], 1),
    }


@app.get("/api/capability")
def capability() -> dict:
    """What this GPU can actually do, in plain terms -- the "what this means
    for you" card on the Home screen is built entirely from this response.
    Every number here is a rough extrapolation from one measured checkpoint,
    not a benchmark; the response says so explicitly so the UI never has to
    invent that caveat itself.

    streaming_fast_max_params_b can come out LARGER than
    streaming_slow_max_params_b -- that is not a bug. They answer different
    questions ("biggest model that stays fast under speculation" vs.
    "biggest model minimum-memory streaming can still limp through") and are
    not nested: nothing here promises the fast profile's acceptance rate
    (measured only at the 14B/0.6B draft pair) holds at other scales.
    """
    import torch
    ref = MEASURED_REFERENCE
    gpu = _detect_gpu()
    vram_gb = gpu.get("vram_gb")
    ram_gb = _detect_ram_gb()

    native_fit_max_params_b = None
    streaming_fast_max_params_b = None
    streaming_slow_max_params_b = None
    if vram_gb:
        # Whole bf16 model resident, no streaming -- roughly what you could
        # do WITHOUT Afterimage. ~15% headroom reserved for KV cache and
        # activations, itself a rough rule of thumb, not a measurement.
        native_fit_max_params_b = round(
            (vram_gb * 0.85) / ref["bf16_gb_per_b_params"], 1)
        # "Fast" (speculative) needs roughly the measured fixed VRAM floor
        # (residency + draft model) to first order, regardless of size; if
        # the card clears that floor, speed then scales ~linearly with
        # params. 15s/token is a chosen "still feels interactive" ceiling.
        # Extrapolated from the one measured 14B/0.6B draft pair -- larger
        # targets may see a lower speculative acceptance rate than that
        # pair did, which this simple scaling does not model.
        if vram_gb >= ref["fast_vram_floor_gb"]:
            streaming_fast_max_params_b = round(
                15.0 / ref["fast_s_per_token_per_b"], 1)
        # Minimum-memory streaming has essentially no VRAM floor to speak
        # of, so size stops being the limit at all -- only speed does.
        # 45s/token is a chosen "still usable, but slow" ceiling.
        streaming_slow_max_params_b = round(
            45.0 / ref["min_memory_s_per_token_per_b"], 1)

    return {
        "vram_gb": vram_gb, "ram_gb": ram_gb,
        "cuda_available": torch.cuda.is_available(),
        "measured_reference_model": ref["model"],
        "native_fit_max_params_b": native_fit_max_params_b,
        "streaming_fast_max_params_b": streaming_fast_max_params_b,
        "streaming_slow_max_params_b": streaming_slow_max_params_b,
        "estimates": [_capability_estimate(p) for p in (4, 7, 14, 32, 70)],
    }


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


# This engine's hard-coded Llama-family layout (see streaming_engine.py's
# construction-time architecture check) -- kept in sync manually rather than
# imported, since HF's search API returns architecture strings, not a
# loaded config we could introspect the same way the engine does.
_SUPPORTED_ARCHITECTURES = (
    "LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM",
    "MistralForCausalLM",
)


@app.get("/api/models/search")
def search_models(q: str = "", limit: int = 20) -> dict:
    """Search the HuggingFace Hub for bf16 safetensors checkpoints and
    classify each one's fit against this machine's detected VRAM, using the
    same estimate /api/capability is built from. Best-effort: a network or
    API failure returns an empty list with an explanation rather than a
    500 -- this must never block someone who already knows the model id
    they want and is using /api/compress directly."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        results = list(api.list_models(
            search=q or None, filter="safetensors", sort="downloads",
            limit=max(1, min(limit, 50)),
            expand=["safetensors", "config", "downloads"]))
    except Exception as exc:
        return {"models": [], "error": "%s: %s" % (type(exc).__name__, exc)}

    cap = capability()
    vram_gb = cap["vram_gb"]
    ref = MEASURED_REFERENCE
    already = {m["model_id"] for m in list_models()["models"]}

    out = []
    for m in results:
        model_id = m.id
        params_b = None
        safetensors = getattr(m, "safetensors", None)
        if safetensors and getattr(safetensors, "total", None):
            params_b = safetensors.total / 1e9
        architectures = list(getattr(m, "config", {}).get("architectures", [])
                            if getattr(m, "config", None) else [])
        supported = (not architectures
                    or any(a in _SUPPORTED_ARCHITECTURES for a in architectures))

        row = {
            "model_id": model_id, "downloads": getattr(m, "downloads", None),
            "params_b": round(params_b, 2) if params_b else None,
            "architectures": architectures,
            "supported_architecture": supported,
            "already_compressed": model_id in already,
            "fit": "unknown",
        }
        if not supported:
            row["fit"] = "unsupported"
        elif params_b:
            estimate = _capability_estimate(params_b)
            row.update(estimate)
            if vram_gb is None:
                row["fit"] = "unknown"
            elif params_b <= (vram_gb * 0.85) / ref["bf16_gb_per_b_params"]:
                row["fit"] = "native"
            elif estimate["fast_s_per_token"] <= 15.0 and vram_gb >= ref["fast_vram_floor_gb"]:
                row["fit"] = "streams_fast"
            elif estimate["min_memory_s_per_token"] <= 45.0:
                row["fit"] = "streams_slow"
            else:
                row["fit"] = "streams_very_slow"
        out.append(row)
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


# -- profile comparison ---------------------------------------------------

class CompareRequest(BaseModel):
    model_id: str
    prompt: str = "The capital of France is"
    max_new_tokens: int = 12


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict:
    """Runs the same prompt under all three named profiles
    (min-memory/balanced/fast) sequentially on this machine and reports
    each one's real measured numbers. This is what turns "3.15x" from a
    number in a README into something a user watched happen on their own
    GPU. Runs as a job (like compression) since a 14B-class model can take
    several minutes across three profiles -- the caller polls/watches it
    exactly like a compress job."""
    from afterimage.cli import RUN_PROFILES

    store_dir = _store_dir_for(req.model_id)
    if not (store_dir / "manifest.json").exists():
        raise HTTPException(
            404, "no compressed store for %r -- POST /api/compress first" % req.model_id)

    def work(control):
        import torch
        from transformers import AutoTokenizer
        from afterimage.runtime.streaming_engine import (
            StreamingLosslessModel, load_draft_model,
        )

        tok = AutoTokenizer.from_pretrained(req.model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ids = tok(req.prompt, return_tensors="pt").input_ids.to(device)

        rows = []
        profile_names = list(RUN_PROFILES.keys())
        for i, name in enumerate(profile_names):
            control.checkpoint()
            control.report(stage=name, profile_index=i, total_profiles=len(profile_names))
            preset = RUN_PROFILES[name]
            cfg = EngineConfig(vram_budget_gb=preset["vram_budget_gb"], progress=False,
                               draft_mode=("model" if preset["draft_model"] else "none"))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with StreamingLosslessModel(req.model_id, store_dir, device=device,
                                        config=cfg, control=control) as sm:
                if preset["draft_model"]:
                    draft = load_draft_model(preset["draft_model"], device=device)
                    seq, _policy = sm.generate_adaptive(
                        ids, max_new_tokens=req.max_new_tokens, draft_model=draft,
                        temperature=0.0)
                else:
                    seq = sm.generate_greedy(ids, max_new_tokens=req.max_new_tokens)
                wall_s = time.perf_counter() - t0
                n_tokens = seq.shape[1] - ids.shape[1]
                text = tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True)
                rows.append({
                    "profile": name,
                    "vram_budget_gb": preset["vram_budget_gb"],
                    "draft_model": preset["draft_model"],
                    "text": text,
                    "tokens": n_tokens,
                    "wall_seconds": wall_s,
                    "seconds_per_token": wall_s / max(n_tokens, 1),
                    "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9
                                    if torch.cuda.is_available() else None),
                })
        baseline = next((r for r in rows if r["profile"] == "min-memory"), None)
        for row in rows:
            if baseline and row["seconds_per_token"] > 0:
                row["speedup_vs_min_memory"] = (
                    baseline["seconds_per_token"] / row["seconds_per_token"])
        return {"prompt": req.prompt, "rows": rows}

    job = registry.create("compare", work)
    return {"job_id": job.id}


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
            "progress": job.progress, "error": job.error,
            "result": job.result if job.status == "done" else None}


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


# -- hypothesis experiments -----------------------------------------------

@app.get("/api/experiments")
def experiments() -> dict:
    return registry_payload()


@app.get("/api/experiments/{hypothesis_id}")
def experiment_definition(hypothesis_id: str) -> dict:
    hypothesis = HYPOTHESES.get(hypothesis_id)
    if hypothesis is None:
        raise HTTPException(404, "no such hypothesis: %r" % hypothesis_id)
    from afterimage.protocols import protocol_for
    return {"hypothesis": dataclasses.asdict(hypothesis),
            "candidate": dataclasses.asdict(PROFILES[hypothesis.candidate_profile]),
            "control": dataclasses.asdict(PROFILES[hypothesis.control_profile]),
            "protocol": dataclasses.asdict(protocol_for(hypothesis_id))}


class ExperimentRunRequest(BaseModel):
    model_id: str | None = None
    prompt: str = "The capital of France is"
    max_new_tokens: int = 8
    repeats: int = 3
    seed: int = 0
    draft_model_id: str | None = None
    config_overrides: dict = Field(default_factory=dict)
    candidate_overrides: dict = Field(default_factory=dict)
    control_overrides: dict = Field(default_factory=dict)
    inputs: dict = Field(default_factory=dict)


def _resolved_experiment_config(hypothesis_id: str, req: ExperimentRunRequest,
                                profile) -> EngineConfig:
    hypothesis = HYPOTHESES[hypothesis_id]
    overrides = dict(req.config_overrides)
    overrides.update(req.candidate_overrides
                     if profile.id == hypothesis.candidate_profile
                     else req.control_overrides)
    overrides.update(profile.overrides)
    if (hypothesis_id in ("h2-hazard-cost", "h11-neural-utility-spec")
            and profile.id == hypothesis.control_profile):
        overrides.pop("spec_policy_state", None)
        overrides.pop("spec_policy_learn", None)
    return EngineConfig.from_dict(overrides)


def _specialized_experiment(hypothesis_id: str, req: ExperimentRunRequest,
                            control) -> ExperimentRun:
    """Artifact-driven experiments that do not execute a generation pair."""
    from afterimage.runtime.controllers import LinearProfileBandit
    from afterimage.runtime.representations import RepresentationOption, plan_representations
    from afterimage.runtime.xor_reference import audit_reference_candidates

    hypothesis = HYPOTHESES[hypothesis_id]
    run = ExperimentRun(uuid.uuid4().hex[:16], hypothesis_id, "running", time.time(),
                        metadata={"inputs": sorted(req.inputs), "seed": req.seed,
                                  "environment": environment_manifest(
                                      pathlib.Path(__file__).parents[2])})
    control.checkpoint()
    if hypothesis.runner == "oracle_gap":
        result = oracle_gap(req.inputs["result_dataset"])
        run.summary = result
        run.verdict = "favored" if result["joint_uplift"] >= hypothesis.minimum_effect else "falsified"
    elif hypothesis.runner == "profile_bandit":
        rows = req.inputs["result_dataset"]
        calibration = req.inputs["calibration_dataset"]
        profiles = sorted(rows[0]["rewards"])
        dim = len(rows[0]["context"])
        learner = LinearProfileBandit(profiles, dim, algorithm="linucb",
                                      baseline_profile=profiles[0], seed=req.seed)
        # Calibration has full feedback and is disjoint from chronological
        # evaluation. Loading one observation per profile avoids charging
        # cold exploration to the deployed controller or leaking test rows.
        for row in calibration:
            if set(row["rewards"]) != set(profiles):
                raise ValueError("calibration/evaluation profiles differ")
            for profile in profiles:
                learner.update(profile, row["context"], float(row["rewards"][profile]))
        chosen, oracle_total, baseline_total = 0.0, 0.0, 0.0
        for i, row in enumerate(rows):
            arm = learner.choose(row["context"])
            reward = float(row["rewards"][arm])
            learner.update(arm, row["context"], reward)
            chosen += reward
            oracle_total += max(float(v) for v in row["rewards"].values())
            baseline_total += float(row["rewards"][profiles[0]])
            control.report(phase="profile_bandit", completed=i + 1, total=len(rows))
        fraction = chosen / max(oracle_total, 1e-12)
        run.summary = {"oracle_fraction": fraction, "chosen_reward": chosen,
                       "oracle_reward": oracle_total, "baseline_reward": baseline_total}
        run.verdict = "favored" if fraction >= hypothesis.minimum_effect else "falsified"
    elif hypothesis.runner == "representation_plan":
        options = [RepresentationOption(**row) for row in req.inputs["representation_options"]]
        plan = plan_representations(
            options, vram_budget_bytes=int(req.inputs.get("vram_budget_bytes", 0)),
            ram_budget_bytes=int(req.inputs.get("ram_budget_bytes", 0)),
            storage_budget_bytes=req.inputs.get("storage_budget_bytes"),
            quantum_bytes=int(req.inputs.get("quantum_bytes", 16 << 20)))
        run.summary = plan.to_dict()
        control_s = float(req.inputs["uniform_prepare_s"])
        gain = 1.0 - plan.predicted_prepare_s / max(control_s, 1e-12)
        run.summary["gain_over_uniform"] = gain
        run.verdict = ("favored" if plan.feasible and gain >= hypothesis.minimum_effect
                       else "falsified" if plan.feasible else "invalid")
    elif hypothesis.runner == "xor_audit":
        tensors = {}
        # ``load_file`` eagerly materializes every tensor in a shard.  Real
        # MoE shards are several gigabytes even when an audit needs only a
        # handful of experts, so loading once per requested expert made H7
        # practically unrunnable.  ``safe_open`` memory-maps each unique
        # shard and materializes only the named tensors.
        from safetensors import safe_open
        by_path = {}
        for item in req.inputs["expert_tensors"]:
            by_path.setdefault(item["path"], []).append(item)
        for path, items in by_path.items():
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for item in items:
                    if item["tensor_key"] not in available:
                        raise ValueError(
                            "tensor %s is missing from %s" %
                            (item["tensor_key"], path))
                    tensors[item["id"]] = handle.get_tensor(item["tensor_key"])
        bases = set(req.inputs["reference_bases"])
        audit = audit_reference_candidates(tensors, base_keys=bases)
        independent = req.inputs["independent_compressed_bytes"]
        missing_sizes = set(tensors) - set(independent)
        if missing_sizes:
            raise ValueError("independent_compressed_bytes missing %s" % sorted(missing_sizes))
        independent_total = sum(float(independent[key]) for key in tensors)
        candidate_total = 0.0
        reference_total = 0.0
        for key in tensors:
            if key in bases or audit[key] is None:
                candidate_total += float(independent[key])
                reference_total += float(independent[key])
            else:
                candidate_total += min(float(independent[key]),
                                       float(audit[key]["compressed_bytes"]))
                reference_total += float(audit[key]["compressed_bytes"])
        total_gain = 1.0 - candidate_total / max(independent_total, 1.0)
        run.summary = {"audit": audit, "reference_bases": sorted(bases),
                       "independent_bytes": independent_total,
                       "candidate_bytes": candidate_total,
                       "reference_only_bytes": reference_total,
                       "reference_only_reduction": (
                           1.0 - reference_total / max(independent_total, 1.0)),
                       "total_storage_reduction": total_gain}
        run.verdict = "favored" if total_gain >= hypothesis.minimum_effect else "falsified"
    elif hypothesis.runner == "trace_simulator":
        rows = req.inputs["trace_dataset"]
        actual, predicted = [], []
        chosen_reward = baseline_reward = oracle_reward = 0.0
        policy_rows = all("actual_rewards" in row and "predicted_rewards" in row
                          for row in rows)
        for row in rows:
            if policy_rows:
                profiles = sorted(row["actual_rewards"])
                if set(profiles) != set(row["predicted_rewards"]):
                    raise ValueError("actual_rewards and predicted_rewards need identical profiles")
                chosen = max(profiles, key=lambda key: float(row["predicted_rewards"][key]))
                baseline = row.get("baseline_profile", profiles[0])
                chosen_reward += float(row["actual_rewards"][chosen])
                baseline_reward += float(row["actual_rewards"][baseline])
                oracle_reward += max(float(value) for value in row["actual_rewards"].values())
                actual.extend(float(row["actual_rewards"][key]) for key in profiles)
                predicted.extend(float(row["predicted_rewards"][key]) for key in profiles)
            else:
                actual.append(float(row["actual_s"]))
                predicted.append(float(row["predicted_s"]))
        mape = statistics.mean(abs(a - p) / max(abs(a), 1e-12)
                               for a, p in zip(actual, predicted))
        if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0:
            actual_rank = np.argsort(np.argsort(actual))
            predicted_rank = np.argsort(np.argsort(predicted))
            corr = float(np.corrcoef(actual_rank, predicted_rank)[0, 1])
        else:
            corr = 1.0
        run.summary = {"mape": mape, "rank_correlation": corr}
        calibrated = mape <= 0.10 and corr >= 0.90
        if policy_rows:
            improvement = chosen_reward / max(baseline_reward, 1e-12) - 1.0
            run.summary.update(
                chosen_reward=chosen_reward, baseline_reward=baseline_reward,
                oracle_reward=oracle_reward,
                oracle_fraction=chosen_reward / max(oracle_reward, 1e-12),
                improvement_over_baseline=improvement)
            run.verdict = ("favored" if calibrated and
                           improvement >= hypothesis.minimum_effect else "falsified")
        else:
            # Calibration-only data proves the simulator prerequisite, not
            # that acting on it beats the contextual control.
            run.verdict = "inconclusive" if calibrated else "falsified"
    else:
        raise ValueError("unsupported specialized runner %r" % hypothesis.runner)
    run.status = "done"
    run.completed_at = time.time()
    return run


@app.post("/api/experiments/{hypothesis_id}/runs")
def start_experiment(hypothesis_id: str, req: ExperimentRunRequest) -> dict:
    hypothesis = HYPOTHESES.get(hypothesis_id)
    if hypothesis is None:
        raise HTTPException(404, "no such hypothesis: %r" % hypothesis_id)
    if req.repeats < 1:
        raise HTTPException(400, "repeats must be >= 1")
    missing = []
    for field in hypothesis.required_inputs:
        if field == "draft_model_id":
            present = bool(req.draft_model_id)
        elif field in ("critical_path_profile", "spec_policy_state",
                       "replay_plan_state"):
            present = bool(req.config_overrides.get(field))
        else:
            present = field in req.inputs
        if not present:
            missing.append(field)
    if missing:
        raise HTTPException(400, "experiment requires inputs: %s" % ", ".join(missing))
    if hypothesis_id in ("h2-hazard-cost", "h11-neural-utility-spec"):
        state_path = pathlib.Path(req.config_overrides["spec_policy_state"])
        if not state_path.exists():
            raise HTTPException(
                400, "%s spec_policy_state does not exist: %s" %
                (hypothesis_id, state_path))
        if req.config_overrides.get("spec_policy_learn") is not False:
            raise HTTPException(
                400, "%s held-out evaluation requires config_overrides."
                "spec_policy_learn=false; calibrate the state on separate prompts first"
                % hypothesis_id)
    if hypothesis_id in ("h1-critical-path", "h16-spec-critical-path"):
        profile_path = pathlib.Path(req.config_overrides["critical_path_profile"])
        if not profile_path.exists():
            raise HTTPException(
                400, "%s critical_path_profile does not exist: %s" %
                (hypothesis_id, profile_path))
        if req.config_overrides.get("trace_events"):
            raise HTTPException(
                400, ("%s evaluation must run with tracing off; collect the profile "
                      "in separate control runs because CUDA trace synchronization "
                      "is intrusive") % hypothesis_id)
    if hypothesis_id in (
            "h10-replay-cem", "h13-qubo-residency",
            "h15-extent-qubo-residency"):
        plan_path = pathlib.Path(req.config_overrides["replay_plan_state"])
        profile_path = pathlib.Path(req.config_overrides["critical_path_profile"])
        if not plan_path.exists() or not profile_path.exists():
            raise HTTPException(
                400, "%s requires existing replay_plan_state and "
                "critical_path_profile files" % hypothesis_id)
        # The Placement family's shared L1 prerequisite (docs/RESEARCH_METHODS.md
        # section 5, "Evidence levels") is that the frozen plan
        # actually differs from its control -- a search that returns its own
        # seed cannot support or falsify anything about the search method,
        # only about the seed. H13 and H15 previously both failed exactly
        # this way (repair() silently reconstructing the control; see
        # docs/HYPOTHESIS_LINEAGE.md), so the gate is enforced here for all
        # three plan-search hypotheses, not only H15.
        from afterimage.runtime.replay_planner import ReplayResidencyPlan
        plan = ReplayResidencyPlan.load(plan_path)
        if not plan.report.treatment_diverged:
            raise HTTPException(
                400, "%s mechanism gate requires a frozen plan that differs "
                "from its control (report.treatment_diverged); a search that "
                "returns its seed provides no evidence about the search "
                "method" % hypothesis_id)
        if (hypothesis_id == "h15-extent-qubo-residency"
                and plan.report.optimized_over_control < 0.02):
            raise HTTPException(
                400, "H15 mechanism gate additionally requires at least 2% "
                "predicted replay gain over control")
    if hypothesis.runner == "generation" and not req.model_id:
        raise HTTPException(400, "generation experiments require model_id")
    if hypothesis.runner == "generation" and req.repeats < hypothesis.minimum_repeats:
        raise HTTPException(
            400, "%s requires at least %d repeats for its paired test" %
            (hypothesis_id, hypothesis.minimum_repeats))
    if (hypothesis.runner == "generation"
            and req.max_new_tokens < hypothesis.minimum_new_tokens):
        raise HTTPException(
            400, "%s requires at least %d generated tokens per trial" %
            (hypothesis_id, hypothesis.minimum_new_tokens))
    if hypothesis.runner == "generation":
        if not (_store_dir_for(req.model_id) / "manifest.json").exists():
            raise HTTPException(404, "no compressed store for %r" % req.model_id)
        try:
            _resolved_experiment_config(
                hypothesis_id, req, PROFILES[hypothesis.candidate_profile])
            _resolved_experiment_config(
                hypothesis_id, req, PROFILES[hypothesis.control_profile])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def work(control):
        if hypothesis.runner != "generation":
            run = _specialized_experiment(hypothesis_id, req, control)
        else:
            import torch
            from transformers import AutoTokenizer
            from afterimage.runtime.streaming_engine import (
                StreamingLosslessModel, load_draft_model,
            )

            tokenizer = AutoTokenizer.from_pretrained(req.model_id)
            prompt_ids = tokenizer(req.prompt, return_tensors="pt").input_ids
            draft = (load_draft_model(req.draft_model_id, device="cuda")
                     if req.draft_model_id else None)

            def execute(profile, repeat):
                cfg = _resolved_experiment_config(hypothesis_id, req, profile)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                with StreamingLosslessModel(req.model_id, _store_dir_for(req.model_id),
                                            device="cuda", config=cfg,
                                            control=control) as engine:
                    ids = prompt_ids.to(engine.device)
                    engine.stats.reset()
                    start = time.perf_counter()
                    policy = None
                    if cfg.draft_mode == "model":
                        sequence, policy = engine.generate_adaptive(
                            ids, req.max_new_tokens, draft_model=draft,
                            temperature=1.0,
                            generator=torch.Generator(device=engine.device).manual_seed(
                                req.seed + repeat))
                    else:
                        sequence = engine.generate_greedy(ids, req.max_new_tokens)
                    wall = time.perf_counter() - start
                    generated = int(sequence.shape[1] - ids.shape[1])
                    head_rows = int(engine.model.lm_head.weight.shape[0])
                    mips_possible_rows = generated * head_rows
                    return {
                        "committed_tokens_per_second": generated / max(wall, 1e-12),
                        "wall_seconds": wall,
                        "generated_tokens": generated,
                        "bytes_read": engine.stats.bytes_read,
                        "peak_vram_bytes": (torch.cuda.max_memory_allocated()
                                            if torch.cuda.is_available() else 0),
                        "prefetch_hits": engine.stats.prefetch_hits,
                        "prefetch_misses": engine.stats.prefetch_misses,
                        "prefetch_wait_seconds": engine.stats.prefetch_wait_seconds,
                        "pageable_ram_fallback_keys": sorted(
                            engine._ram_cache_pageable_keys),
                        "policy_state": (policy.state_dict()
                                         if policy is not None else None),
                        "mips_certified": engine.stats.mips_certified,
                        "mips_fallbacks": engine.stats.mips_fallbacks,
                        "mips_rows_evaluated": engine.stats.mips_rows_evaluated,
                        "mips_rows_pruned": engine.stats.mips_rows_pruned,
                        "mips_index_build_seconds": engine.stats.mips_index_build_seconds,
                        "mips_index_bytes": engine.mips_index_bytes,
                        "mips_certificate_rate": (
                            engine.stats.mips_certified / max(generated, 1)
                            if cfg.lm_head_policy == "certified_mips" else None),
                        "mips_pruned_fraction": (
                            engine.stats.mips_rows_pruned /
                            max(mips_possible_rows, 1)
                            if cfg.lm_head_policy == "certified_mips" else None),
                        "output_token_ids": sequence[0, ids.shape[1]:].tolist(),
                        "exact": cfg.exactness_contract != "approximate",
                        "config": cfg.to_dict(), "config_fingerprint": cfg.fingerprint(),
                    }

            run = run_paired(hypothesis_id, execute, repeats=req.repeats,
                             seed=req.seed,
                             metadata={"model_id": req.model_id, "prompt": req.prompt,
                                       "max_new_tokens": req.max_new_tokens,
                                       "seed": req.seed,
                                       "environment": environment_manifest(
                                           pathlib.Path(__file__).parents[2])},
                             progress=lambda update: control.report(**update))
        path = _EXPERIMENT_RESULTS.write_once(run)
        return {"run_id": run.id, "result_path": str(path), "run": run.to_dict()}

    job = registry.create("experiment:" + hypothesis_id, work)
    return {"job_id": job.id, "hypothesis_id": hypothesis_id}


@app.get("/api/experiment-runs/{run_id}")
def experiment_result(run_id: str) -> dict:
    result = _EXPERIMENT_RESULTS.get(run_id)
    if result is None:
        raise HTTPException(404, "no such completed experiment run: %r" % run_id)
    return result


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
        self._draft_key = None
        self._draft = None
        self._last_completion_len: int | None = None

    def get(self, model_id: str, cfg: EngineConfig):
        # The full config fingerprint, not a hand-picked field subset -- an
        # earlier version keyed on only 5 fields, so a request that changed
        # e.g. draft_mode or lm_head_slice_rows but matched on those 5
        # silently reused the previous request's engine with the previous
        # request's settings instead of reloading.
        key = (model_id, cfg.fingerprint())
        with self._lock:
            if self._key != key:
                if self._sm is not None:
                    logger.info("evicting engine for %s to load %s", self._key[0], model_id)
                    self._sm.close()
                    self._sm = None
                from transformers import AutoTokenizer
                from afterimage.runtime.streaming_engine import StreamingLosslessModel

                store_dir = _store_dir_for(model_id)
                if not (store_dir / "manifest.json").exists():
                    raise HTTPException(
                        404, "no compressed store for %r -- POST /api/compress first" % model_id)
                logger.info("loading %s (config %s)", model_id, cfg.fingerprint())
                self._tok = AutoTokenizer.from_pretrained(model_id)
                self._sm = StreamingLosslessModel(model_id, store_dir, device="cuda", config=cfg)
                self._key = key
                logger.info("%s loaded", model_id)
            return self._sm, self._tok

    def get_draft(self, draft_model_id: str, device: str):
        """A small resident draft model, cached across requests the same
        way -- reloading a fresh copy every chat request would defeat the
        whole point of it being small and fast."""
        key = (draft_model_id, device)
        with self._lock:
            if self._draft_key != key:
                from afterimage.runtime.streaming_engine import load_draft_model
                self._draft = load_draft_model(draft_model_id, device=device)
                self._draft_key = key
            return self._draft


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
    # Not OpenAI-standard fields: Afterimage's dial. draft_model enables
    # speculative decoding (the 3.15x measured win, see README) -- omit it
    # for plain greedy decoding. lm_head_slice_rows > 0 trades bit-exactness
    # for a lower VRAM floor (see EngineConfig.lm_head_slice_rows).
    draft_model: str | None = None
    spec_k: int = 8
    spec_target_cache: bool = False
    lm_head_slice_rows: int = 0


@app.get("/v1/models")
def openai_models() -> dict:
    models = list_models()["models"]
    return {"object": "list",
            "data": [{"id": m["model_id"], "object": "model", "owned_by": "afterimage"}
                     for m in models]}


def _build_prompt(tok, messages: list[ChatMessage]) -> str:
    return tok.apply_chat_template([m.model_dump() for m in messages],
                                   tokenize=False, add_generation_prompt=True)


def _stats_usage(sm, ids_len: int, completion_len: int) -> dict:
    usage = {"prompt_tokens": ids_len, "completion_tokens": completion_len,
             "total_tokens": ids_len + completion_len}
    afterimage_stats = {
        "seconds_per_token": (
            (sm.stats.io_seconds + sm.stats.decode_seconds + sm.stats.compute_seconds)
            / max(completion_len, 1)),
        "io_seconds": sm.stats.io_seconds,
        "decode_seconds": sm.stats.decode_seconds,
        "compute_seconds": sm.stats.compute_seconds,
        "bytes_read_gb": sm.stats.bytes_read / 1e9,
        "prefetch_hit_rate": (sm.stats.prefetch_hits /
                              max(sm.stats.prefetch_hits + sm.stats.prefetch_misses, 1)),
        "spec_sweeps": sm.stats.spec_sweeps,
        "spec_accepted_tokens": sm.stats.spec_accepted_tokens,
        "spec_cache_crops": sm.stats.spec_cache_crops,
        "spec_cached_prefix_tokens": sm.stats.spec_cached_prefix_tokens,
    }
    import torch
    if torch.cuda.is_available():
        afterimage_stats["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    usage["afterimage"] = afterimage_stats
    return usage


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    logger.info("chat completion: model=%s max_tokens=%d stream=%s draft=%s",
               req.model, req.max_tokens, req.stream, req.draft_model or "none")
    cfg = EngineConfig(vram_cap_gb=req.vram_cap_gb, vram_budget_gb=req.vram_budget_gb,
                       ram_budget_gb=req.ram_budget_gb, progress=False,
                       draft_mode=("model" if req.draft_model else "none"),
                       spec_k=req.spec_k,
                       spec_target_cache=req.spec_target_cache,
                       lm_head_slice_rows=req.lm_head_slice_rows)
    sm, tok = _engine_cache.get(req.model, cfg)
    draft = _engine_cache.get_draft(req.draft_model, sm.device) if req.draft_model else None
    prompt = _build_prompt(tok, req.messages)
    ids = tok(prompt, return_tensors="pt").input_ids.to(sm.device)
    stop_ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()

    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    if req.stream:
        # Token-by-token SSE needs each chunk emitted as it's produced, not
        # collected after the fact -- _stream_chat bridges generate_greedy's
        # (or generate_adaptive's) synchronous on_token callback (running in
        # a background thread) to this generator via a queue.
        return StreamingResponse(
            _stream_chat(sm, tok, ids, req, cid, created, stop_ids, draft),
            media_type="text/event-stream")

    import torch
    sm.stats.reset()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        if draft is not None:
            seq, _policy = sm.generate_adaptive(
                ids, max_new_tokens=req.max_tokens, draft_model=draft,
                temperature=req.temperature, stop_token_ids=stop_ids)
        else:
            seq = sm.generate_greedy(ids, max_new_tokens=req.max_tokens, stop_token_ids=stop_ids)
    text = tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True)
    completion_len = seq.shape[1] - ids.shape[1]
    _engine_cache._last_completion_len = completion_len
    return {"id": cid, "object": "chat.completion", "created": created, "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop"}],
            "usage": _stats_usage(sm, ids.shape[1], completion_len)}


def _stream_chat(sm, tok, ids, req: ChatCompletionRequest, cid: str, created: int,
                 stop_ids: set, draft=None):
    """Runs generation in a background thread, pushing each token's decoded
    text piece into a queue that this generator drains -- the standard
    bridge for turning a synchronous, blocking token loop into an SSE
    stream without blocking the async event loop on GPU work. Works
    identically for plain greedy and speculative decoding: generate_adaptive
    calls on_token once per accepted token, same contract as generate_greedy.

    At 9-33 s/token, the gap between tokens is long enough that "nothing
    happened yet" is indistinguishable from "it's stuck" if all we ever
    send is token deltas. The queue.get(timeout=...) below turns each silent
    gap into a periodic "progress" event carrying the live StreamStats
    snapshot (bytes read, I/O/decode seconds) instead -- no new engine
    hooks needed, since those counters already update incrementally during
    generation."""
    import queue
    import threading

    import torch

    q: queue.Queue = queue.Queue()
    TOKEN, DONE = "token", "done"
    gen_t0 = time.perf_counter()

    def on_token(tok_id: int) -> None:
        piece = tok.decode([tok_id], skip_special_tokens=True)
        q.put((TOKEN, piece))

    def run():
        try:
            sm.stats.reset()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                if draft is not None:
                    sm.generate_adaptive(ids, max_new_tokens=req.max_tokens,
                                         draft_model=draft, temperature=req.temperature,
                                         on_token=on_token, stop_token_ids=stop_ids)
                else:
                    sm.generate_greedy(ids, max_new_tokens=req.max_tokens,
                                       on_token=on_token, stop_token_ids=stop_ids)
        finally:
            q.put((DONE, None))

    threading.Thread(target=run, daemon=True).start()

    def chunk(delta: dict, finish_reason=None) -> str:
        payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                  "model": req.model,
                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
        return "data: " + json.dumps(payload) + "\n\n"

    def progress_event() -> str:
        payload = {
            "id": cid, "object": "chat.completion.chunk.progress",
            "progress": {
                "elapsed_seconds": round(time.perf_counter() - gen_t0, 1),
                "bytes_read_gb": round(sm.stats.bytes_read / 1e9, 3),
                "io_seconds": round(sm.stats.io_seconds, 1),
                "decode_seconds": round(sm.stats.decode_seconds, 1),
                "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                                if torch.cuda.is_available() else None),
            },
        }
        return "data: " + json.dumps(payload) + "\n\n"

    yield chunk({"role": "assistant", "content": ""})
    n_tokens = 0
    while True:
        try:
            kind, value = q.get(timeout=1.0)
        except queue.Empty:
            yield progress_event()
            continue
        if kind == DONE:
            break
        n_tokens += 1
        yield chunk({"content": value})
    yield chunk({}, finish_reason="stop")
    _engine_cache._last_completion_len = n_tokens
    usage = _stats_usage(sm, ids.shape[1], n_tokens)
    yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk.usage",
                                 "usage": usage}) + "\n\n"
    yield "data: [DONE]\n\n"
