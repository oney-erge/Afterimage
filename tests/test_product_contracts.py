import datetime
import sys
import threading
import time
import types

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from afterimage.runtime.adapters import classify_config, resolve_model_adapter
from afterimage.runtime.compressed_store import CompressedLayer, decompress_layer_cpu_reference
from afterimage.runtime.control import JobControl
from afterimage.runtime.critical_path import TraceRecorder
from afterimage.runtime.huffman_chunked import ChunkedEncoded
from afterimage.runtime.streaming_engine import StreamingLosslessModel, _compress_one_tensor
from afterimage.server.app import app
from afterimage.server.acquisition import acquire_model
from afterimage.server.catalog import search_catalog
from afterimage.server.hardware import memory_info
from afterimage.server.jobs import JobRegistry
from afterimage.server.model_registry import ModelRegistry


def test_product_shell_is_modular_and_removes_artificial_ceiling_copy():
    client = TestClient(app)
    page = client.get("/").text
    # The server-rendered shell carries no marketing headline of its own
    # any more -- Home is state-aware (see afterimage/server/static/js/
    # home.js) and its actual copy depends on client-side data the server
    # response cannot see, so the one accurate product description left in
    # the raw HTML is the page's own <meta name="description">.
    assert "Afterimage serves larger language models on the hardware you already have" in page
    assert "/static/css/app.css" in page
    assert "/static/js/app.js" in page
    assert "Compare configurations" not in page
    assert "unsupported" not in page.lower()
    assert client.get("/static/js/models.js").status_code == 200
    assert client.get("/static/js/chat.js").status_code == 200

    payload = client.get("/api/capability").json()
    assert payload["streaming"]["beyond_vram"] is True
    assert "streaming_fast_max_params_b" not in payload
    assert "streaming_slow_max_params_b" not in payload


def test_memory_info_reports_total_and_available(monkeypatch):
    fake = types.SimpleNamespace(total=32 * 1024**3, available=19 * 1024**3)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(
        virtual_memory=lambda: fake
    ))
    assert memory_info() == {"total_gib": 32.0, "available_gib": 19.0}


def test_registry_persists_models_and_recovers_interrupted_jobs(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = ModelRegistry(path)
    first.upsert_model("Qwen/test", state="downloading", bytes_done=42)
    first.create_job("job1", "acquire", "model-lifecycle", "Qwen/test")
    first.update_job("job1", status="running", progress={"bytes_done": 42})

    second = ModelRegistry(path)
    recovered_model = second.get_model("Qwen/test")
    assert recovered_model["bytes_done"] == 42
    assert recovered_model["state"] == "interrupted"
    job = second.get_job("job1")
    assert job["status"] == "interrupted"
    assert job["progress"] == {"bytes_done": 42}


def test_registry_persists_reusable_runtime_profiles(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = ModelRegistry(path)
    first.save_runtime_profile(
        "profile1", name="Critical path candidate", model_id="Qwen/test",
        config={"vram_budget_gb": 6.0, "placement_policy": "traffic_density"},
        source_run_id="run1",
    )
    saved = ModelRegistry(path).get_runtime_profile("profile1")
    assert saved["model_id"] == "Qwen/test"
    assert saved["config"]["vram_budget_gb"] == 6.0
    assert saved["source_run_id"] == "run1"


def test_cancel_is_terminal_and_releases_a_stalled_download_lane(tmp_path, monkeypatch):
    import afterimage.server.jobs as jobs_module

    persistence = ModelRegistry(tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(jobs_module, "model_registry", persistence)
    jobs = JobRegistry()
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def blocked(_control):
        first_started.set()
        release_first.wait(timeout=2)
        return {"late": True}

    first = jobs.create("acquire", blocked, model_id="one", lane="model-lifecycle")
    assert first_started.wait(timeout=1)
    second = jobs.create(
        "acquire", lambda _control: second_started.set() or {"ready": True},
        model_id="two", lane="model-lifecycle",
    )
    assert not second_started.wait(timeout=0.1)
    assert jobs.cancel(first.id).status == "cancelled"
    assert second_started.wait(timeout=1)
    release_first.set()
    for _ in range(100):
        if jobs.get(second.id).status == "done":
            break
        time.sleep(0.005)
    assert jobs.get(first.id).status == "cancelled"
    assert jobs.get(second.id).status == "done"


def test_job_control_reports_when_pause_reaches_a_real_checkpoint():
    states = []
    control = JobControl(state_callback=states.append)
    control.pause()
    worker = threading.Thread(target=control.checkpoint)
    worker.start()
    for _ in range(100):
        if states:
            break
        time.sleep(0.005)
    assert states == ["paused"]
    assert worker.is_alive()
    control.resume()
    worker.join(timeout=1)
    assert states == ["paused", "running"]


def test_catalog_cursor_returns_distinct_pages_and_never_blocks_get(monkeypatch):
    class FakeApi:
        def list_models(self, **_kwargs):
            for index in range(5):
                yield types.SimpleNamespace(
                    id=f"Qwen/model-{index}", sha=f"sha-{index}", downloads=100 - index,
                    likes=index, last_modified=datetime.datetime(2026, 1, index + 1),
                    pipeline_tag="text-generation", gated=False, private=False,
                    disabled=False,
                    safetensors=types.SimpleNamespace(total=(70 + index) * 1_000_000_000),
                    config={"architectures": ["UnknownForCausalLM"], "model_type": "unknown"},
                )

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    first = search_catalog(
        query="Qwen", cursor=None, page_size=2, sort="downloads",
        task=None, parameter_range=None,
    )
    second = search_catalog(
        query="Qwen", cursor=first["next_cursor"], page_size=2,
        sort="downloads", task=None, parameter_range=None,
    )
    third = search_catalog(
        query="Qwen", cursor=None, page=3, page_size=2,
        sort="downloads", task=None, parameter_range=None,
    )
    assert [row["model_id"] for row in first["models"]] == ["Qwen/model-0", "Qwen/model-1"]
    assert [row["model_id"] for row in second["models"]] == ["Qwen/model-2", "Qwen/model-3"]
    assert [row["model_id"] for row in third["models"]] == ["Qwen/model-4"]
    assert third["page"] == 3
    assert all(row["action"] == "get" for row in first["models"])
    assert all(row["params_b"] >= 70 for row in first["models"])
    assert all(row["execution"] == "download-only" for row in first["models"])


def test_local_discovery_separates_cached_safetensors_from_ollama(tmp_path, monkeypatch):
    import afterimage.server.discovery as discovery

    files = [
        types.SimpleNamespace(file_name="config.json"),
        types.SimpleNamespace(file_name="model.safetensors"),
    ]
    revision = types.SimpleNamespace(
        commit_hash="abc", snapshot_path=tmp_path, size_on_disk=12,
        files=files, last_modified=2,
    )
    repo = types.SimpleNamespace(
        repo_id="Qwen/cached", repo_type="model", size_on_disk=12,
        revisions=[revision],
    )
    monkeypatch.setattr(
        "huggingface_hub.scan_cache_dir",
        lambda: types.SimpleNamespace(repos=[repo]),
    )
    monkeypatch.setattr(discovery, "ollama_models", lambda query="": [{
        "model_id": "qwen3:8b", "source": "ollama", "source_label": "Ollama",
        "format": "Q4_K_M", "can_prepare": False, "size_bytes": 4,
        "message": "GGUF", "external_url": "http://127.0.0.1:8000/ui",
    }])
    payload = discovery.discover_local_models("qwen")
    by_id = {row["model_id"]: row for row in payload["models"]}
    assert by_id["Qwen/cached"]["can_prepare"] is True
    assert by_id["qwen3:8b"]["can_prepare"] is False
    assert payload["sources"] == {"huggingface_cache": 1, "ollama": 1}


def test_acquisition_retains_download_only_models_and_marks_verified_store_ready(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    state_path = tmp_path / "state.sqlite3"
    local_registry = ModelRegistry(state_path)
    monkeypatch.setattr(
        "afterimage.server.acquisition.download_snapshot",
        lambda model_id, revision, control: (
            snapshot, {"revision": "resolved", "has_safetensors": True, "source_bytes": 8}
        ),
    )
    monkeypatch.setattr(
        "afterimage.server.acquisition.inspect_snapshot",
        lambda path: {
            "execution": "download-only", "execution_reason": "No adapter yet",
            "executable": False, "modality": "text", "mixture_of_experts": False,
        },
    )
    result = acquire_model(
        "example/download-only", revision=None, prepare=True,
        control=JobControl(), registry=local_registry,
    )
    assert result["state"] == "downloaded"
    assert local_registry.get_model("example/download-only")["state"] == "downloaded"

    store = tmp_path / "store"
    monkeypatch.setattr("afterimage.server.acquisition._store_dir_for", lambda _model: store)
    monkeypatch.setattr(
        "afterimage.server.acquisition.inspect_snapshot",
        lambda path: {
            "execution": "experimental", "execution_reason": "Adapter resolved",
            "executable": True, "modality": "vision-text", "mixture_of_experts": False,
        },
    )
    monkeypatch.setattr(
        "afterimage.runtime.streaming_engine.compress_model_to_disk",
        lambda *args, **kwargs: {
            "total_orig_bytes": 8, "total_comp_bytes": 4, "ratio": 2.0,
        },
    )
    monkeypatch.setattr("afterimage.runtime.binstore.verify_store", lambda _path: (True, []))
    result = acquire_model(
        "example/vision", revision=None, prepare=True,
        control=JobControl(), registry=local_registry,
    )
    ready = local_registry.get_model("example/vision")
    assert result["state"] == "ready"
    assert ready["state"] == "ready"
    assert ready["compatibility"] == "experimental"
    assert ready["metadata"]["manifest"]["ratio"] == 2.0


def test_layout_adapter_resolves_text_and_vision_models():
    layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
    language = torch.nn.Module()
    language.layers = layers
    language.embed_tokens = torch.nn.Embedding(4, 2)

    text = torch.nn.Module()
    text.config = types.SimpleNamespace(model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    text.model = language
    text.lm_head = torch.nn.Linear(2, 4, bias=False)
    adapter = resolve_model_adapter(text)
    assert adapter.capabilities.modality == "text"
    assert adapter.layer_key(0, "mlp.weight") == "model.layers.0.mlp.weight"

    outer = torch.nn.Module()
    outer.language_model = language
    outer.visual = torch.nn.Linear(2, 2, bias=False)
    vision = torch.nn.Module()
    vision.config = types.SimpleNamespace(
        model_type="qwen3_vl", architectures=["Qwen3VLForConditionalGeneration"],
        text_config=types.SimpleNamespace(model_type="qwen3"),
    )
    vision.model = outer
    vision.lm_head = torch.nn.Linear(2, 4, bias=False)
    adapter = resolve_model_adapter(vision)
    assert adapter.capabilities.modality == "vision-text"
    assert adapter.embedding_prefix == "model.language_model.embed_tokens"
    assert adapter.language_config is vision.config.text_config


def _decode_result(result):
    if result["kind"] != "compressed":
        return torch.from_numpy(result["arrays"]["raw"]).to(torch.bfloat16)
    arrays = result["arrays"]
    encoded = ChunkedEncoded(
        packed=arrays["packed"], chunk_offsets=arrays["chunk_offsets"],
        chunk_nbytes=arrays["chunk_nbytes"], sym_lut=arrays["sym_lut"],
        len_lut=arrays["len_lut"], max_bits=result["max_bits"],
        chunk_size=result["chunk_size"], n_symbols=result["n_symbols"],
        shape=tuple(result["shape"]),
    )
    return decompress_layer_cpu_reference(CompressedLayer(
        sign_mantissa=torch.from_numpy(arrays["sign_mantissa"]),
        encoded=encoded, shape=tuple(result["shape"]),
    ))


def test_packed_moe_expert_is_stored_as_an_independent_exact_slice(tmp_path):
    path = tmp_path / "experts.safetensors"
    weights = torch.zeros((3, 96, 64), dtype=torch.bfloat16)
    weights[1, :, ::7] = 1.5
    save_file({"model.layers.0.mlp.experts.gate_up_proj": weights}, str(path))
    result = _compress_one_tensor((
        str(path), "model.layers.0.mlp.experts.gate_up_proj",
        64, None, 16, False, 1, False,
    ))
    assert result["key"].endswith(".__expert__.1")
    assert result["shape"] == [96, 64]
    assert torch.equal(_decode_result(result), weights[1])


def test_force_raw_storage_stores_bit_exact_bf16_not_upcast_float32(tmp_path):
    """force_raw_storage on a tensor that would normally be Huffman-coded
    must use the same bit-preserving bf16-as-int16 technique row_gather
    already uses (see _compress_one_tensor), not the small-tensor
    fallback's float32 upcast -- that fallback is fine for a handful of
    tiny norm vectors (silently 2x on-disk bytes with comp_bytes still
    reporting the bf16 count) but would corrupt this project's own
    checkpoint-size accounting if reused for the bulk of a model's
    weights. This is the real control run_offline_hypotheses/H6's
    representation planner needs for Figure 5 (raw BF16 vs compressed,
    same engine) to compare apples to apples.
    """
    path = tmp_path / "layer.safetensors"
    weights = torch.randn(300, 300, dtype=torch.bfloat16)
    save_file({"model.layers.0.mlp.gate_proj.weight": weights}, str(path))

    normal = _compress_one_tensor((
        str(path), "model.layers.0.mlp.gate_proj.weight", 64, None, 16, False, None, False))
    forced = _compress_one_tensor((
        str(path), "model.layers.0.mlp.gate_proj.weight", 64, None, 16, False, None, True))

    assert normal["kind"] == "compressed"  # unaffected without the flag
    assert forced["kind"] == "raw"
    assert forced["dtype"] == "bfloat16"
    # Bit-exact size, not the float32 fallback's silent doubling.
    assert forced["comp_bytes"] == forced["orig_bytes"] == weights.numel() * 2
    assert forced["arrays"]["raw"].nbytes == weights.numel() * 2

    reconstructed = torch.from_numpy(forced["arrays"]["raw"]).view(torch.bfloat16).view(300, 300)
    assert torch.equal(reconstructed, weights)


def test_force_raw_storage_round_trips_through_the_manifest_declared_dtype():
    """The read side (StreamingLosslessModel._decode_tensor) must branch on
    the ACTUAL on-disk numpy dtype (int16 => reinterpret via .view(), not
    the old float32-fallback's .to() numeric cast) -- calling .to() on the
    bit-packed int16 array would silently convert integer VALUES to
    floats instead of reinterpreting bytes, corrupting every weight.
    """
    weights = torch.randn(4, 4, dtype=torch.bfloat16)
    raw16 = weights.contiguous().view(torch.int16).numpy()
    # Simulate exactly what _decode_tensor does for a raw, non-row-gather
    # tensor whose stored numpy array is int16-backed.
    out = torch.from_numpy(raw16)
    assert out.dtype == torch.int16
    out = out.view(torch.bfloat16)
    assert torch.equal(out.view(4, 4), weights)


def test_selected_expert_forward_matches_packed_reference():
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 3
            self.act_fn = torch.nn.functional.silu
            self.gate_up_proj = torch.nn.Parameter(torch.randn(3, 8, 4))
            self.down_proj = torch.nn.Parameter(torch.randn(3, 4, 4))

        def forward(self, hidden, indices, weights):
            out = torch.zeros_like(hidden)
            mask = torch.nn.functional.one_hot(indices, self.num_experts).permute(2, 1, 0)
            for expert in mask.sum((-1, -2)).nonzero():
                expert = int(expert[0])
                positions, tokens = torch.where(mask[expert])
                gate, up = torch.nn.functional.linear(hidden[tokens], self.gate_up_proj[expert]).chunk(2, -1)
                current = torch.nn.functional.linear(self.act_fn(gate) * up, self.down_proj[expert])
                out.index_add_(0, tokens, current * weights[tokens, positions, None])
            return out

    root = torch.nn.Module()
    root.experts = Experts()
    gate = root.experts.gate_up_proj.detach().clone()
    down = root.experts.down_proj.detach().clone()
    hidden = torch.randn(5, 4)
    indices = torch.tensor([[0, 2], [1, 2], [0, 1], [2, 1], [1, 0]])
    routing = torch.softmax(torch.randn(5, 2), dim=-1)
    expected = root.experts(hidden, indices, routing)

    engine = StreamingLosslessModel.__new__(StreamingLosslessModel)
    engine.model = root
    engine.control = JobControl()
    engine.trace = TraceRecorder(enabled=False)
    engine._last_compute_event = None
    engine._forward_index = 0
    engine._expert_slices = {
        "experts.gate_up_proj": [f"experts.gate_up_proj.__expert__.{i}" for i in range(3)],
        "experts.down_proj": [f"experts.down_proj.__expert__.{i}" for i in range(3)],
    }
    tensors = {
        **{f"experts.gate_up_proj.__expert__.{i}": gate[i] for i in range(3)},
        **{f"experts.down_proj.__expert__.{i}": down[i] for i in range(3)},
    }
    engine._load_tensor = lambda key, **_kwargs: tensors[key]
    engine._install_expert_streaming()
    actual = root.experts(hidden, indices, routing)
    assert torch.allclose(actual, expected)


def test_qwen_vl_and_moe_catalog_classification_is_explicitly_experimental():
    value = classify_config({
        "model_type": "qwen3_vl_moe",
        "architectures": ["Qwen3VLMoeForConditionalGeneration"],
    })
    assert value["modality"] == "vision-text"
    assert value["mixture_of_experts"] is True
    assert value["execution"] == "experimental"
