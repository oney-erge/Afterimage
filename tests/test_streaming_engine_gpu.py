"""End-to-end GPU tests for StreamingLosslessModel against a tiny synthetic
Qwen3 checkpoint (random weights, no network, no real download): proves the
engine (raw binstore, row-gather, prefetch, and now three-tier VRAM/RAM/disk
residency) reproduces a plain in-memory forward pass bit-exactly, the same
standard this project has held every other piece of the engine to.

Requires CUDA + triton, like the rest of the GPU test suite; skips cleanly
otherwise.
"""
import pytest
import torch
from safetensors.torch import save_file

try:
    import triton  # noqa: F401
    _HAS = torch.cuda.is_available()
except ImportError:
    _HAS = False

pytestmark = pytest.mark.skipif(not _HAS, reason="needs CUDA + triton")


def _build_tiny_model(tie: bool, seed: int, intermediate_size: int = 64,
                      vocab_size: int = 97, hidden_size: int = 32):
    from transformers import Qwen3Config, AutoModelForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab_size, hidden_size=hidden_size,
                      intermediate_size=intermediate_size,
                      num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=64, tie_word_embeddings=tie)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    model.eval()
    return cfg, model


def _make_store(tmp_path, monkeypatch, model, cfg, model_id, tag):
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import compress_model_to_disk

    snap = tmp_path / ("snap_" + tag)
    snap.mkdir()
    # A real tied checkpoint never stores lm_head.weight separately -- it is
    # meant to alias embed_tokens.weight (see _materialize_resident's
    # re-tying logic). state_dict() returns both names pointing at the same
    # storage, which safetensors correctly refuses to save twice; dropping
    # the alias here reproduces what a real tied checkpoint on disk actually
    # looks like.
    sd = model.state_dict()
    if cfg.tie_word_embeddings:
        sd.pop("lm_head.weight", None)
    save_file(sd, str(snap / "model.safetensors"))

    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda mid, **kw: str(snap))
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained",
                        lambda mid, **kw: cfg)

    store_dir = tmp_path / ("store_" + tag)
    compress_model_to_disk(model_id, store_dir, config=EngineConfig(chunk_size=32),
                           max_workers=2)
    return store_dir


def test_untied_row_gather_and_prefetch_are_bit_exact_vs_reference(tmp_path, monkeypatch):
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=0)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/untied-engine", "untied")

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(42)
    ids = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    with torch.no_grad():
        ref_logits = ref_model_gpu(input_ids=ids).logits

    # monkeypatch must still be active: StreamingLosslessModel's own
    # __init__ calls AutoConfig.from_pretrained too.
    sm_pre = StreamingLosslessModel("fake/untied-engine", store_dir, device="cuda",
                                    config=EngineConfig(io_prefetch_depth=1))
    assert sm_pre.manifest["tensors"]["model.embed_tokens.weight"]["row_gather"] is True
    with torch.no_grad():
        logits_prefetch = sm_pre.forward_logits(ids)
    per_blob_calls = sm_pre.stats.storage_read_calls
    sm_pre.close()

    sm_extent = StreamingLosslessModel(
        "fake/untied-engine", store_dir, device="cuda",
        config=EngineConfig(io_prefetch_depth=1,
                            storage_read_policy="coalesced_extents",
                            storage_extent_max_bytes=1 << 20))
    with torch.no_grad():
        logits_extent = sm_extent.forward_logits(ids)
    extent_calls = sm_extent.stats.storage_read_calls
    sm_extent.close()

    sm_no = StreamingLosslessModel("fake/untied-engine", store_dir, device="cuda",
                                   config=EngineConfig(io_prefetch_depth=0))
    with torch.no_grad():
        logits_no_prefetch = sm_no.forward_logits(ids)
    sm_no.close()

    assert torch.equal(logits_prefetch, ref_logits), (
        "row-gathered + prefetched engine diverged from a plain reference "
        "forward pass on the identical weights")
    assert torch.equal(logits_no_prefetch, ref_logits), (
        "row-gathered engine (no prefetch) diverged from reference")
    assert torch.equal(logits_extent, ref_logits), (
        "coalesced storage extents changed exact logits")
    assert extent_calls < per_blob_calls
    assert torch.equal(logits_prefetch, logits_no_prefetch), (
        "prefetch changed the output -- it must only change WHEN bytes are "
        "read, never WHAT is read")


def test_tied_model_still_works_with_row_gather_storage_present(tmp_path, monkeypatch):
    """Regression check: a tied model must take the OLD full-materialization
    path for embed_tokens (row_gather absent from its manifest entry) and
    still reproduce reference logits exactly, now that row-gather exists as
    a code path it must correctly avoid."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=True, seed=1)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/tied-engine", "tied")

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    with torch.no_grad():
        ref_logits = ref_model_gpu(input_ids=ids).logits

    sm = StreamingLosslessModel("fake/tied-engine", store_dir, device="cuda",
                                config=EngineConfig(io_prefetch_depth=1))
    assert not sm.manifest["tensors"]["model.embed_tokens.weight"].get("row_gather")
    with torch.no_grad():
        logits = sm.forward_logits(ids)
    sm.close()

    assert torch.equal(logits, ref_logits)


def test_candidate_sweep_latency_reports_one_measurement_per_requested_count(
        tmp_path, monkeypatch):
    """H19 (Candidate-Amortization Hypothesis, afterimage/experiments.py):
    measure_candidate_sweep_latency must actually stream real weights
    through a real forward pass at each requested candidate count -- this
    is a real engine primitive, not a synthetic timer, so bytes_read and
    io_seconds/decode_seconds/compute_seconds should all be positive at
    every count, and results must come back in request order."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=3)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/candidate-sweep-engine", "sweep")

    torch.manual_seed(11)
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")

    sm = StreamingLosslessModel("fake/candidate-sweep-engine", store_dir,
                                device="cuda", config=EngineConfig(io_prefetch_depth=1))
    try:
        counts = [1, 2, 4]
        results = sm.measure_candidate_sweep_latency(ids, counts)
        assert [row["candidate_positions"] for row in results] == counts
        for row in results:
            assert row["verification_sweep_seconds"] > 0
            assert row["bytes_read"] > 0
    finally:
        sm.close()


def test_candidate_sweep_latency_rejects_bad_counts(tmp_path, monkeypatch):
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=4)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/candidate-sweep-invalid", "sweep-invalid")
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")

    sm = StreamingLosslessModel("fake/candidate-sweep-invalid", store_dir,
                                device="cuda", config=EngineConfig(io_prefetch_depth=1))
    try:
        with pytest.raises(ValueError, match="non-empty"):
            sm.measure_candidate_sweep_latency(ids, [])
        with pytest.raises(ValueError, match=">= 1"):
            sm.measure_candidate_sweep_latency(ids, [4, 0, 8])
    finally:
        sm.close()


def test_tied_embedding_streams_again_for_lm_head_at_low_vram(
        tmp_path, monkeypatch):
    """A disk-tier tied weight has two disjoint live ranges per forward."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(
        tie=True, seed=101, vocab_size=2048, hidden_size=64,
        intermediate_size=128)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/tied-streamed-engine", "tied_streamed")
    torch.manual_seed(102)
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    ref_model = ref_model.to("cuda")
    with torch.no_grad():
        reference = ref_model(input_ids=ids).logits
    del ref_model
    torch.cuda.empty_cache()

    # 134.62 MB leaves enough planner headroom for one materialized tensor
    # but essentially no permanent-residency capacity, forcing the tied
    # embedding/head matrix onto the disk tier in this small fixture.
    engine = StreamingLosslessModel(
        "fake/tied-streamed-engine", store_dir, device="cuda",
        config=EngineConfig(
            vram_budget_gb=0.13462, decode_slice_elems=32,
            io_prefetch_depth=0))
    assert engine._tier["model.embed_tokens.weight"] == "disk"
    with torch.no_grad():
        actual = engine.forward_logits(ids)
    assert engine.stats.layer_loads == cfg.num_hidden_layers
    engine.close()

    assert torch.equal(actual, reference)


def test_generate_greedy_matches_reference_argmax_sequence(tmp_path, monkeypatch):
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=2)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/greedy-engine", "greedy")

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device="cuda")

    def ref_generate_greedy(seq, n):
        with torch.no_grad():
            for _ in range(n):
                logits = ref_model_gpu(input_ids=seq).logits
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                seq = torch.cat([seq, nxt], dim=1)
        return seq

    expected = ref_generate_greedy(ids.clone(), 4)

    sm = StreamingLosslessModel("fake/greedy-engine", store_dir, device="cuda",
                                config=EngineConfig(io_prefetch_depth=1))
    with torch.no_grad():
        got = sm.generate_greedy(ids.clone(), max_new_tokens=4)
    sm.close()

    assert torch.equal(got, expected)


def test_context_manager_closes_readers(tmp_path, monkeypatch):
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=8)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/ctx-engine", "ctx")

    ids = torch.randint(0, cfg.vocab_size, (1, 3), device="cuda")
    with StreamingLosslessModel("fake/ctx-engine", store_dir, device="cuda",
                                config=EngineConfig(io_prefetch_depth=1)) as sm:
        with torch.no_grad():
            sm.forward_logits(ids)
        assert not sm._reader._fh.closed
    assert sm._reader._fh.closed
    for r in sm._prefetch_readers:
        assert r._fh.closed


def test_tiered_vram_ram_and_disk_residency_are_bit_exact(tmp_path, monkeypatch):
    """The actual product claim of lever A: assigning DECODER LAYER weights
    (not just embed/head/norms) independently to VRAM, RAM, or disk
    residency must still reproduce the exact same logits as an
    unconstrained in-memory reference model -- residency tier changes
    WHEN/HOW OFTEN a tensor's bytes are touched, never WHAT value it holds.

    The tiny fixture model here is far too small for the knapsack to
    naturally spill anything into the RAM tier (its default scratch
    headroom alone dwarfs the whole model), so this forces a plan that
    deliberately puts one tensor in each of the three tiers, via the same
    plan_from_manifest entry point _compute_tier_assignment actually calls
    -- this tests the ENGINE's execution of a plan, not the knapsack math
    itself (already covered in isolation by test_vram_planner.py).
    """
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime import vram_planner
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=11)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/tiered-engine", "tiered")

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(21)
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    with torch.no_grad():
        ref_logits = ref_model_gpu(input_ids=ids).logits

    manifest = __import__("json").loads((store_dir / "manifest.json").read_text())
    layer_keys = sorted(k for k in manifest["tensors"] if k.startswith("model.layers.0."))
    assert len(layer_keys) >= 3, "fixture needs at least 3 layer-0 tensors to force 3 tiers"
    forced_vram, forced_ram, forced_disk = layer_keys[0], layer_keys[1], layer_keys[2]
    other_layer_keys = [k for k in manifest["tensors"]
                        if k.startswith("model.layers.") and k not in
                        (forced_vram, forced_ram, forced_disk)]
    non_layer_keys = [k for k in manifest["tensors"]
                      if not k.startswith("model.layers.")
                      and not manifest["tensors"][k].get("row_gather")]

    def fake_plan(_manifest, vram_budget_gb, ram_budget_gb=0.0, **kw):
        return vram_planner.TierPlan(
            vram_budget_bytes=int(vram_budget_gb * 1e9),
            ram_budget_bytes=int(ram_budget_gb * 1e9),
            vram_headroom_bytes=0,
            vram_keys=[forced_vram] + non_layer_keys,
            ram_keys=[forced_ram],
            disk_keys=[forced_disk] + other_layer_keys,
            row_gather_keys=[k for k, m in manifest["tensors"].items()
                             if m.get("row_gather")],
            vram_bytes=0, ram_bytes=0, disk_bytes_per_token=0,
            feasible=True, reason="",
        )

    monkeypatch.setattr(vram_planner, "plan_from_manifest", fake_plan)

    ecfg = EngineConfig(vram_budget_gb=1.0, ram_budget_gb=1.0, io_prefetch_depth=1)
    sm = StreamingLosslessModel("fake/tiered-engine", store_dir, device="cuda", config=ecfg)

    assert sm._tier[forced_vram] == "vram"
    assert sm._tier[forced_ram] == "ram"
    assert sm._tier[forced_disk] == "disk"
    assert forced_ram in sm._ram_cache

    with torch.no_grad():
        logits = sm.forward_logits(ids)
        # run a second token too: RAM-tier tensors must be correctly
        # re-copied to GPU every call, not just materialized once and
        # forgotten -- a bug here would only show up on the SECOND load,
        # since the first load's real-tensor state could accidentally
        # survive from _materialize_resident's own initial placement.
        logits2 = sm.forward_logits(ids)
    sm.close()

    assert torch.equal(logits, ref_logits)
    assert torch.equal(logits2, ref_logits)


def test_deeper_prefetch_depth_is_bit_exact(tmp_path, monkeypatch):
    """Lever B: prefetching several layers ahead (not just one) must not
    change what is read, only when."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=13)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/deep-prefetch-engine", "deep_prefetch")

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(5)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device="cuda")
    with torch.no_grad():
        ref_logits = ref_model_gpu(input_ids=ids).logits

    sm = StreamingLosslessModel("fake/deep-prefetch-engine", store_dir, device="cuda",
                                config=EngineConfig(io_prefetch_depth=3))
    assert len(sm._prefetch_readers) == 3
    with torch.no_grad():
        logits = sm.forward_logits(ids)
    sm.close()

    assert torch.equal(logits, ref_logits)


def test_compute_seconds_does_not_double_count_io_and_decode(tmp_path, monkeypatch):
    """P1-7: compute_seconds must reflect GPU compute alone, not the whole
    forward-call wall time (which also contains the hook-triggered I/O and
    decode work, already tracked separately in io_seconds/decode_seconds).
    Before the fix, compute_seconds was measured as that raw wall time, so
    it was always roughly equal to (io_seconds + decode_seconds +
    compute_seconds) instead of being the small remainder."""
    import time as _time

    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=17)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/compute-timing-engine", "compute_timing")

    ids = torch.randint(0, cfg.vocab_size, (1, 4), device="cuda")
    sm = StreamingLosslessModel("fake/compute-timing-engine", store_dir, device="cuda",
                                config=EngineConfig(io_prefetch_depth=1))
    # Warm up first: the FIRST CUDA call in a process pays one-time context
    # init / kernel compilation cost that has nothing to do with this
    # engine's io/decode/compute split, and would otherwise swamp the tiny
    # amount of real work this tiny, mostly-uncompressed fixture model
    # does. That one-time cost legitimately belongs in "compute" by this
    # test's own definition, but it makes "compute should be small"
    # meaningless for a model this small -- exactly the mistake an earlier
    # version of this test made (it asserted compute < wall*0.9, which is
    # only true when io+decode dominate, which is a property of a REAL,
    # meaningfully-sized, meaningfully-compressed model -- verified
    # separately against the real 1.5B and 14B stores, where compute_s
    # measured 0.00s -- not a property this synthetic fixture has at all).
    with torch.no_grad():
        sm.forward_logits(ids)
    sm.stats.reset()

    t0 = _time.perf_counter()
    with torch.no_grad():
        sm.forward_logits(ids)
    torch.cuda.synchronize()
    wall = _time.perf_counter() - t0
    sm.close()

    # The actual invariant the fix guarantees: io + decode + compute must
    # PARTITION wall time (sum to it, within measurement slop), never
    # double-count it. Before the fix, compute_seconds alone was measured
    # as roughly the WHOLE wall time regardless of io/decode, so this sum
    # would have been roughly (wall + io + decode) instead of wall.
    total = sm.stats.io_seconds + sm.stats.decode_seconds + sm.stats.compute_seconds
    assert total <= wall * 1.15, (
        "io_seconds + decode_seconds + compute_seconds (%.4f) exceeds wall "
        "time (%.4f) by more than measurement slop -- compute_seconds is "
        "double-counting work already attributed to io/decode"
        % (total, wall))


def test_compressed_ram_tier_is_bit_exact_and_persists_across_tokens(tmp_path, monkeypatch):
    """the archived streaming proposal's own H1 (unrelated to the current H1
    critical-path-residency hypothesis): caching compressed bytes instead of
    a decoded tensor trades a memcpy for a real GPU decode every token --
    must still be bit-exact, and the cache must hold RAW bytes (not a
    decoded tensor) so the memory-saving claim is actually true."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime import vram_planner
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=61)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/ram-compressed-engine", "ram_compressed")

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(23)
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    with torch.no_grad():
        ref_logits = ref_model_gpu(input_ids=ids).logits

    manifest = __import__("json").loads((store_dir / "manifest.json").read_text())
    layer_keys = sorted(k for k in manifest["tensors"] if k.startswith("model.layers.0."))
    forced_ram = layer_keys[0]
    other_keys = [k for k in manifest["tensors"] if k != forced_ram
                 and not manifest["tensors"][k].get("row_gather")]
    row_gather_keys = [k for k, m in manifest["tensors"].items() if m.get("row_gather")]

    def fake_plan(_manifest, vram_budget_gb, ram_budget_gb=0.0, **kw):
        return vram_planner.TierPlan(
            vram_budget_bytes=int(vram_budget_gb * 1e9), ram_budget_bytes=int(ram_budget_gb * 1e9),
            vram_headroom_bytes=0, vram_keys=other_keys, ram_keys=[forced_ram], disk_keys=[],
            row_gather_keys=row_gather_keys, vram_bytes=0, ram_bytes=0, disk_bytes_per_token=0,
            feasible=True, reason="")

    monkeypatch.setattr(vram_planner, "plan_from_manifest", fake_plan)

    ecfg = EngineConfig(vram_budget_gb=1.0, ram_budget_gb=1.0, io_prefetch_depth=1,
                        ram_tier_format="compressed")
    sm = StreamingLosslessModel("fake/ram-compressed-engine", store_dir, device="cuda", config=ecfg)

    assert sm._tier[forced_ram] == "ram"
    cached = sm._ram_cache[forced_ram]
    assert isinstance(cached, dict), (
        "compressed RAM tier must cache raw arrays, not a decoded tensor -- "
        "got %r" % type(cached))

    with torch.no_grad():
        logits = sm.forward_logits(ids)
        logits2 = sm.forward_logits(ids)
    sm.close()

    assert torch.equal(logits, ref_logits)
    assert torch.equal(logits2, ref_logits)


def test_streamed_lm_head_is_bit_exact_and_never_stays_resident(tmp_path, monkeypatch):
    """The mechanism that makes a VRAM-MATCHED comparison against AirLLM
    possible: lm_head (the largest single tensor, and the entire VRAM gap
    against AirLLM, which streams it) must be assignable to the disk tier
    and still produce identical logits.

    Before _install_streamed_module_hooks existed this was not merely slow,
    it was broken -- _materialize_resident leaves disk-tier tensors on the
    meta device and only decoder layers had load/free hooks, so a plan that
    evicted lm_head produced a meta-tensor failure rather than a slower
    correct answer."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime import vram_planner
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=41)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/streamed-head", "streamed_head")

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(19)
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    with torch.no_grad():
        ref_logits = ref_model_gpu(input_ids=ids).logits

    manifest = __import__("json").loads((store_dir / "manifest.json").read_text())
    all_keys = [k for k, m in manifest["tensors"].items() if not m.get("row_gather")]
    row_gather_keys = [k for k, m in manifest["tensors"].items() if m.get("row_gather")]

    def fake_plan(_manifest, vram_budget_gb, ram_budget_gb=0.0, **kw):
        # Everything streams, including lm_head -- the minimum-VRAM plan.
        return vram_planner.TierPlan(
            vram_budget_bytes=int(vram_budget_gb * 1e9),
            ram_budget_bytes=int(ram_budget_gb * 1e9), vram_headroom_bytes=0,
            vram_keys=[], ram_keys=[], disk_keys=all_keys,
            row_gather_keys=row_gather_keys,
            vram_bytes=0, ram_bytes=0, disk_bytes_per_token=0,
            feasible=True, reason="")

    monkeypatch.setattr(vram_planner, "plan_from_manifest", fake_plan)

    sm = StreamingLosslessModel("fake/streamed-head", store_dir, device="cuda",
                                config=EngineConfig(vram_budget_gb=1.0, io_prefetch_depth=1))
    assert sm._tier["lm_head.weight"] == "disk"
    assert "lm_head" in sm._streamed_module_params()

    with torch.no_grad():
        logits = sm.forward_logits(ids)
        logits2 = sm.forward_logits(ids)  # must reload correctly a second time

    # After a forward pass lm_head must be back on meta, not left resident --
    # otherwise the VRAM saving this exists for silently doesn't happen.
    assert sm.model.lm_head.weight.device.type == "meta"
    sm.close()

    assert torch.equal(logits, ref_logits)
    assert torch.equal(logits2, ref_logits)


def test_ram_overlay_lm_head_is_exact_and_returns_to_meta(tmp_path, monkeypatch):
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime import vram_planner
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=43)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/ram-overlay-head", "ram_overlay_head")
    ref_model_gpu = ref_model.to("cuda")
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    with torch.no_grad():
        ref_logits = ref_model_gpu(input_ids=ids).logits

    manifest = __import__("json").loads((store_dir / "manifest.json").read_text())
    row_gather = [key for key, meta in manifest["tensors"].items()
                  if meta.get("row_gather")]
    other = [key for key in manifest["tensors"]
             if key != "lm_head.weight" and key not in row_gather]

    def fake_plan(_manifest, vram_budget_gb, ram_budget_gb=0.0, **kw):
        assert kw["forced_ram_keys"] == {"lm_head.weight"}
        return vram_planner.TierPlan(
            vram_budget_bytes=int(vram_budget_gb * 1e9),
            ram_budget_bytes=int(ram_budget_gb * 1e9), vram_headroom_bytes=0,
            vram_keys=other, ram_keys=["lm_head.weight"], disk_keys=[],
            row_gather_keys=row_gather, vram_bytes=0, ram_bytes=0,
            disk_bytes_per_token=0, feasible=True, reason="")

    monkeypatch.setattr(vram_planner, "plan_from_manifest", fake_plan)
    sm = StreamingLosslessModel(
        "fake/ram-overlay-head", store_dir, device="cuda",
        config=EngineConfig(vram_budget_gb=1.0, ram_budget_gb=1.0,
                            lm_head_policy="ram_overlay"))
    assert sm._tier["lm_head.weight"] == "ram"
    assert "lm_head" in sm._streamed_module_params()
    with torch.no_grad():
        logits = sm.forward_logits(ids)
        logits2 = sm.forward_logits(ids)
    assert sm.model.lm_head.weight.device.type == "meta"
    sm.close()
    assert torch.equal(logits, ref_logits)
    assert torch.equal(logits2, ref_logits)


@pytest.mark.parametrize("slice_elems", [1 << 25, 4096, 64])
def test_decode_slice_size_never_changes_decoded_values(tmp_path, monkeypatch, slice_elems):
    """decode_slice_elems exists purely to bound transient decode scratch
    (it is most of the residual VRAM gap against AirLLM, which stores
    weights uncompressed and needs no decode scratch at all). It is a
    memory/throughput knob and must be provably incapable of changing a
    single decoded weight -- so the smallest and largest settings must
    produce byte-identical logits."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=53)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/slice-engine", "slice")

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(4)
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    with torch.no_grad():
        ref_logits = ref_model_gpu(input_ids=ids).logits

    sm = StreamingLosslessModel("fake/slice-engine", store_dir, device="cuda",
                                config=EngineConfig(io_prefetch_depth=1,
                                                    decode_slice_elems=slice_elems))
    with torch.no_grad():
        logits = sm.forward_logits(ids)
    sm.close()

    assert torch.equal(logits, ref_logits)


def test_on_token_callback_and_stop_token_ids(tmp_path, monkeypatch):
    """These exist for the FastAPI server's SSE streaming endpoint, which
    needs to observe/stop generation token-by-token without reimplementing
    generate_greedy's loop."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=37)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/on-token-engine", "on_token")

    ids = torch.randint(0, cfg.vocab_size, (1, 4), device="cuda")
    sm = StreamingLosslessModel("fake/on-token-engine", store_dir, device="cuda",
                                config=EngineConfig(io_prefetch_depth=1))

    seen = []
    with torch.no_grad():
        seq = sm.generate_greedy(ids, max_new_tokens=8, on_token=seen.append)
    generated = seq[0, ids.shape[1]:].tolist()
    assert seen == generated

    # Stop early on whatever token the unconstrained run produced first --
    # proves stop_token_ids actually truncates generation rather than being
    # ignored.
    stop_on = generated[0]
    with torch.no_grad():
        seq2 = sm.generate_greedy(ids, max_new_tokens=8, stop_token_ids={stop_on})
    sm.close()
    generated2 = seq2[0, ids.shape[1]:].tolist()
    assert generated2 == [stop_on]


def test_schema_version_mismatch_is_a_clear_error(tmp_path, monkeypatch):
    import json as _json

    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=23)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/schema-engine", "schema")

    manifest_path = store_dir / "manifest.json"
    manifest = _json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest_path.write_text(_json.dumps(manifest))

    with pytest.raises(RuntimeError, match="schema_version"):
        StreamingLosslessModel("fake/schema-engine", store_dir, device="cuda",
                               config=EngineConfig())


def test_kv_cache_is_bit_exact_against_full_recompute(tmp_path, monkeypatch):
    """P0-4/lever E: mathematically, incremental KV-cached attention over a
    growing sequence should equal full recomputation at every step -- but
    bf16 matmul reduction order is not guaranteed bit-identical across
    different input shapes in general, and this project's entire premise is
    bit-exactness, so that equivalence is checked empirically here rather
    than assumed from the math alone."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=31)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/kv-cache-engine", "kv_cache")

    torch.manual_seed(9)
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")

    sm_cached = StreamingLosslessModel("fake/kv-cache-engine", store_dir, device="cuda",
                                       config=EngineConfig(io_prefetch_depth=1))
    with torch.no_grad():
        seq_cached = sm_cached.generate_greedy(ids.clone(), max_new_tokens=6, use_cache=True)
    sm_cached.close()

    sm_nocache = StreamingLosslessModel("fake/kv-cache-engine", store_dir, device="cuda",
                                        config=EngineConfig(io_prefetch_depth=1))
    with torch.no_grad():
        seq_nocache = sm_nocache.generate_greedy(ids.clone(), max_new_tokens=6, use_cache=False)
    sm_nocache.close()

    assert torch.equal(seq_cached, seq_nocache), (
        "KV-cached generation diverged from full-recompute generation -- "
        "not safe to default use_cache=True if this fails")


def test_truncated_weights_file_is_a_clear_error(tmp_path, monkeypatch):
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=29)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/truncated-engine", "truncated")

    weights_path = store_dir / "weights.bin"
    data = weights_path.read_bytes()
    weights_path.write_bytes(data[: len(data) // 2])

    with pytest.raises(RuntimeError, match="truncated"):
        StreamingLosslessModel("fake/truncated-engine", store_dir, device="cuda",
                               config=EngineConfig())


# -- adaptive speculation (PROPOSAL_ADAPTIVE.md) ----------------------------

def test_self_draft_logits_bit_exact_vs_truncated_reference(tmp_path, monkeypatch):
    """draft_self_logits must equal a plain in-memory forward pass over the
    SAME layer prefix -- proving the early-exit path (embeddings -> layers
    [0, N) -> model.norm -> lm_head, via temporarily truncating
    model.model.layers) is not silently doing something different from
    just having a shorter model, e.g. missing the causal mask the way this
    engine's very first hand-rolled layer loop once did."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=61)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/self-draft-engine", "self_draft")

    ref_model_gpu = ref_model.to("cuda")
    torch.manual_seed(11)
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")

    exit_layer = 2  # cfg has num_hidden_layers=3
    full_ref_layers = ref_model_gpu.model.layers
    ref_model_gpu.model.layers = full_ref_layers[:exit_layer]
    try:
        with torch.no_grad():
            ref_logits = ref_model_gpu(input_ids=ids).logits
    finally:
        ref_model_gpu.model.layers = full_ref_layers

    sm = StreamingLosslessModel("fake/self-draft-engine", store_dir, device="cuda",
                                config=EngineConfig(io_prefetch_depth=1))
    draft_logits = sm.draft_self_logits(ids, exit_layer)
    sm.close()

    assert torch.equal(draft_logits, ref_logits)


def test_draft_self_logits_rejects_out_of_range_exit_layer(tmp_path, monkeypatch):
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=62)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/self-draft-range-engine", "self_draft_range")
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device="cuda")

    sm = StreamingLosslessModel("fake/self-draft-range-engine", store_dir, device="cuda",
                                config=EngineConfig(io_prefetch_depth=1))
    with pytest.raises(ValueError):
        sm.draft_self_logits(ids, 0)
    with pytest.raises(ValueError):
        sm.draft_self_logits(ids, cfg.num_hidden_layers)  # must be < n_layers
    sm.close()


@pytest.mark.parametrize("spec_k_policy", ["fixed", "gamma", "threshold"])
def test_generate_adaptive_self_draft_matches_greedy_at_temperature_zero(
        tmp_path, monkeypatch, spec_k_policy):
    """The correctness argument from the archived adaptive test plan §3: at
    temperature<=0, verify.temperature_probs makes draft and target
    distributions one-hot, so speculative_sample_step's accept/reject
    collapses to 'accept iff draft's argmax == target's argmax, else emit
    target's argmax' -- exactly generate_greedy's rule, for ANY draft
    quality, k, or spec_k_policy. This is the real per-arm correctness
    assertion the adaptive test plan uses instead of a distributional check."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=63)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/adaptive-self-%s" % spec_k_policy,
                            "adaptive_self_" + spec_k_policy)

    torch.manual_seed(21)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device="cuda")

    sm_greedy = StreamingLosslessModel(
        "fake/adaptive-self-%s" % spec_k_policy, store_dir, device="cuda",
        config=EngineConfig(io_prefetch_depth=1))
    with torch.no_grad():
        ref_seq = sm_greedy.generate_greedy(ids.clone(), max_new_tokens=6, use_cache=False)
    sm_greedy.close()

    sm_adaptive = StreamingLosslessModel(
        "fake/adaptive-self-%s" % spec_k_policy, store_dir, device="cuda",
        config=EngineConfig(io_prefetch_depth=1, draft_mode="self",
                            draft_exit_layer=1, spec_k=3,
                            spec_k_policy=spec_k_policy))
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        adaptive_seq, policy = sm_adaptive.generate_adaptive(
            ids.clone(), max_new_tokens=6, temperature=0.0, generator=gen)
    sm_adaptive.close()

    assert torch.equal(adaptive_seq, ref_seq), (
        "self-draft generate_adaptive (%s policy) diverged from generate_greedy "
        "at temperature=0 -- the exact-argmax guarantee should make this "
        "impossible regardless of draft quality" % spec_k_policy)
    assert policy.choose_k() >= 1


def test_generate_adaptive_model_draft_matches_greedy_at_temperature_zero(tmp_path, monkeypatch):
    """Same guarantee, draft_mode='model': a completely different (different
    seed, different weights) small model as the draft must still reproduce
    greedy exactly at temperature=0 -- proving the guarantee does not depend
    on the draft being self-derived."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=64)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/adaptive-model-engine", "adaptive_model")
    _, unrelated_draft = _build_tiny_model(tie=False, seed=999)  # deliberately different weights
    draft_gpu = unrelated_draft.to("cuda")

    torch.manual_seed(22)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device="cuda")

    sm_greedy = StreamingLosslessModel("fake/adaptive-model-engine", store_dir, device="cuda",
                                       config=EngineConfig(io_prefetch_depth=1))
    with torch.no_grad():
        ref_seq = sm_greedy.generate_greedy(ids.clone(), max_new_tokens=5, use_cache=False)
    sm_greedy.close()

    sm_adaptive = StreamingLosslessModel(
        "fake/adaptive-model-engine", store_dir, device="cuda",
        config=EngineConfig(io_prefetch_depth=1, draft_mode="model", spec_k=4))
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        adaptive_seq, _ = sm_adaptive.generate_adaptive(
            ids.clone(), max_new_tokens=5, draft_model=draft_gpu,
            temperature=0.0, generator=gen)
    sm_adaptive.close()

    assert torch.equal(adaptive_seq, ref_seq)


def test_rollback_cached_speculation_matches_full_prefix_verification(
        tmp_path, monkeypatch):
    """H18 must change only target-prefix reuse, never accepted tokens."""
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=164)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/spec-cache-engine", "spec_cache")
    _, unrelated_draft = _build_tiny_model(tie=False, seed=1999)
    draft_gpu = unrelated_draft.to("cuda")
    ids = torch.randint(0, cfg.vocab_size, (1, 7), device="cuda")

    outputs = []
    cached_stats = None
    for target_cache in (False, True):
        engine = StreamingLosslessModel(
            "fake/spec-cache-engine", store_dir, device="cuda",
            config=EngineConfig(io_prefetch_depth=1, draft_mode="model",
                                spec_k=2, spec_target_cache=target_cache))
        generator = torch.Generator(device="cuda").manual_seed(123)
        with torch.no_grad():
            sequence, _ = engine.generate_adaptive(
                ids.clone(), max_new_tokens=7, draft_model=draft_gpu,
                temperature=0.0, generator=generator)
        outputs.append(sequence)
        if target_cache:
            cached_stats = (engine.stats.spec_cache_crops,
                            engine.stats.spec_cached_prefix_tokens)
        engine.close()

    assert torch.equal(outputs[0], outputs[1])
    assert cached_stats is not None and cached_stats[0] > 0
    assert cached_stats[1] > 0


def test_generate_adaptive_rejects_draft_mode_none(tmp_path, monkeypatch):
    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel

    cfg, ref_model = _build_tiny_model(tie=False, seed=65)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/adaptive-none-engine", "adaptive_none")
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device="cuda")

    sm = StreamingLosslessModel("fake/adaptive-none-engine", store_dir, device="cuda",
                                config=EngineConfig(io_prefetch_depth=1))
    with pytest.raises(ValueError, match="draft_mode"):
        sm.generate_adaptive(ids, max_new_tokens=4, temperature=0.0)
    sm.close()


def test_pin_draft_layers_prioritizes_early_layers_over_late_ones(tmp_path, monkeypatch):
    """Mechanism C, engine-integrated: with pin_draft_layers=True and a VRAM
    budget sized to hold exactly one layer's worth of tensors beyond fixed
    headroom, the planner must give that room to the draft layer (layer 0)
    over an equally-sized later layer (layer 2) -- vram_planner's isolated
    unit tests already prove the ranking math; this proves
    _compute_tier_assignment actually wires spec_k/draft_exit_layer through
    to it in the real engine.

    This tiny model's tensors are all well under the 65536-element
    compression threshold (hidden_size=32, intermediate_size=64), so every
    tensor is stored raw and has comp_bytes == orig_bytes -- i.e. value
    density 1.0 for everyone EXCEPT the draft layer, which pin_draft_layers
    scales to spec_k+1. That makes the budget arithmetic below exact rather
    than dependent on how well any particular tensor happens to compress.
    """
    import json

    from afterimage.runtime.config import EngineConfig
    from afterimage.runtime.streaming_engine import StreamingLosslessModel
    from afterimage.runtime.vram_planner import DEFAULT_ACTIVATION_SLACK_BYTES

    cfg, ref_model = _build_tiny_model(tie=False, seed=66, intermediate_size=64)
    store_dir = _make_store(tmp_path, monkeypatch, ref_model, cfg,
                            "fake/pin-draft-engine", "pin_draft")

    man = json.loads((store_dir / "manifest.json").read_text())
    eligible = {k: m for k, m in man["tensors"].items() if not m.get("row_gather")}
    for m in eligible.values():
        assert m["comp_bytes"] == m["orig_bytes"], (
            "test assumption violated: this tiny model's tensors must all "
            "be stored raw for the budget arithmetic below to be exact")

    decode_slice_elems = 64  # shrink decode scratch to ~0 so headroom is dominated
                             # by the (fixed, known) activation slack constant
    largest = max(m["orig_bytes"] for m in eligible.values())
    headroom_bytes = largest + decode_slice_elems * 10 + DEFAULT_ACTIVATION_SLACK_BYTES
    layer0_bytes = sum(m["orig_bytes"] for k, m in eligible.items()
                       if k.startswith("model.layers.0."))
    # Exactly enough room beyond headroom for layer 0's tensors and nothing else.
    vram_budget_gb = (headroom_bytes + layer0_bytes + 1) / 1e9

    sm = StreamingLosslessModel(
        "fake/pin-draft-engine", store_dir, device="cuda",
        config=EngineConfig(io_prefetch_depth=1, vram_budget_gb=vram_budget_gb,
                            decode_slice_elems=decode_slice_elems, draft_mode="self",
                            draft_exit_layer=1, spec_k=8, pin_draft_layers=True))
    tiers = dict(sm._tier)
    sm.close()

    layer0_tiers = {tiers[k] for k in eligible if k.startswith("model.layers.0.")}
    layer2_tiers = {tiers[k] for k in eligible if k.startswith("model.layers.2.")}
    lm_head_tier = tiers.get("lm_head.weight")

    assert layer0_tiers == {"vram"}, (
        "draft layer 0 should win the one available slot once "
        "pin_draft_layers=True, got tiers %r" % layer0_tiers)
    assert layer2_tiers == {"disk"}, (
        "non-draft layer 2 should NOT get the slot draft layer 0 won -- "
        "got tiers %r" % layer2_tiers)
    assert lm_head_tier == "disk", (
        "lm_head has the same (raw, density-1.0) cost as layer 2 and should "
        "lose to the draft layer's scaled density the same way, got %r"
        % lm_head_tier)
