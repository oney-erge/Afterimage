"""Unit tests for cli.py's pure helper functions -- the CLI's actual
`run` command needs a GPU-or-CPU engine and a real compressed store, so it
isn't exercised end-to-end here. These test the two pieces that fixed the
run command's biggest usability gaps: applying a chat template correctly
(_render_prompt) and streaming decoded text incrementally without
corrupting multi-token characters (_make_stream_printer).
"""
from afterimage.cli import _benchmark_disk_read_mb_s, _make_stream_printer, _render_prompt


class _FakeTokenizer:
    """Records apply_chat_template calls; supports_thinking=False makes it
    raise TypeError on enable_thinking, like an older tokenizer would."""

    def __init__(self, supports_thinking: bool = True):
        self.supports_thinking = supports_thinking
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        if "enable_thinking" in kwargs and not self.supports_thinking:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return "<rendered:%s:%s>" % (messages[0]["content"], kwargs.get("enable_thinking"))


def test_render_prompt_disables_thinking_by_default():
    tok = _FakeTokenizer()
    out = _render_prompt(tok, "hello", think=False)
    assert out == "<rendered:hello:False>"
    assert tok.calls[0]["kwargs"]["enable_thinking"] is False
    assert tok.calls[0]["kwargs"]["add_generation_prompt"] is True
    assert tok.calls[0]["kwargs"]["tokenize"] is False
    assert tok.calls[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_render_prompt_think_flag_omits_enable_thinking():
    tok = _FakeTokenizer()
    _render_prompt(tok, "hello", think=True)
    assert "enable_thinking" not in tok.calls[0]["kwargs"]


def test_render_prompt_falls_back_when_tokenizer_rejects_enable_thinking():
    """Older chat templates don't accept enable_thinking at all -- the
    fallback must retry without it rather than crash, exactly like
    bench/prompt_suite.py's render_chat_prompt does."""
    tok = _FakeTokenizer(supports_thinking=False)
    out = _render_prompt(tok, "hello", think=False)
    assert out == "<rendered:hello:None>"
    assert len(tok.calls) == 1
    assert "enable_thinking" not in tok.calls[0]["kwargs"]


class _FakeStreamTokenizer:
    """decode() joins a simple id->str map; skip_special_tokens is accepted
    but ignored since these tests don't use special ids."""

    def __init__(self, vocab: dict[int, str]):
        self.vocab = vocab

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self.vocab[i] for i in token_ids)


def test_stream_printer_prints_incremental_suffix(capsys):
    tok = _FakeStreamTokenizer({1: "Hel", 2: "lo ", 3: "world"})
    on_token, state = _make_stream_printer(tok)
    on_token(1)
    on_token(2)
    on_token(3)
    out = capsys.readouterr().out
    assert out == "Hello world"
    assert state["printed"] == "Hello world"


def test_stream_printer_handles_empty_increment():
    """A token that decodes to the same running text as before (can happen
    with some special-token handling) must not print anything or corrupt
    the tracked state."""
    tok = _FakeStreamTokenizer({1: "abc", 2: ""})
    on_token, state = _make_stream_printer(tok)
    on_token(1)
    on_token(2)
    assert state["printed"] == "abc"


def test_stream_printer_skips_non_monotonic_decode_without_crashing(capsys):
    """If decode() of the growing id list ever stops being a prefix
    extension of what was already printed (not prefix-stable), the
    increment must be silently skipped -- never print garbled or
    duplicated text -- and state must stay internally consistent for the
    next call."""
    class _Weird:
        def decode(self, token_ids, skip_special_tokens=True):
            # Deliberately non-monotonic: adding a second token produces
            # text that does NOT start with the first token's decode.
            return "AB" if len(token_ids) == 1 else "XYZ"

    on_token, state = _make_stream_printer(_Weird())
    on_token(1)
    assert state["printed"] == "AB"
    on_token(2)  # "XYZ" does not start with "AB" -- must be skipped, not raise
    out = capsys.readouterr().out
    assert out == "AB"
    assert state["printed"] == "AB"


def test_disk_benchmark_measures_a_real_positive_throughput(tmp_path):
    result = _benchmark_disk_read_mb_s(tmp_path, size_mb=8)
    assert result is not None
    mb_s, cache_dropped = result
    assert mb_s > 0
    assert isinstance(cache_dropped, bool)


def test_disk_benchmark_cleans_up_its_probe_file(tmp_path):
    _benchmark_disk_read_mb_s(tmp_path, size_mb=8)
    assert list(tmp_path.iterdir()) == []


def test_disk_benchmark_returns_none_on_write_failure(tmp_path, monkeypatch):
    """Must never raise -- doctor calls this unconditionally, and a probe
    failure (read-only filesystem, disk full, permissions) is a reason to
    report 'couldn't measure', not a reason for `doctor` to crash."""
    def _boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr("builtins.open", _boom)
    assert _benchmark_disk_read_mb_s(tmp_path, size_mb=8) is None


def test_every_subcommand_is_wired_to_its_own_function():
    """Regression test for a real bug caught while adding pull/verify: the
    pull subparser was assigned to the same local variable name (`p`) the
    top-level parser itself used, silently overwriting it -- every
    subcommand's --help, and argument parsing generally, broke, but
    `python -m py_compile` and ruff both saw nothing wrong since it's valid
    Python, just the wrong object. Parsing one full command per subcommand
    and checking it resolves to the intended handler is the only thing that
    actually catches this class of bug."""
    import afterimage.cli as cli_mod
    parser = cli_mod.build_parser()

    cases = [
        (["doctor"], cli_mod.cmd_doctor),
        (["compress", "org/model"], cli_mod.cmd_compress),
        (["verify", "org/model"], cli_mod.cmd_verify),
        (["pull", "org/model", "--store-repo", "org/store"], cli_mod.cmd_pull),
        (["run", "org/model", "hello"], cli_mod.cmd_run),
        (["quickstart"], cli_mod.cmd_quickstart),
        (["serve", "--open"], cli_mod.cmd_serve),
    ]
    for argv, expected_func in cases:
        args = parser.parse_args(argv)
        assert args.func is expected_func, argv

    assert parser.parse_args(["serve", "--open"]).open is True


def test_pull_parses_repo_type_and_verify_defaults():
    import afterimage.cli as cli_mod
    parser = cli_mod.build_parser()
    args = parser.parse_args(["pull", "org/model", "--store-repo", "org/store"])
    assert args.repo_type == "model"
    assert args.verify is True
    args2 = parser.parse_args(
        ["pull", "org/model", "--store-repo", "org/store",
         "--repo-type", "dataset", "--no-verify"])
    assert args2.repo_type == "dataset"
    assert args2.verify is False


def _pull_args(tmp_path, model="org/model", store_repo="org/store", verify=True,
               repo_type="model"):
    import types
    return types.SimpleNamespace(model=model, store_repo=store_repo, repo_type=repo_type,
                                 store=str(tmp_path / "store"), verify=verify)


def test_pull_fetches_manifest_and_weights_into_the_store_dir(tmp_path, monkeypatch):
    import json

    from afterimage.cli import cmd_pull

    remote = tmp_path / "remote"
    remote.mkdir()
    manifest_path = remote / "manifest.json"
    weights_path = remote / "weights.bin"
    manifest_path.write_text(json.dumps({
        "model_id": "org/model", "total_comp_bytes": 123, "ratio": 1.45}))
    weights_path.write_bytes(b"fake-weights")

    def fake_hf_hub_download(repo_id, filename, repo_type=None):
        return str(remote / filename)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        "afterimage.runtime.binstore.verify_store", lambda store_dir: (True, []))

    args = _pull_args(tmp_path)
    assert cmd_pull(args) == 0
    store_dir = tmp_path / "store"
    assert (store_dir / "manifest.json").read_text() == manifest_path.read_text()
    assert (store_dir / "weights.bin").read_bytes() == b"fake-weights"


def test_pull_warns_but_proceeds_on_model_id_mismatch(tmp_path, monkeypatch, capsys):
    import json

    from afterimage.cli import cmd_pull

    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "manifest.json").write_text(json.dumps({"model_id": "org/other-model"}))
    (remote / "weights.bin").write_bytes(b"x")

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename, repo_type=None: str(remote / filename))
    monkeypatch.setattr(
        "afterimage.runtime.binstore.verify_store", lambda store_dir: (True, []))

    args = _pull_args(tmp_path, model="org/model")
    assert cmd_pull(args) == 0
    assert "org/other-model" in capsys.readouterr().err


def test_pull_returns_1_when_the_repo_has_no_store(tmp_path, monkeypatch, capsys):
    from afterimage.cli import cmd_pull

    def fake_hf_hub_download(repo_id, filename, repo_type=None):
        raise Exception("404")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    args = _pull_args(tmp_path)
    assert cmd_pull(args) == 1
    assert "Could not fetch" in capsys.readouterr().err


def test_pull_no_verify_skips_the_checksum_pass(tmp_path, monkeypatch):
    import json

    from afterimage.cli import cmd_pull

    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "manifest.json").write_text(json.dumps({"model_id": "org/model"}))
    (remote / "weights.bin").write_bytes(b"x")

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename, repo_type=None: str(remote / filename))

    def _must_not_be_called(store_dir):
        raise AssertionError("verify_store should not run when --no-verify is set")
    monkeypatch.setattr("afterimage.runtime.binstore.verify_store", _must_not_be_called)

    args = _pull_args(tmp_path, verify=False)
    assert cmd_pull(args) == 0


def test_verify_reports_ok(tmp_path, monkeypatch, capsys):
    import json

    from afterimage.cli import cmd_verify

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    (store_dir / "manifest.json").write_text(json.dumps({"model_id": "org/model"}))
    monkeypatch.setattr(
        "afterimage.runtime.binstore.verify_store", lambda store_dir: (True, []))

    args = _pull_args(tmp_path)
    assert cmd_verify(args) == 0
    assert "OK" in capsys.readouterr().out


def test_verify_reports_failure_and_lists_bad_keys(tmp_path, monkeypatch, capsys):
    import json

    from afterimage.cli import cmd_verify

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    (store_dir / "manifest.json").write_text(json.dumps({"model_id": "org/model"}))
    monkeypatch.setattr(
        "afterimage.runtime.binstore.verify_store",
        lambda store_dir: (False, ["model.layers.0.mlp.weight"]))

    args = _pull_args(tmp_path)
    assert cmd_verify(args) == 1
    err = capsys.readouterr().err
    assert "FAILED" in err
    assert "model.layers.0.mlp.weight" in err


def test_verify_missing_store_returns_1_without_touching_binstore(tmp_path, monkeypatch, capsys):
    from afterimage.cli import cmd_verify

    def _must_not_be_called(store_dir):
        raise AssertionError("verify_store should not run against a nonexistent store")
    monkeypatch.setattr("afterimage.runtime.binstore.verify_store", _must_not_be_called)

    args = _pull_args(tmp_path)  # store dir was never created
    assert cmd_verify(args) == 1
    assert "No compressed store" in capsys.readouterr().err


def _compress_args(tmp_path, *, force_raw_storage=False, dry_run=False, yes=True):
    import types
    return types.SimpleNamespace(
        model="org/model", out=str(tmp_path / "store"), chunk_size=1024,
        quantize=None, force_raw_storage=force_raw_storage,
        progress_every=50, workers=None, dry_run=dry_run, yes=yes)


def test_compress_dry_run_reports_full_size_for_force_raw_storage(
        tmp_path, monkeypatch, capsys):
    """A --force-raw-storage store is roughly checkpoint-sized, not
    compressed -- the dry-run estimate must say so, or a real (non-dry)
    run could start on a host without enough free space for it."""
    from afterimage.cli import cmd_compress

    monkeypatch.setattr("afterimage.cli._estimate_download_bytes", lambda model: 10_000_000_000)

    args = _compress_args(tmp_path, force_raw_storage=True, dry_run=True)
    assert cmd_compress(args) == 0
    out = capsys.readouterr().out
    assert "not compressed" in out
    # 10 GB download -> ~10 GB raw store (not ~6.9 GB at the 1.453x ratio).
    assert "store      : ~10.0 GB" in out


def test_compress_dry_run_reports_compressed_estimate_by_default(
        tmp_path, monkeypatch, capsys):
    from afterimage.cli import cmd_compress

    monkeypatch.setattr("afterimage.cli._estimate_download_bytes", lambda model: 10_000_000_000)

    args = _compress_args(tmp_path, force_raw_storage=False, dry_run=True)
    assert cmd_compress(args) == 0
    out = capsys.readouterr().out
    assert "at the measured" in out
    assert "not compressed" not in out


def test_compress_passes_force_raw_storage_into_the_engine_config(tmp_path, monkeypatch):
    """cmd_compress must actually thread --force-raw-storage through to
    compress_model_to_disk's EngineConfig, not just print about it."""
    import afterimage.cli as cli_mod

    seen = {}

    def fake_compress_model_to_disk(model_id, out_dir, config=None, **kwargs):
        seen["force_raw_storage"] = config.force_raw_storage
        return {"total_orig_bytes": 1, "total_comp_bytes": 1, "ratio": 1.0}

    monkeypatch.setattr(
        "afterimage.runtime.streaming_engine.compress_model_to_disk",
        fake_compress_model_to_disk)
    monkeypatch.setattr(cli_mod, "_disk_preflight", lambda *a, **k: True)

    args = _compress_args(tmp_path, force_raw_storage=True, dry_run=False)
    assert cli_mod.cmd_compress(args) == 0
    assert seen["force_raw_storage"] is True
