"""The classifier and the structural resolver must agree.

They used to disagree: ``classify_config`` carried a hand-maintained
allowlist while ``resolve_model_adapter`` inspected the object graph, so
families the engine can actually stream (Phi, OLMo 2, StarCoder2, Granite)
were reported to users as "No executable Afterimage adapter is known for
this layout yet."  These tests build a real (meta-device, no weights, no
network) model per registered architecture and assert the two agree, so the
table cannot drift again.
"""
from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from afterimage.runtime.adapters import (  # noqa: E402
    ARCHITECTURE_LAYOUTS,
    LAYOUTS,
    classify_config,
    resolve_model_adapter,
)

TINY = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=128)

# One concrete config per structural layout. Kept small deliberately: these
# are built on the meta device, so nothing is allocated and nothing is
# downloaded, but the object graph is exactly what a real checkpoint has.
LAYOUT_CASES = [
    ("causal-language", "LlamaConfig", TINY),
    ("causal-language", "Qwen3Config", TINY),
    ("causal-language", "Gemma2Config", TINY),
    ("causal-language", "Olmo2Config", TINY),
    ("causal-language", "Starcoder2Config", TINY),
    ("causal-language", "GraniteConfig", TINY),
    ("gpt2-transformer", "GPT2Config",
     dict(n_embd=32, n_layer=2, n_head=4, vocab_size=128)),
    ("mpt-transformer", "MptConfig",
     dict(d_model=32, n_layers=2, n_heads=4, vocab_size=128)),
    ("bloom-transformer", "BloomConfig",
     dict(hidden_size=32, n_layer=2, n_head=4, vocab_size=128)),
    ("bloom-transformer", "FalconConfig",
     dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
          vocab_size=128)),
    ("neox", "GPTNeoXConfig",
     dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
          num_attention_heads=4, vocab_size=128)),
]


def _build(config_cls_name: str, kwargs: dict):
    config_cls = getattr(transformers, config_cls_name, None)
    if config_cls is None:
        pytest.skip("%s not in transformers %s"
                    % (config_cls_name, transformers.__version__))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = config_cls(**kwargs)
        with torch.device("meta"):
            return transformers.AutoModelForCausalLM.from_config(config)


@pytest.mark.parametrize("expected_layout,config_cls_name,kwargs", LAYOUT_CASES)
def test_resolver_finds_the_expected_layout(expected_layout, config_cls_name, kwargs):
    model = _build(config_cls_name, kwargs)
    adapter = resolve_model_adapter(model)
    assert adapter.capabilities.layout == expected_layout


@pytest.mark.parametrize("expected_layout,config_cls_name,kwargs", LAYOUT_CASES)
def test_adapter_addresses_real_layer_and_embedding_weights(
        expected_layout, config_cls_name, kwargs):
    """The prefixes are used to build state-dict keys, so they must match
    the model's real parameter names -- a prefix that looks plausible but
    addresses nothing would make every weight 'missing from the store'."""
    model = _build(config_cls_name, kwargs)
    adapter = resolve_model_adapter(model)
    names = set(dict(model.named_parameters()))

    embed_key = adapter.embedding_prefix + ".weight"
    assert embed_key in names, f"{embed_key} not among {sorted(names)[:6]}"

    assert len(adapter.layers) == 2
    layer_param_names = [n for n in names if adapter.is_layer_key(n)]
    assert layer_param_names, "adapter.is_layer_key matched no real parameter"
    # layer_key(0) must be a real prefix of real parameters.
    assert any(n.startswith(adapter.layer_key(0) + ".") for n in names)


@pytest.mark.parametrize("expected_layout,config_cls_name,kwargs", LAYOUT_CASES)
def test_classifier_agrees_with_resolver(expected_layout, config_cls_name, kwargs):
    """The bug this file exists for: a model the resolver handles must never
    be reported to a user as having no known adapter."""
    model = _build(config_cls_name, kwargs)
    resolve_model_adapter(model)  # must not raise
    architecture = type(model).__name__
    verdict = classify_config({"architectures": [architecture],
                               "model_type": model.config.model_type})
    assert verdict["execution"] != "download-only", (
        f"{architecture} resolves structurally but classify_config still "
        f"reports download-only: {verdict['execution_reason']}")
    assert verdict["layout"] == expected_layout


def test_every_registered_architecture_maps_to_a_real_layout():
    known_layouts = {spec.layout for spec in LAYOUTS} | {"vision-language"}
    for architecture, layout in ARCHITECTURE_LAYOUTS.items():
        assert layout in known_layouts, (
            f"{architecture} maps to unknown layout {layout!r}")


def test_genuinely_unknown_architecture_is_still_reported_as_download_only():
    """The fix must not turn the classifier into a rubber stamp."""
    verdict = classify_config({"architectures": ["TotallyMadeUpForCausalLM"],
                               "model_type": "madeup"})
    assert verdict["execution"] == "download-only"
    assert verdict["layout"] is None


def test_missing_architecture_metadata_is_unknown_not_download_only():
    verdict = classify_config({"architectures": [], "model_type": ""})
    assert verdict["execution"] == "unknown"


def test_unsupported_layout_raises_a_directly_actionable_error():
    class Opaque:
        config = type("C", (), {"architectures": ["OpaqueForCausalLM"]})()

    with pytest.raises(RuntimeError, match="not yet executable"):
        resolve_model_adapter(Opaque())
