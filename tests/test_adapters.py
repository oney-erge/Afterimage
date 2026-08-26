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
    LLAMA4_VISION_LAYOUT,
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
    # Verified 2026-08-26 against real Hub config.json files (web research
    # into current model families, not name-guessing) plus a real
    # meta-device build per family here: all four still fit the plain
    # ``model.layers``/``embed_tokens`` shape despite being MoE or hybrid
    # linear-attention/Mamba architectures internally.
    ("causal-language", "Glm4MoeConfig", TINY),  # GLM-4.5/4.6 (MoE)
    ("causal-language", "DeepseekV32Config", TINY),  # DeepSeek-V3.2 (MoE)
    ("causal-language", "Qwen3NextConfig", TINY),  # Qwen3-Next (hybrid linear attn)
    ("causal-language", "GptOssConfig",
     dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
          num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
          num_local_experts=4)),  # OpenAI gpt-oss (MoE)
    ("causal-language", "JambaConfig", TINY),  # AI21 Jamba (hybrid Mamba+attention)
    ("causal-language", "GraniteMoeHybridConfig",
     dict(TINY, mamba_n_heads=8)),  # IBM Granite 4 (hybrid Mamba+attention);
    # mamba_n_heads must divide mamba_expand * hidden_size (2 * 32 = 64).
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
    known_layouts = (
        {spec.layout for spec in LAYOUTS}
        | {"vision-language", LLAMA4_VISION_LAYOUT.layout}
    )
    for architecture, layout in ARCHITECTURE_LAYOUTS.items():
        assert layout in known_layouts, (
            f"{architecture} maps to unknown layout {layout!r}")


def _build_llama4():
    """Llama4ForConditionalGeneration on the meta device: the smallest
    Llama4Config that still has a real vision_config (so the classifier's
    vision_config-based detection has something real to find) and a real
    text_config (so the decoder stack exists to resolve against)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text_config = transformers.Llama4TextConfig(
            hidden_size=32, intermediate_size=64, intermediate_size_mlp=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, vocab_size=128, num_local_experts=2,
            num_experts_per_tok=1)
        vision_config = transformers.Llama4VisionConfig(
            hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, image_size=32, patch_size=16)
        config = transformers.Llama4Config(
            text_config=text_config, vision_config=vision_config)
        with torch.device("meta"):
            return transformers.AutoModelForImageTextToText.from_config(config)


def test_llama4_resolves_the_dedicated_vision_layout():
    model = _build_llama4()
    adapter = resolve_model_adapter(model)
    assert adapter.capabilities.layout == "llama4-vision-language"
    assert adapter.capabilities.modality == "vision-text"


def test_llama4_output_head_is_on_language_model_not_the_outer_wrapper():
    """The structurally novel part of this layout: unlike every other
    registered family, the lm_head is not on the top-level model itself but
    on the inner ``language_model`` -- a distinct object with its own
    lm_head, not the outer vision+text wrapper's."""
    model = _build_llama4()
    adapter = resolve_model_adapter(model)
    assert adapter.output_head is model.language_model.lm_head
    assert adapter.output_head is not getattr(model, "lm_head", None)


def test_llama4_vision_model_and_layer_weights_resolve_to_real_parameters():
    model = _build_llama4()
    adapter = resolve_model_adapter(model)
    names = set(dict(model.named_parameters()))

    embed_key = adapter.embedding_prefix + ".weight"
    assert embed_key in names, f"{embed_key} not among {sorted(names)[:6]}"
    assert len(adapter.layers) == 2
    assert any(n.startswith(adapter.layer_key(0) + ".") for n in names)
    assert adapter.vision_model is not None


def test_llama4_classifier_detects_vision_via_vision_config_not_name_substring():
    """model_type "llama4" and the class name contain none of "vision",
    "vl", or "image" -- the substring heuristic alone would miss this
    family entirely. The fix checks for a real vision_config instead."""
    verdict = classify_config({
        "architectures": ["Llama4ForConditionalGeneration"],
        "model_type": "llama4",
        "vision_config": {"hidden_size": 16},
    })
    assert verdict["modality"] == "vision-text"
    assert verdict["layout"] == "llama4-vision-language"


def test_deepseek_and_glm_moe_are_classified_as_moe_without_moe_in_the_name():
    """DeepseekV3ForCausalLM and DeepseekV32ForCausalLM carry no "moe"
    substring in either the class name or "deepseek_v3"/"deepseek_v32"
    model_type, so the cheap substring check alone would under-classify a
    real sparse MoE model as fully "expected" rather than "experimental"."""
    for architecture, model_type in [
        ("DeepseekV3ForCausalLM", "deepseek_v3"),
        ("DeepseekV32ForCausalLM", "deepseek_v32"),
    ]:
        verdict = classify_config({
            "architectures": [architecture], "model_type": model_type})
        assert verdict["mixture_of_experts"] is True, architecture
        assert verdict["execution"] == "experimental", architecture


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
