"""Model-layout adapters shared by discovery, preflight, and the runtime.

Adapters describe structure only.  Resolving an adapter is not proof that a
checkpoint has passed Afterimage's end-to-end correctness suite.

Two separate questions are answered here, and keeping them separate matters:

``resolve_model_adapter`` asks *structurally* whether a loaded model exposes a
decoder stack this engine can stream, by looking at the object graph.  It is
the ground truth, because it inspects the thing that will actually execute.

``classify_config`` asks the same question from Hub *metadata* alone, before
any weights are downloaded, so discovery can label a search result without
paying for a checkpoint.  It can only consult architecture names, so it is
necessarily an approximation of the resolver.

Those two used to disagree: the classifier carried a hand-maintained allowlist
while the resolver was purely structural, so families the resolver handles
perfectly well (Phi, OLMo 2, StarCoder2, Granite, ...) were reported as
"No executable Afterimage adapter is known for this layout yet."  The
architecture table below is now the single source shared by both, and
``tests/test_adapters.py`` builds one model per registered architecture and
asserts the two agree.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Iterable


@dataclasses.dataclass(frozen=True)
class AdapterCapabilities:
    layout: str
    modality: str
    mixture_of_experts: bool
    supports_certified_head: bool


@dataclasses.dataclass(frozen=True)
class LayoutSpec:
    """One structural decoder layout.

    ``container_attr`` is the attribute on the top-level model holding the
    decoder (``model`` for Llama-family, ``transformer`` for GPT-2-family,
    ``gpt_neox`` for NeoX).  ``layers_attr`` and ``embed_attr`` are the
    attribute names *inside* that container.  Everything else in the engine
    addresses weights by string key, so these three names are the whole
    difference between families.
    """

    layout: str
    container_attr: str
    layers_attr: str
    embed_attr: str
    modality: str = "text"
    supports_certified_head: bool = True

    @property
    def prefix(self) -> str:
        return self.container_attr


#: Structural layouts, tried in order by ``resolve_model_adapter``.
#: ``model.layers``/``embed_tokens`` is first because it covers the large
#: majority of current checkpoints and is the release-verified path.
LAYOUTS: tuple[LayoutSpec, ...] = (
    LayoutSpec("causal-language", "model", "layers", "embed_tokens"),
    # GPT-2 / Falcon / BLOOM / MPT keep the decoder under ``transformer`` and
    # disagree only on the two inner names.
    LayoutSpec("gpt2-transformer", "transformer", "h", "wte",
               supports_certified_head=False),
    LayoutSpec("mpt-transformer", "transformer", "blocks", "wte",
               supports_certified_head=False),
    LayoutSpec("bloom-transformer", "transformer", "h", "word_embeddings",
               supports_certified_head=False),
    LayoutSpec("neox", "gpt_neox", "layers", "embed_in",
               supports_certified_head=False),
)

#: Vision-language models nest the decoder one level deeper and additionally
#: carry a vision tower; handled separately because the container is found by
#: walking two attributes rather than one.
VISION_LAYOUT = LayoutSpec(
    "vision-language", "model.language_model", "layers", "embed_tokens",
    modality="vision-text", supports_certified_head=False)


class ModelAdapter:
    """Access the language decoder without hard-coding one object path."""

    def __init__(self, model: Any, *, spec: LayoutSpec, language_model: Any):
        self.model = model
        self.spec = spec
        self.language_model = language_model
        self.layer_prefix = f"{spec.prefix}.{spec.layers_attr}"
        self.embedding_prefix = f"{spec.prefix}.{spec.embed_attr}"
        model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
        architectures = " ".join(
            getattr(getattr(model, "config", None), "architectures", None) or []
        )
        moe = "moe" in f"{model_type} {architectures}".lower() or any(
            hasattr(getattr(layer, "mlp", None), "experts")
            for layer in list(self.layers)[:1]
        )
        self.capabilities = AdapterCapabilities(
            layout=spec.layout,
            modality=spec.modality,
            mixture_of_experts=moe,
            supports_certified_head=spec.supports_certified_head,
        )

    @property
    def layers(self):
        return getattr(self.language_model, self.spec.layers_attr)

    @layers.setter
    def layers(self, value) -> None:
        setattr(self.language_model, self.spec.layers_attr, value)

    @property
    def embedding(self):
        return getattr(self.language_model, self.spec.embed_attr)

    @property
    def output_head(self):
        return self.model.lm_head

    @property
    def language_config(self):
        config = self.model.config
        return getattr(config, "text_config", config)

    @property
    def vision_model(self):
        return getattr(getattr(self.model, "model", None), "visual", None)

    def is_layer_key(self, key: str) -> bool:
        return key.startswith(self.layer_prefix + ".")

    def layer_key(self, index: int, suffix: str = "") -> str:
        base = f"{self.layer_prefix}.{index}"
        return f"{base}.{suffix}" if suffix else base


def _matches(container: Any, spec: LayoutSpec) -> bool:
    return (
        container is not None
        and hasattr(container, spec.layers_attr)
        and hasattr(container, spec.embed_attr)
    )


def resolve_model_adapter(model: Any) -> ModelAdapter:
    """Resolve a supported structural layout or raise a direct error."""

    inner = getattr(model, "model", None)

    # Vision-language is checked before the plain causal path: these models
    # also expose ``model.layers`` on some releases, and the deeper decoder
    # plus vision tower is the more specific match.
    language = getattr(inner, "language_model", None)
    if _matches(language, VISION_LAYOUT) and hasattr(inner, "visual"):
        return ModelAdapter(model, spec=VISION_LAYOUT, language_model=language)

    for spec in LAYOUTS:
        container = getattr(model, spec.container_attr, None)
        if _matches(container, spec):
            return ModelAdapter(model, spec=spec, language_model=container)

    architecture = getattr(getattr(model, "config", None), "architectures", None)
    raise RuntimeError(
        "model layout is not yet executable by Afterimage: %s. "
        "The checkpoint can still be downloaded and retained locally."
        % (architecture or [type(model).__name__])
    )


#: Architecture name -> layout name, for metadata-only classification.
#: Every entry is asserted against ``resolve_model_adapter`` in
#: ``tests/test_adapters.py``, so this table cannot silently drift from the
#: structural resolver the way the previous hand-maintained allowlist did.
ARCHITECTURE_LAYOUTS: dict[str, str] = {
    # --- model.layers / embed_tokens -------------------------------------
    "LlamaForCausalLM": "causal-language",
    "MistralForCausalLM": "causal-language",
    "Qwen2ForCausalLM": "causal-language",
    "Qwen3ForCausalLM": "causal-language",
    "Qwen2MoeForCausalLM": "causal-language",
    "Qwen3MoeForCausalLM": "causal-language",
    "MixtralForCausalLM": "causal-language",
    "GemmaForCausalLM": "causal-language",
    "Gemma2ForCausalLM": "causal-language",
    "Gemma3ForCausalLM": "causal-language",
    "Gemma3TextForCausalLM": "causal-language",
    "PhiForCausalLM": "causal-language",
    "Phi3ForCausalLM": "causal-language",
    "StableLmForCausalLM": "causal-language",
    "OlmoForCausalLM": "causal-language",
    "Olmo2ForCausalLM": "causal-language",
    "OlmoeForCausalLM": "causal-language",
    "CohereForCausalLM": "causal-language",
    "Cohere2ForCausalLM": "causal-language",
    "Starcoder2ForCausalLM": "causal-language",
    "GraniteForCausalLM": "causal-language",
    "GraniteMoeForCausalLM": "causal-language",
    "PersimmonForCausalLM": "causal-language",
    "MiniCPMForCausalLM": "causal-language",
    "InternLM2ForCausalLM": "causal-language",
    "DeepseekV2ForCausalLM": "causal-language",
    "DeepseekV3ForCausalLM": "causal-language",
    "ExaoneForCausalLM": "causal-language",
    "NemotronForCausalLM": "causal-language",
    "Glm4ForCausalLM": "causal-language",
    "GlmForCausalLM": "causal-language",
    "SmolLM3ForCausalLM": "causal-language",
    "HeliumForCausalLM": "causal-language",
    "ZambaForCausalLM": "causal-language",
    # --- transformer.h / wte ---------------------------------------------
    "GPT2LMHeadModel": "gpt2-transformer",
    "GPTJForCausalLM": "gpt2-transformer",
    "CodeGenForCausalLM": "gpt2-transformer",
    "GPTBigCodeForCausalLM": "gpt2-transformer",
    # --- transformer.blocks / wte ----------------------------------------
    "MptForCausalLM": "mpt-transformer",
    # --- transformer.h / word_embeddings ---------------------------------
    "BloomForCausalLM": "bloom-transformer",
    "FalconForCausalLM": "bloom-transformer",
    # --- gpt_neox.layers / embed_in --------------------------------------
    "GPTNeoXForCausalLM": "neox",
    "StableLMEpochForCausalLM": "neox",
}

#: Families with real Afterimage end-to-end evidence in ``results/``.
#: Everything else that resolves structurally is "expected": the engine can
#: address its weights, but no measured correctness run exists yet.
VERIFIED_ARCHITECTURES = frozenset({
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Phi3ForCausalLM",
})


def classify_config(config: dict[str, Any] | Any) -> dict[str, Any]:
    """Return an honest metadata classification without loading weights."""

    if isinstance(config, dict):
        architectures = [str(value) for value in config.get("architectures", [])]
        model_type = str(config.get("model_type", ""))
    else:
        architectures = [str(value) for value in getattr(config, "architectures", []) or []]
        model_type = str(getattr(config, "model_type", ""))
    joined = " ".join([model_type, *architectures]).lower()
    vision = any(token in joined for token in ("vision", "vl", "image"))
    moe = "moe" in joined or "mixtral" in joined
    architecture_set = set(architectures)
    known = {name: ARCHITECTURE_LAYOUTS[name]
             for name in architecture_set if name in ARCHITECTURE_LAYOUTS}

    if vision:
        execution = "experimental"
        reason = "Vision-language execution requires the multimodal adapter path."
    elif moe:
        execution = "experimental"
        reason = "The layout is expected, but this MoE family needs end-to-end validation."
    elif architecture_set & VERIFIED_ARCHITECTURES:
        execution = "verified"
        reason = "This architecture family has Afterimage end-to-end evidence."
    elif known:
        execution = "expected"
        layout = sorted(set(known.values()))[0]
        reason = (
            "The decoder structure matches Afterimage's %s adapter, but this "
            "family is not release-verified." % layout
        )
    elif not architectures:
        execution = "unknown"
        reason = "The Hub result did not include enough architecture metadata."
    else:
        execution = "download-only"
        reason = (
            "No executable Afterimage adapter is known for this layout yet. "
            "Afterimage streams decoder stacks it can address by name; this "
            "architecture is not in the known-layout table. It can still be "
            "downloaded and inspected, and `afterimage doctor` on a local "
            "snapshot reports what the structural resolver actually finds."
        )
    return {
        "architectures": architectures,
        "model_type": model_type or None,
        "modality": "vision-text" if vision else "text",
        "mixture_of_experts": moe,
        "execution": execution,
        "execution_reason": reason,
        "layout": sorted(set(known.values()))[0] if known else None,
    }


def architecture_names(configs: Iterable[dict[str, Any]]) -> set[str]:
    return {
        name
        for config in configs
        for name in classify_config(config)["architectures"]
    }
