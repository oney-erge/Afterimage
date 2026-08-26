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
    ``gpt_neox`` for NeoX). It is used only as a literal string prefix for
    building weight keys (see ``.prefix``); ``resolve_model_adapter`` walks
    the actual object graph by hand for each layout family, since the depth
    and attribute names genuinely differ (a single `getattr` for most
    families, two hops for the nested vision layout, a different two hops
    again for Llama 4). ``layers_attr`` and ``embed_attr`` are the attribute
    names *inside* that container.

    ``head_owner_attr``, when non-empty, is the attribute on the top-level
    model that owns ``lm_head`` -- empty means the top-level model itself
    does, which is true for every family except Llama 4's conditional-
    generation wrapper, where the output head lives on the inner
    ``language_model`` (itself a complete CausalLM), not on the outer
    vision+text wrapper.
    """

    layout: str
    container_attr: str
    layers_attr: str
    embed_attr: str
    modality: str = "text"
    supports_certified_head: bool = True
    head_owner_attr: str = ""

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

#: Llama 4's conditional-generation wrapper is a third, distinct shape, not a
#: parametrization of VISION_LAYOUT: `language_model` sits directly on the
#: outer model (not nested under `model.language_model`) and is itself a
#: complete Llama4ForCausalLM, with its own `.model.layers`/`.model.
#: embed_tokens` and its own `.lm_head` -- the output head is on
#: language_model, not on the outer model. Verified against a real (meta-
#: device) Llama4Config build, not assumed from the family name.
LLAMA4_VISION_LAYOUT = LayoutSpec(
    "llama4-vision-language", "language_model.model", "layers", "embed_tokens",
    modality="vision-text", supports_certified_head=False,
    head_owner_attr="language_model")


class ModelAdapter:
    """Access the language decoder without hard-coding one object path."""

    def __init__(self, model: Any, *, spec: LayoutSpec, language_model: Any):
        self.model = model
        self.spec = spec
        self.language_model = language_model
        self._head_owner = (
            getattr(model, spec.head_owner_attr) if spec.head_owner_attr else model)
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
        return self._head_owner.lm_head

    @property
    def language_config(self):
        config = self.model.config
        return getattr(config, "text_config", config)

    @property
    def vision_model(self):
        # Llama 4 carries its vision tower as `vision_model` directly on the
        # outer model; the earlier Qwen-VL-style layout nests it as
        # `model.visual`. Checked in that order so a family exposing neither
        # (the common, non-vision case) cleanly falls through to None.
        direct = getattr(self.model, "vision_model", None)
        if direct is not None:
            return direct
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

    # Llama 4's `language_model` sits directly on the outer model and is
    # itself a complete CausalLM (its own `.model.layers`/`.model.
    # embed_tokens`, its own `.lm_head`) -- checked before the nested
    # vision layout below, which looks one level too shallow for this shape
    # and would not match it.
    direct_language = getattr(model, "language_model", None)
    direct_inner = getattr(direct_language, "model", None)
    if (_matches(direct_inner, LLAMA4_VISION_LAYOUT)
            and hasattr(direct_language, "lm_head")
            and hasattr(model, "vision_model")):
        return ModelAdapter(model, spec=LLAMA4_VISION_LAYOUT, language_model=direct_inner)

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
    # Added 2026-08-26 after checking current (post-2025) model families
    # against real Hub configs, not name-guessing: fetched each repo's
    # config.json directly and, where the class was available in the
    # installed transformers, built a real (meta-device) model and
    # confirmed it resolves through resolve_model_adapter with a verified
    # real parameter-name match, exactly like the original nine.
    "DeepseekV32ForCausalLM": "causal-language",  # DeepSeek-V3.2 (MoE)
    "Glm4MoeForCausalLM": "causal-language",  # GLM-4.5/4.6 (MoE)
    "Qwen3NextForCausalLM": "causal-language",  # Qwen3-Next (hybrid linear attn)
    "GptOssForCausalLM": "causal-language",  # OpenAI gpt-oss (MoE)
    "JambaForCausalLM": "causal-language",  # AI21 Jamba (hybrid Mamba+attention)
    "GraniteMoeHybridForCausalLM": "causal-language",  # IBM Granite 4 (hybrid Mamba+attention)
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
    # --- language_model.model.layers / embed_tokens, head on language_model
    "Llama4ForConditionalGeneration": "llama4-vision-language",
}

#: Known Hub architectures that require `trust_remote_code=True` to load at
#: all (their config.json has an `auto_map` pointing at custom modeling
#: code on the repo itself). afterimage/server/acquisition.py's
#: inspect_snapshot() deliberately calls AutoConfig.from_pretrained with
#: trust_remote_code=False, so these can never resolve through Afterimage's
#: normal pipeline regardless of anything in this table -- listed here so
#: that fact is written down once, not so any code currently reads it.
#: Confirmed 2026-08-26 by checking each repo's real config.json for an
#: `auto_map` entry, not assumed: MiniMax (MiniMaxM1ForCausalLM/
#: MiniMaxText01ForCausalLM), Baichuan (BaichuanForCausalLM), Ant Group's
#: Ling/Bailing (BailingMoeForCausalLM), and Apple's OpenELM
#: (OpenELMForCausalLM) all require it as of the checked snapshots.
REQUIRES_TRUST_REMOTE_CODE = frozenset({
    "MiniMaxM1ForCausalLM",
    "MiniMaxText01ForCausalLM",
    "BaichuanForCausalLM",
    "BailingMoeForCausalLM",
    "OpenELMForCausalLM",
})

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

#: Architectures that are Mixture-of-Experts even though their class name
#: does not contain the substring "moe" (DeepSeek's and GLM's naming does
#: not follow that convention), so the cheap substring check in
#: classify_config() would otherwise miss them and under-classify a real
#: sparse MoE model as fully "expected" rather than "experimental".
MOE_ARCHITECTURES_WITHOUT_MOE_IN_NAME = frozenset({
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "Glm4MoeForCausalLM",
    "GptOssForCausalLM",
})


def classify_config(config: dict[str, Any] | Any) -> dict[str, Any]:
    """Return an honest metadata classification without loading weights."""

    if isinstance(config, dict):
        architectures = [str(value) for value in config.get("architectures", [])]
        model_type = str(config.get("model_type", ""))
        has_vision_config = "vision_config" in config
    else:
        architectures = [str(value) for value in getattr(config, "architectures", []) or []]
        model_type = str(getattr(config, "model_type", ""))
        has_vision_config = getattr(config, "vision_config", None) is not None
    joined = " ".join([model_type, *architectures]).lower()
    # Substring matching on the family name catches most cases (Qwen3-VL,
    # anything with "vision" in its model_type) but missed a real one:
    # Llama4ForConditionalGeneration / model_type "llama4" contains neither
    # "vision", "vl", nor "image". A text+vision config almost universally
    # carries a `vision_config` sub-config -- checking for it directly is a
    # structural signal, not a name guess, and catches that case.
    vision = has_vision_config or any(
        token in joined for token in ("vision", "vl", "image"))
    architecture_set = set(architectures)
    # "moe" as a substring catches most families by convention (Qwen3Moe,
    # GraniteMoe, ...) but DeepSeek and GLM's MoE releases do not follow it
    # (DeepseekV3ForCausalLM, Glm4MoeForCausalLM's own model_type
    # "glm4_moe" does contain it, but the class name check runs on
    # `joined`, which also includes model_type, so this line still needs
    # the explicit set for the DeepSeek family specifically) -- the
    # explicit set is the precise signal, the substring check is the
    # cheap first pass.
    moe = ("moe" in joined or "mixtral" in joined
           or bool(architecture_set & MOE_ARCHITECTURES_WITHOUT_MOE_IN_NAME))
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
