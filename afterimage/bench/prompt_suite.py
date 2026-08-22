"""Small, deterministic prompt suites for bounded hardware comparisons.

The runtime methods change execution, not model semantics.  A useful quick
benchmark therefore needs two checks at once: diverse enough prompts to expose
workload-dependent speculative acceptance, and simple answers that can be
audited without turning a systems run into a multi-hour quality evaluation.
"""
from __future__ import annotations

import dataclasses
import re


@dataclasses.dataclass(frozen=True)
class PromptCase:
    id: str
    semantic_bucket: str
    split: str
    user_text: str
    expected_any: tuple[str, ...]

    def matches(self, generated_text: str) -> bool:
        text = _normalise(generated_text)
        return any(_normalise(expected) in text for expected in self.expected_any)


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


PROMPT_SUITE_VERSION = "bounded-chat-v2"

# Evaluation and calibration are deliberately disjoint.  The calibration
# prompts are used only to build a critical-path profile and a frozen
# rejection-hazard state; reporting uses the four evaluation cases.
PROMPT_CASES = (
    PromptCase(
        "fact-gold", "factual", "evaluation",
        "Answer with only the chemical symbol for gold.",
        ("Au",),
    ),
    PromptCase(
        "arithmetic-17x6", "arithmetic", "evaluation",
        "Compute 17 multiplied by 6. Answer with only the integer.",
        ("102",),
    ),
    PromptCase(
        "code-square", "code", "evaluation",
        "Fill the blank and output only the missing Python expression: "
        "def square(x): return ___",
        ("x * x", "x*x", "x ** 2", "x**2"),
    ),
    PromptCase(
        "retrieval-7319", "long_context_retrieval", "evaluation",
        "A maintenance note lists several identifiers. The amber cabinet is "
        "2048, the cobalt cabinet is 5501, and the Orion access code is 7319. "
        "A later paragraph discusses inventory counts of 18, 42, and 96; none "
        "of those are access codes. What is the Orion access code? Answer with "
        "only the four digits.",
        ("7319",),
    ),
    PromptCase(
        "summary-photosynthesis", "summarisation", "calibration",
        "Plants use light energy to convert carbon dioxide and water into "
        "sugars, releasing oxygen. Name this process with one word.",
        ("photosynthesis",),
    ),
    PromptCase(
        "logic-glippets", "logic", "calibration",
        "Every glippet is blue. No blue object is transparent. Can a glippet "
        "be transparent? Answer only yes or no.",
        ("no",),
    ),
    PromptCase(
        "copy-nonce", "exact_copy", "calibration",
        "Repeat exactly this identifier and nothing else: K7M-42Q-Z9.",
        ("K7M-42Q-Z9",),
    ),
    PromptCase(
        "json-status", "structured_output", "calibration",
        "Output one minified JSON object with status equal to ok and count "
        "equal to 3. Do not add prose.",
        ('{"status":"ok","count":3}',),
    ),
    PromptCase(
        "code-squares", "code", "calibration",
        "Fill the blank and output only the Python expression: "
        "squares = ___  # square every x in values",
        ("[x * x for x in values]", "[x*x for x in values]"),
    ),
    PromptCase(
        "logic-ravens", "counterfactual_logic", "calibration",
        "All ravens in a sanctuary are black. Mira is a white bird in that "
        "sanctuary. Must Mira be a raven? Answer only yes or no.",
        ("no",),
    ),
    PromptCase(
        "retrieval-tulip", "long_context_retrieval", "calibration",
        "The delta ledger assigns 1182 to maple, 9407 to tulip, and 6630 to "
        "cedar. A later audit lists invoice totals 94, 207, and 811, none of "
        "which are ledger identifiers. Return only tulip's four-digit code.",
        ("9407",),
    ),
    PromptCase(
        "transform-sequence", "symbolic_transformation", "calibration",
        "Reverse the order of these comma-separated tokens without changing "
        "their spelling. Output tokens only: amber,cobalt,ivory",
        ("ivory,cobalt,amber",),
    ),
)


def prompt_cases(split: str = "evaluation") -> tuple[PromptCase, ...]:
    if split not in {"evaluation", "calibration", "all"}:
        raise ValueError("split must be evaluation, calibration, or all")
    if split == "all":
        return PROMPT_CASES
    return tuple(case for case in PROMPT_CASES if case.split == split)


def render_chat_prompt(tokenizer, case: PromptCase) -> str:
    """Render one prompt consistently across Afterimage and AirLLM.

    Qwen3 accepts ``enable_thinking=False``.  Older compatible tokenizers do
    not, so the fallback preserves the same messages while omitting only that
    optional template switch.
    """
    messages = [
        {"role": "system", "content": "Follow the requested output format exactly."},
        {"role": "user", "content": case.user_text},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)
