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

# split="paper_generation" is what the paper plan calls "paper-generation-v1":
# four realistic ~120-180-word-eliciting prompts (explanation, summarization,
# code generation, analytical/structured response), used for the 100-128
# token decode-TPS/speculation workload -- not the short factual-answer
# "paper-short-v1" cases above (split="evaluation"), which are for the
# TTFT/short-cold-start workload. Forcing the short factual prompts on to
# 128 tokens produces a strange speculative-decoding workload: the model
# gives its one-token answer almost immediately, then has nothing
# substantive left to generate for the remaining ~120 tokens. These four
# instead have a real ~120-180-word answer to give, so decode-length
# measurements reflect steady-state generation, not padding after an
# already-finished answer.
#
# expected_any is deliberately empty for every case in this split: there is
# no single short string that correctly scores a free-form ~150-word
# answer, and that is not what this suite is for -- it measures throughput
# at a realistic generation length, not answer correctness. Read
# expected_match/expected_match_rate as not applicable for this split, not
# as "the model got it wrong".
PAPER_GENERATION_CASES = (
    PromptCase(
        "explain-bicycle-balance", "explanation", "paper_generation",
        "In about 150 words, explain how a moving bicycle stays upright "
        "without a rider actively balancing it. Cover both the steering "
        "geometry (trail) and gyroscopic effects, and note which one modern "
        "research considers more important.",
        (),
    ),
    PromptCase(
        "summarize-coral-bleaching", "summarization", "paper_generation",
        "Coral bleaching happens when unusually warm water causes corals to "
        "expel the symbiotic algae, called zooxanthellae, living in their "
        "tissues. These algae provide corals with most of their energy "
        "through photosynthesis and give them their vivid color, so a coral "
        "that loses them turns white and starves unless the algae return "
        "within a few weeks. Mass bleaching events have become more frequent "
        "as ocean temperatures have risen, with the Great Barrier Reef "
        "experiencing severe bleaching in 2016, 2017, 2020, and 2022. "
        "Bleached coral is not dead, but it is significantly more vulnerable "
        "to disease and has reduced reproductive capacity, and repeated "
        "bleaching events give reefs less time to recover between episodes. "
        "In about 150 words, summarize what coral bleaching is, what causes "
        "it, and why repeated events are especially damaging.",
        (),
    ),
    PromptCase(
        "code-binary-search", "code", "paper_generation",
        "Write a complete Python function `binary_search(items, target)` "
        "that performs binary search on a sorted list and returns the index "
        "of `target`, or -1 if it is not present. Include a short docstring "
        "and handle an empty list correctly.",
        (),
    ),
    PromptCase(
        "analyze-hashmap-vs-btree", "analytical", "paper_generation",
        "In a short structured comparison, analyze the trade-offs between a "
        "hash table and a balanced binary search tree as the backing "
        "structure for an in-memory key-value store. Cover average-case "
        "lookup time, memory overhead, and whether ordered iteration over "
        "keys is supported.",
        (),
    ),
)

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
    if split not in {"evaluation", "calibration", "all", "paper_generation"}:
        raise ValueError(
            "split must be evaluation, calibration, paper_generation, or all")
    if split == "paper_generation":
        return PAPER_GENERATION_CASES
    if split == "all":
        return PROMPT_CASES + PAPER_GENERATION_CASES
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
