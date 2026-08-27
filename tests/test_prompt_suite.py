import pytest

from afterimage.bench.prompt_suite import (
    PAPER_GENERATION_CASES,
    PROMPT_CASES,
    prompt_cases,
    render_chat_prompt,
)


def test_suite_has_disjoint_diverse_splits():
    evaluation = prompt_cases("evaluation")
    calibration = prompt_cases("calibration")
    assert len(evaluation) == 4
    assert len(calibration) == 8
    assert {case.id for case in evaluation}.isdisjoint(case.id for case in calibration)
    assert len({case.semantic_bucket for case in evaluation}) == len(evaluation)
    assert len({case.id for case in PROMPT_CASES}) == len(PROMPT_CASES)


def test_paper_generation_split_has_four_disjoint_cases():
    """paper-generation-v1: the long-form workload for 100-128-token
    decode-TPS/speculation measurement, disjoint from the short factual
    paper-short-v1 cases (split="evaluation")."""
    generation = prompt_cases("paper_generation")
    assert len(generation) == 4
    assert {case.id for case in generation}.isdisjoint(
        case.id for case in prompt_cases("evaluation"))
    assert len({case.semantic_bucket for case in generation}) == len(generation)


def test_paper_generation_cases_elicit_long_form_answers_not_short_facts():
    """The whole reason this split exists: forcing a one-token-answer
    prompt on to 128 tokens produces a strange speculative-decoding
    workload. Each prompt here must actually ask for a substantial answer."""
    for case in PAPER_GENERATION_CASES:
        assert case.expected_any == (), (
            f"{case.id} has expected_any set; this split is throughput-only, "
            "not correctness-scored")
        assert len(case.user_text) > 80, f"{case.id} reads like a short-answer prompt"


def test_prompt_cases_all_includes_every_split():
    every_id = {case.id for case in prompt_cases("all")}
    assert every_id == (
        {case.id for case in PROMPT_CASES} | {case.id for case in PAPER_GENERATION_CASES})


def test_prompt_cases_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="split must be"):
        prompt_cases("not-a-real-split")


def test_expected_answer_matching_is_case_and_whitespace_insensitive():
    fact = prompt_cases("evaluation")[0]
    code = prompt_cases("evaluation")[2]
    assert fact.matches("  au\n")
    assert code.matches("x  **   2")
    assert not fact.matches("Ag")


class ThinkingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return messages[-1]["content"]


class LegacyTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[-1]["content"]


def test_render_disables_thinking_and_supports_legacy_templates():
    case = prompt_cases("evaluation")[0]
    assert "chemical symbol" in render_chat_prompt(ThinkingTokenizer(), case)
    assert "chemical symbol" in render_chat_prompt(LegacyTokenizer(), case)
