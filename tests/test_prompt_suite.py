from afterimage.bench.prompt_suite import (
    PROMPT_CASES,
    prompt_cases,
    render_chat_prompt,
)


def test_suite_has_disjoint_diverse_splits():
    evaluation = prompt_cases("evaluation")
    calibration = prompt_cases("calibration")
    assert len(evaluation) == 4
    assert len(calibration) == 2
    assert {case.id for case in evaluation}.isdisjoint(case.id for case in calibration)
    assert len({case.semantic_bucket for case in evaluation}) == len(evaluation)
    assert len({case.id for case in PROMPT_CASES}) == len(PROMPT_CASES)


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
