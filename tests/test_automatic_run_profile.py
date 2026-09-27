"""automatic_run_profile picks --profile fast's speculation for --auto, whose
draft model (RUN_PROFILES["fast"]["draft_model"]) is a fixed Qwen/Qwen3-0.6B.
That only shares a vocabulary with a Qwen3 target -- for anything else,
--auto must downgrade to balanced rather than recommend a combination
generate_adaptive's vocabulary guard now refuses at runtime.
"""
from afterimage.cli import automatic_run_profile


def test_auto_picks_fast_for_a_qwen3_target_with_enough_vram():
    profile, reason = automatic_run_profile(8.0, model_id="Qwen/Qwen3-14B")
    assert profile == "fast"
    assert "8.0" in reason


def test_auto_downgrades_to_balanced_for_a_non_qwen3_target():
    profile, reason = automatic_run_profile(8.0, model_id="meta-llama/Llama-3.3-70B-Instruct")
    assert profile == "balanced"
    assert "Qwen3" in reason


def test_auto_without_a_model_id_keeps_the_old_unconditional_behaviour():
    # Callers that don't pass model_id (there should be none left, but the
    # default keeps old behaviour rather than silently downgrading everyone).
    profile, _reason = automatic_run_profile(8.0)
    assert profile == "fast"


def test_auto_still_respects_the_vram_tiers_regardless_of_model():
    assert automatic_run_profile(None, model_id="meta-llama/Llama-3.3-70B-Instruct")[0] == "min-memory"
    assert automatic_run_profile(4.0, model_id="meta-llama/Llama-3.3-70B-Instruct")[0] == "balanced"
    assert automatic_run_profile(1.0, model_id="meta-llama/Llama-3.3-70B-Instruct")[0] == "min-memory"
