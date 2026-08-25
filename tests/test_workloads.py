import time

from afterimage.probe.workloads import (
    FOCUSED_CODE,
    LONG_FORM_PROSE,
    MULTI_TURN_CHAT,
    topic_switch_prompts,
)



import pytest

pytestmark = pytest.mark.archive  # Phase-0 subspace-activation-cache branch, killed per its own gate
def test_topic_switch_prompts_terminates_quickly():
    """The direct regression test for the infinite-loop bug: an earlier
    version of topic_switch_prompts() could spin forever once the two
    smaller pools were exhausted. This asserts real wall-clock termination,
    not just a correct return value, since a hang would never reach the
    assertions below at all."""
    t0 = time.perf_counter()
    result = topic_switch_prompts()
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"took {elapsed:.2f}s -- should be near-instant"
    assert isinstance(result, list)


def test_topic_switch_prompts_contains_every_source_item_exactly_once():
    result = topic_switch_prompts()
    expected_total = len(FOCUSED_CODE) + len(MULTI_TURN_CHAT) + len(LONG_FORM_PROSE)
    assert len(result) == expected_total

    all_sources = FOCUSED_CODE + MULTI_TURN_CHAT + LONG_FORM_PROSE
    assert sorted(result) == sorted(all_sources)


def test_topic_switch_prompts_actually_interleaves():
    """Not just correct membership -- genuinely switches topic, i.e. the
    output isn't just all of one pool followed by all of another."""
    result = topic_switch_prompts(switch_every=3)
    focused_positions = [i for i, x in enumerate(result) if x in FOCUSED_CODE]
    # if it were one contiguous block, positions would be a single run;
    # interleaving means the span is much wider than the block size alone
    assert max(focused_positions) - min(focused_positions) > len(FOCUSED_CODE)


def test_topic_switch_prompts_handles_uneven_switch_every():
    for k in [1, 2, 3, 5, 100]:
        t0 = time.perf_counter()
        result = topic_switch_prompts(switch_every=k)
        assert time.perf_counter() - t0 < 1.0
        assert len(result) == len(FOCUSED_CODE) + len(MULTI_TURN_CHAT) + len(LONG_FORM_PROSE)
