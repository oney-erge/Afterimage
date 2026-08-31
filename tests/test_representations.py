from afterimage.runtime.representations import (
    RepresentationOption, _prune_dominated, plan_representations,
    validate_artifacts,
)


def test_multi_choice_plan_selects_fast_options_that_fit():
    options = [
        RepresentationOption("a", "disk", prepare_s=5, storage_bytes=10),
        RepresentationOption("a", "ram", ram_bytes=10, prepare_s=1, storage_bytes=10),
        RepresentationOption("b", "disk", prepare_s=5, storage_bytes=10),
        RepresentationOption("b", "ram", ram_bytes=10, prepare_s=1, storage_bytes=10),
    ]
    plan = plan_representations(options, vram_budget_bytes=0, ram_budget_bytes=10,
                                quantum_bytes=10)
    assert plan.feasible
    assert plan.predicted_prepare_s == 6
    assert sum(option.name == "ram" for option in plan.choices.values()) == 1


def test_approximate_options_are_never_selected():
    plan = plan_representations([
        RepresentationOption("a", "lossy", prepare_s=0, exact=False),
        RepresentationOption("a", "exact", prepare_s=2),
    ], vram_budget_bytes=0, ram_budget_bytes=0)
    assert plan.choices["a"].name == "exact"


def test_missing_artifacts_are_reported(tmp_path):
    option = RepresentationOption("a", "indexed", artifact="a.idx")
    plan = plan_representations([option], vram_budget_bytes=0, ram_budget_bytes=0)
    assert validate_artifacts(plan, tmp_path) == ["a.idx"]


def test_representation_plan_round_trip(tmp_path):
    option = RepresentationOption("a", "disk", prepare_s=1.5)
    plan = plan_representations([option], vram_budget_bytes=0,
                                ram_budget_bytes=0)
    path = tmp_path / "plan.json"
    plan.save(path)
    loaded = type(plan).load(path)
    assert loaded == plan


def test_representation_plan_reserves_headroom_and_reports_exact_bytes():
    plan = plan_representations([
        RepresentationOption("a", "disk", prepare_s=4),
        RepresentationOption("a", "vram", vram_bytes=3, prepare_s=0),
        RepresentationOption("b", "disk", prepare_s=4),
        RepresentationOption("b", "vram", vram_bytes=3, prepare_s=0),
    ], vram_budget_bytes=10, ram_budget_bytes=0,
        vram_headroom_bytes=4, quantum_bytes=4)

    # Each 3-byte resident option consumes one conservative 4-byte DP unit;
    # the returned runtime accounting is nevertheless the real 3 bytes.
    assert plan.vram_bytes == 3
    assert plan.vram_headroom_bytes == 4
    assert plan.vram_bytes + plan.vram_headroom_bytes <= plan.vram_budget_bytes
    assert sum(option.name == "vram" for option in plan.choices.values()) == 1


def test_schema_one_representation_plan_remains_loadable(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"choices":{"a":{"tensor_key":"a","name":"disk",'
        '"vram_bytes":0,"ram_bytes":0,"storage_bytes":0,"prepare_s":1.0,'
        '"exact":true,"artifact":null}},"vram_bytes":0,"ram_bytes":0,'
        '"storage_bytes":0,"predicted_prepare_s":1.0,"feasible":true,'
        '"reason":"","schema_version":1}', encoding="utf-8")

    loaded = type(plan_representations([
        RepresentationOption("a", "disk")
    ], vram_budget_bytes=0, ram_budget_bytes=0)).load(path)
    assert loaded.schema_version == 1
    assert loaded.vram_budget_bytes is None
    assert loaded.vram_headroom_bytes == 0


def test_equal_storage_fast_prune_matches_brute_force_dominance():
    states = {}
    for vram in range(9):
        for ram in range(11):
            # Deterministic, non-monotone costs exercise memory/cost tradeoffs.
            cost = float(((vram * 17 + ram * 29) % 31) + (vram + ram) / 100)
            states[(vram, ram)] = (cost, 1234, (vram, ram))

    expected = {}
    for state, value in states.items():
        dominated = any(
            other_state != state
            and other_state[0] <= state[0]
            and other_state[1] <= state[1]
            and other[0] <= value[0]
            and other[1] <= value[1]
            for other_state, other in states.items()
        )
        if not dominated:
            expected[state] = value

    assert _prune_dominated(states) == expected
