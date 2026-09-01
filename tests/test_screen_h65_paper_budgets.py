import importlib.util
import pathlib
import types

import pytest


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts/screen_h65_paper_budgets.py"
SPEC = importlib.util.spec_from_file_location("screen_h65_paper_budgets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_budgets_preserves_independent_surface():
    assert MODULE.parse_budgets("3:6,3:8,4:6,4:8") == (
        (3.0, 6.0), (3.0, 8.0), (4.0, 6.0), (4.0, 8.0))


@pytest.mark.parametrize("value", ("", "3", "0:6", "3:-1", "3:6,3:6"))
def test_parse_budgets_rejects_invalid_or_duplicate_points(value):
    with pytest.raises(ValueError):
        MODULE.parse_budgets(value)


def test_plan_fingerprint_changes_only_when_physical_choices_change():
    option = types.SimpleNamespace(
        name="compressed_disk", vram_bytes=0, ram_bytes=0,
        storage_bytes=10, artifact="tensor.bin")
    same = types.SimpleNamespace(**vars(option))
    changed = types.SimpleNamespace(**{**vars(option), "name": "decoded_ram"})
    plan_a = types.SimpleNamespace(choices={"tensor": option})
    plan_b = types.SimpleNamespace(choices={"tensor": same})
    plan_c = types.SimpleNamespace(choices={"tensor": changed})
    assert MODULE.plan_fingerprint(plan_a) == MODULE.plan_fingerprint(plan_b)
    assert MODULE.plan_fingerprint(plan_a) != MODULE.plan_fingerprint(plan_c)


def test_compare_plans_reports_transition_and_predicted_gain():
    disk = types.SimpleNamespace(
        name="compressed_disk", vram_bytes=0, ram_bytes=0,
        storage_bytes=10, artifact="tensor.bin")
    ram = types.SimpleNamespace(
        name="compressed_ram", vram_bytes=0, ram_bytes=10,
        storage_bytes=10, artifact="tensor.bin")
    control = types.SimpleNamespace(
        choices={"tensor": disk}, predicted_prepare_s=10.0)
    candidate = types.SimpleNamespace(
        choices={"tensor": ram}, predicted_prepare_s=9.0)
    delta = MODULE.compare_plans(control, candidate)
    assert delta == {
        "changed_tensors": 1,
        "changed_proxy_bytes": 10,
        "transitions": {"compressed_disk->compressed_ram": 1},
        "predicted_latency_reduction": pytest.approx(0.1),
    }
