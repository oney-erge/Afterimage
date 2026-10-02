"""The README's benchmark table is hand-copied into the CLI help, the profile
table in docs/CONFIGURATION.md, and afterimage/reference.py. That copying has
drifted twice already: when AirLLM's baseline moved from 3.1.0 to 3.2.0, the
'fast' ratio stayed at 3.15x (now 2.93x) and the min-memory ratio at 0.89x (now
0.83x) in places nobody re-read. These checks tie every copy back to the one
table, and the table's own ratios back to its own seconds column.

Historical records (docs/RESULTS_LOG.md, docs/CROSS_MODEL_BENCHMARK_*.md) are
deliberately not checked: they quote old baselines on purpose.
"""
import pathlib
import re

from afterimage import cli
from afterimage.reference import MEASURED_REFERENCE
from scripts.make_results_chart import build_svg
from scripts.make_results_chart import read_results_table as _readme_table

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _row(table, prefix):
    matches = [v for k, v in table.items() if k.startswith(prefix)]
    assert len(matches) == 1, "expected one README row starting %r" % prefix
    return matches[0]


def test_readme_vs_airllm_ratios_recompute_from_its_own_seconds():
    table = _readme_table()
    airllm_seconds = _row(table, "AirLLM")[1]
    for name, (_vram, seconds, ratio) in table.items():
        assert round(airllm_seconds / seconds, 2) == ratio, (
            "%s: README says %.2fx, its own seconds give %.2fx"
            % (name, ratio, airllm_seconds / seconds))


def test_reference_module_matches_the_readme_table():
    table = _readme_table()
    fast_vram, fast_s, _ = _row(table, "Afterimage + fixed speculation")
    min_vram, min_s, _ = _row(table, "Afterimage exact minimum-memory")
    params = MEASURED_REFERENCE["params_b"]
    assert round(MEASURED_REFERENCE["fast_s_per_token_per_b"] * params, 3) == fast_s
    assert MEASURED_REFERENCE["fast_vram_floor_gb"] == fast_vram
    assert round(MEASURED_REFERENCE["min_memory_s_per_token_per_b"] * params, 3) == min_s
    assert MEASURED_REFERENCE["min_memory_vram_gb"] == min_vram


def test_profile_ratios_in_cli_help_and_configuration_match_the_readme():
    table = _readme_table()
    expected = {
        "min-memory": _row(table, "Afterimage exact minimum-memory")[2],
        "balanced": _row(table, "Afterimage exact + 4 GB residency")[2],
        "fast": _row(table, "Afterimage + fixed speculation")[2],
    }
    config = (ROOT / "docs" / "CONFIGURATION.md").read_text(encoding="utf-8")
    for profile, ratio in expected.items():
        row = re.search(r"^\| `%s` \|.*$" % re.escape(profile), config, re.M)
        assert row, "CONFIGURATION.md has no %s profile row" % profile
        assert "%.2fx" % ratio in row.group(0), (
            "CONFIGURATION.md's %s row should say %.2fx: %s"
            % (profile, ratio, row.group(0)))

    cli_source = pathlib.Path(cli.__file__).read_text(encoding="utf-8")
    assert "balanced = +4GB residency, %.2fx" % expected["balanced"] in cli_source
    assert "%.2fx. Explicit flags below still override it." % expected["fast"] in cli_source
    assert "largest lossless speedup (%.2fx measured)" % expected["fast"] in cli_source


def test_superseded_ratios_do_not_reappear_in_current_claims():
    superseded = ("3.15x",)  # fixed speculation vs AirLLM 3.1.0; now 2.93x
    current = [ROOT / "README.md", ROOT / "afterimage" / "cli.py"] + [
        ROOT / "docs" / name for name in (
            "USAGE.md", "FAQ.md", "CONFIGURATION.md", "TROUBLESHOOTING.md",
            "HOW_IT_WORKS.md", "ALL_HYPOTHESES_AND_BASELINES.md", "ARCHITECTURE.md")]
    for path in current:
        text = path.read_text(encoding="utf-8")
        for value in superseded:
            assert value not in text, "%s still quotes superseded %s" % (
                path.relative_to(ROOT), value)


def test_results_chart_matches_the_readme_table():
    committed = (ROOT / "docs" / "assets" / "results-chart.svg").read_text(encoding="utf-8")
    regenerated = build_svg(_readme_table())
    assert committed == regenerated, (
        "docs/assets/results-chart.svg is stale -- run "
        "python scripts/make_results_chart.py")
