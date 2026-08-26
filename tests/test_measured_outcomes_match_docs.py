"""Enforce the promise in MeasuredOutcome's docstring: the registry in
experiments.py is "single-sourced from docs/RESULTS_LOG.md and
docs/ALL_HYPOTHESES_AND_BASELINES.md so the UI's Lab cards and the written
record can never silently drift apart." Nothing previously checked that; H1
and H4 drifted from the controlling doc's refreshed numbers without any test
noticing. This parses the controlling table and compares it against the
registry directly.
"""
import pathlib
import re

import pytest

from afterimage.experiments import HYPOTHESES

DOC = (pathlib.Path(__file__).resolve().parent.parent
       / "docs" / "ALL_HYPOTHESES_AND_BASELINES.md")

ROW_RE = re.compile(r"^\|\s*\*\*\d+\*\*\s*\|\s*\*\*(H\d+)\*\*\s*\|(.*)\|\s*$")
PCT_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)%")

# Hypotheses where the registry's effect_pct intentionally differs from (or
# omits) the doc table's bolded percentage, and why. Anything not listed
# here must match the doc within tolerance.
KNOWN_DIVERGENCES = {
    "h3-contextual-bandit": "doc reports 0.00% deployed reward parity; "
        "registry omits effect_pct because the reportable number is oracle "
        "fraction (97.50%), not a speed ratio",
    "h7-xor-reference": "doc's -2.24% is the forced-XOR mechanism's raw "
        "storage effect before the safe chooser intervenes; registry's 0.0 "
        "is the deployed safe-chooser's realized effect (it fell back to "
        "independent storage) -- a different and equally real number, not "
        "a drift",
    "h11-neural-utility-spec": "doc explicitly labels its +9.5% apparent "
        "timing as noise from zero action divergence; registry correctly "
        "omits effect_pct rather than report a number it says not to trust",
}


def _doc_effects() -> dict[str, float]:
    effects = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        short_id, rest = match.groups()
        cols = [c.strip() for c in rest.split("|")]
        if len(cols) < 6:
            continue
        effect_col = cols[5]
        pct = PCT_RE.search(effect_col)
        if pct:
            effects.setdefault(short_id, float(pct.group(1)))
    return effects


def test_controlling_doc_table_is_parseable():
    doc_effects = _doc_effects()
    assert len(doc_effects) >= 15, (
        "table row parser found suspiciously few rows -- "
        "ALL_HYPOTHESES_AND_BASELINES.md's table format may have changed")


def test_registry_effect_matches_controlling_doc_table():
    doc_effects = _doc_effects()
    checked = 0
    for hyp_id, hyp in HYPOTHESES.items():
        short_id = hyp_id.split("-", 1)[0].upper()
        if short_id not in doc_effects or hyp_id in KNOWN_DIVERGENCES:
            continue
        registry_effect = hyp.measured.effect_pct if hyp.measured else None
        if registry_effect is None:
            continue
        assert registry_effect == pytest.approx(doc_effects[short_id], abs=0.05), (
            f"{hyp_id}: registry effect_pct={registry_effect} but "
            f"{DOC.name} reports {doc_effects[short_id]}%. Either the "
            f"registry drifted from the controlling doc, or this is an "
            f"intentional divergence that belongs in KNOWN_DIVERGENCES "
            f"with a stated reason.")
        checked += 1
    assert checked >= 12, f"only verified {checked} hypotheses against the doc"
