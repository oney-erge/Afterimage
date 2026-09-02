#!/usr/bin/env python3
"""Create deny-by-default validity sidecars for Paper 1 result artifacts.

Original measurements are immutable.  This script writes a small
``.validity.json`` next to every reviewed campaign result plus a central JSON
ledger and a human-readable Markdown index.  Consumers should require an
explicit usable status instead of treating every completed-looking JSON file
as paper evidence.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib


SCHEMA_VERSION = 1
PRIMARY_USABLE = {"confirmatory"}
PILOT_USABLE = PRIMARY_USABLE | {"regulated_pilot"}


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def artifact_paths(root: pathlib.Path) -> list[pathlib.Path]:
    paths = []
    for path in root.glob("*.json*"):
        if (path.is_file() and not path.name.endswith(".validity.json")
                and path.name not in {"VALIDITY_LEDGER.json"}):
            paths.append(path)
    for directory in root.glob("e6.3-*"):
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json*"):
            if path.is_file() and not path.name.endswith(".validity.json"):
                paths.append(path)
    return sorted(set(paths))


def result_status(payload: dict) -> str | None:
    value = payload.get("status")
    return str(value) if value is not None else None


def classify(relative: str, payload: dict | None,
             parse_error: str | None) -> tuple[str, list[str], str | None]:
    name = relative.replace("\\", "/")
    basename = pathlib.PurePosixPath(name).name
    if parse_error:
        return "invalid", ["unparseable_json", parse_error], None
    assert payload is not None
    if basename.endswith(".watch.json"):
        return "diagnostic_only", [
            "watcher health/ETA state, not a measurement artifact",
        ], None
    if basename.endswith(".partial") or result_status(payload) != "complete":
        return "invalid", [
            "incomplete_or_interrupted_artifact",
            "partial and failed runs must never enter figures or tables",
        ], None

    if basename.startswith((
            "h65-paper-matrix-", "h65-frozen-confirm-", "h65-direct-pair-")):
        if (payload.get("kind") in {
                    "h65_causal_paper_matrix",
                    "h65_frozen_plan_confirmatory_matrix",
                    "h65_direct_pair_confirmatory",
                }
                and payload.get("confirmatory_protocol_satisfied")
                and (payload.get("gates") or {}).get(
                    "confirmatory_execution_eligible")):
            return "confirmatory", [
                "frozen independent protocol executed completely",
                "all recorded causal-matrix scientific gates passed",
            ], None
        if (payload.get("kind") == "h65_causal_paper_matrix"
                and (payload.get("gates") or {}).get("paper_pilot_eligible")):
            return "regulated_pilot", [
                "bounded causal matrix passed its recorded scientific gates",
                "four blocks are for effect-size/power planning, not confirmatory power",
            ], None
        return "invalid", [
            "new H6.5 matrix did not pass paper_pilot_eligible",
        ], None

    if "v2-counterbalanced" in basename and basename.startswith(
            "h65-causal-placement-only-"):
        return "regulated_pilot", [
            "causal runtime and AB/BA method order",
            "only one held-out prompt, one token, and two blocks",
            "usable as supporting pilot evidence only",
        ], "h65-paper-matrix-*"

    if basename.startswith("h65-causal-") or basename.startswith(
            "h65-causal-placement-only-"):
        return "superseded", [
            "causal implementation but evaluation order was not fully counterbalanced",
            "superseded by the counterbalanced gate and multi-prompt matrix",
        ], "*v2-counterbalanced.json or h65-paper-matrix-*"

    pre_causal_prefixes = (
        "h65-quick-",
        "h65-feature-ladder-",
        "h65-qwen3-",
    )
    if basename.startswith(pre_causal_prefixes):
        return "invalid", [
            "predates the corrected causal prefetch/forward lifecycle",
            "numbers are retained for debugging history, not paper evidence",
        ], "h65-paper-matrix-*"

    diagnostic_prefixes = (
        "h65-bugdiag-",
        "h65-high-vram-offline-screen-",
        "h65-small-slow-offline-screen-",
        "h65-tight-budget-grid-",
        "legacy-h6-vs-traffic-",
    )
    if basename.startswith(diagnostic_prefixes):
        return "diagnostic_only", [
            "debugging/offline screen or legacy-H6 comparison",
            "not designed as a paper result",
        ], None

    if name.startswith("e6.3-"):
        failures = payload.get("failures") or []
        if failures:
            return "incomplete_comparison", [
                "%d requested comparison cells failed" % len(failures),
                "do not compute a complete cross-framework table from this artifact",
            ], None
        return "exploratory_only", [
            "artifact declares exploratory/mechanism-screen evidence",
            "predates the new H6.5 causal matrix and is not confirmatory",
        ], None

    legacy_campaign_prefixes = (
        "d3-capacity-demo-",
        "e2.2-h6-budget-sweep-",
        "e3.2-h6-plan-robustness-",
        "e4.1-compression-ablation-",
        "e8.2-breakdown-",
    )
    if basename.startswith(legacy_campaign_prefixes):
        return "exploratory_only", [
            "artifact declares L1 exploratory/mechanism-screen evidence",
            "legacy H6 evidence must not be relabeled as corrected H6.5 evidence",
        ], None

    return "unreviewed", [
        "no explicit validity rule matched; deny by default",
    ], None


def markdown(ledger: dict) -> str:
    counts = collections.Counter(entry["classification"] for entry in ledger["artifacts"])
    lines = [
        "# Paper 1 result validity",
        "",
        "This index is deny-by-default. Original result JSON files remain immutable; "
        "the adjacent `.validity.json` files and this ledger determine allowed use.",
        "",
        "Only artifacts explicitly classified `confirmatory` are authorized for a "
        "primary paper claim. `regulated_pilot` means effect-size/supporting "
        "evidence only.",
        "",
        "## Counts",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append("- `%s`: %d" % (status, count))
    lines.extend([
        "",
        "## Artifact decisions",
        "",
        "| Artifact | Classification | Primary claim | Reason |",
        "|---|---|---:|---|",
    ])
    for entry in ledger["artifacts"]:
        reason = "; ".join(entry["reasons"]).replace("|", "\\|")
        lines.append("| `%s` | `%s` | %s | %s |" % (
            entry["path"], entry["classification"],
            "yes" if entry["allowed_for_primary_paper_claim"] else "no",
            reason))
    lines.extend([
        "",
        "## Consumer rule",
        "",
        "Figures and tables must reject artifacts without a sidecar, artifacts whose "
        "SHA-256 no longer matches the sidecar, and every classification except "
        "`confirmatory`. A draft may include `regulated_pilot` only when it is "
        "visibly labeled pilot/exploratory.",
        "",
    ])
    return "\n".join(lines)


def mark(root: pathlib.Path) -> dict:
    marked_at = dt.datetime.now(dt.timezone.utc).isoformat()
    entries = []
    for path in artifact_paths(root):
        relative = str(path.relative_to(root))
        payload = None
        parse_error = None
        try:
            payload = load(path)
        except Exception as exc:
            parse_error = repr(exc)
        classification, reasons, superseded_by = classify(
            relative, payload, parse_error)
        entry = {
            "path": relative,
            "sha256": sha256(path),
            "classification": classification,
            "allowed_for_primary_paper_claim": classification in PRIMARY_USABLE,
            "allowed_for_pilot_or_exploratory_use": classification in PILOT_USABLE,
            "reasons": reasons,
            "superseded_by": superseded_by,
            "source_status": result_status(payload or {}),
            "source_kind": (payload or {}).get("kind"),
            "source_evidence_level": (payload or {}).get("evidence_level"),
        }
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "marked_at_utc": marked_at,
            **entry,
        }
        write_json(path.with_name(path.name + ".validity.json"), sidecar)
        entries.append(entry)

    quarantine = root.parent.parent / "quarantine_llama_miscalibrated_sweep_20260829"
    if quarantine.exists():
        entries.append({
            "path": str(quarantine),
            "sha256": None,
            "classification": "invalid",
            "allowed_for_primary_paper_claim": False,
            "allowed_for_pilot_or_exploratory_use": False,
            "reasons": [
                "explicitly quarantined miscalibrated Llama sweep",
                "directory contents must not be consumed by paper tooling",
            ],
            "superseded_by": "h65-paper-matrix-*",
            "source_status": None,
            "source_kind": None,
            "source_evidence_level": None,
        })

    ledger = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": marked_at,
        "policy": "deny_by_default",
        "primary_usable_classifications": sorted(PRIMARY_USABLE),
        "pilot_usable_classifications": sorted(PILOT_USABLE),
        "artifacts": entries,
    }
    write_json(root / "VALIDITY_LEDGER.json", ledger)
    (root / "VALIDITY_LEDGER.md").write_text(markdown(ledger), encoding="utf-8")
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--check")
    parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    if not root.is_dir():
        parser.error("result root does not exist: %s" % root)
    ledger = mark(root)
    counts = collections.Counter(
        entry["classification"] for entry in ledger["artifacts"])
    print("marked %d artifacts: %s" % (
        len(ledger["artifacts"]),
        ", ".join("%s=%d" % item for item in sorted(counts.items()))))
    if args.check:
        requested = pathlib.Path(args.check).resolve()
        try:
            relative = str(requested.relative_to(root))
        except ValueError:
            parser.error("--check must be under --root")
        entry = next(
            (item for item in ledger["artifacts"] if item["path"] == relative), None)
        allowed = PILOT_USABLE if args.allow_pilot else PRIMARY_USABLE
        if not entry or entry["classification"] not in allowed:
            print("REJECTED %s: %s" % (
                relative, entry["classification"] if entry else "unlisted"))
            return 2
        print("ACCEPTED %s: %s" % (relative, entry["classification"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
