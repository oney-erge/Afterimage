#!/usr/bin/env python3
"""Recompute the Llama-3.3-70B / RTX 5090 headline numbers from the frozen
artifacts in this folder and check them against the numbers printed in the
arXiv draft (Oney Erge, "Schedule-Aware Exact Weight Placement for Large
Language Models with Limited GPU Memory", draft frozen 2026-09-26).

Every artifact this script reads is copied verbatim from the private
Paper 1 evidence bundle; nothing here recomputes or edits a measurement.
`MANIFEST.sha256` in this directory records the exact bytes, and those
hashes match the sidecar `*.status.json.validity.json` files and the
frozen-plan hashes recorded in the protocols under docs/h65/protocols/. It also
checks each published protocol against the SHA-256 its run recorded.

Run:
    python evidence/h65-paper1/verify_llama_confirmations.py

Exits 0 and prints "ALL CHECKS PASSED" only if every recomputed number is
within its stated tolerance of the paper's printed value.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
LLAMA = HERE / "llama-rtx5090"
PROTOCOLS = HERE.parents[1] / "docs" / "h65" / "protocols"

FAILURES: list[str] = []


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest() -> None:
    """Every artifact in llama-rtx5090/ must match MANIFEST.sha256, and every
    artifact there must be listed -- an unlisted file is unverified evidence."""
    print("=== MANIFEST.sha256 ===")
    listed = {}
    for line in (HERE / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            listed[name.lstrip("*")] = digest
    present = {p.relative_to(LLAMA).as_posix()
               for p in LLAMA.glob("*/*.json")}
    bad = [name for name, digest in listed.items()
           if not (LLAMA / name).exists() or _sha256(LLAMA / name) != digest]
    unlisted = sorted(present - set(listed))
    print("  %d listed, %d mismatched, %d unlisted" % (len(listed), len(bad), len(unlisted)))
    for name in bad:
        FAILURES.append("MANIFEST mismatch: %s" % name)
    for name in unlisted:
        FAILURES.append("artifact not in MANIFEST.sha256: %s" % name)


def check_frozen_inputs() -> None:
    """Every run artifact recorded the SHA-256 of the exact protocol document
    (and, for D2, the prompt file) it ran under, before any measurement. The
    published copies in docs/h65/protocols/ must be those exact bytes, or the
    'frozen before measurement' claim cannot be checked by a reader.
    """
    print("\n=== Frozen protocols and inputs vs. the hashes each run recorded ===")
    artifacts = {
        "D1": LLAMA / "D1-confirmation-1" / "status.json",
        "D2": LLAMA / "D2-confirmation-2" / "status.json",
        "D4": LLAMA / "D4-greedy-mechanism" / "status.json",
        "4-token": LLAMA / "four-token-ablation" / "status.json",
        "2026-08-31": LLAMA / "method-history" / "2026-08-31-8block-before-head-seed.json",
        "2026-09-02": LLAMA / "method-history" / "2026-09-02-12pair-before-head-seed.json",
    }
    for label, path in artifacts.items():
        status = load(path)
        pinned = []
        if status.get("protocol_sha256"):
            pinned.append((status["protocol"], status["protocol_sha256"]))
        if status.get("confirmatory_protocol_sha256"):
            pinned.append((status["confirmatory_protocol"],
                           status["confirmatory_protocol_sha256"]))
        for amendment in status.get("protocol_amendments") or []:
            pinned.append((amendment["path"], amendment["sha256"]))
        for recorded_path, recorded_sha in pinned:
            name = pathlib.PurePosixPath(str(recorded_path)).name
            local = PROTOCOLS / name
            ok = local.exists() and _sha256(local) == recorded_sha
            print("  [%s] %s: %s" % ("OK  " if ok else "FAIL", label, name))
            if not ok:
                FAILURES.append("%s protocol %s does not match its recorded "
                                "SHA-256 %s" % (label, name, recorded_sha))
    d2 = load(artifacts["D2"])
    prompts = LLAMA / "D2-confirmation-2" / "h65-confirmation2-prompts-20260911.json"
    recorded = [sha for key, sha in (d2.get("immutable_input_sha256") or {}).items()
                if key.endswith(prompts.name)]
    ok = bool(recorded) and _sha256(prompts) == recorded[0]
    print("  [%s] D2: %s" % ("OK  " if ok else "FAIL", prompts.name))
    if not ok:
        FAILURES.append("D2 prompt file does not match its recorded SHA-256")


def paired_log_ratio_ci(ratios: list[float]) -> tuple[float, float, float]:
    """Geometric mean and two-sided 95% Student-t interval of paired ratios,
    matching the paper's Eq. (10)-(11): mean and spread of block log ratios.
    """
    logs = [math.log(r) for r in ratios]
    mean_log = statistics.fmean(logs)
    n = len(logs)
    if n < 2:
        return math.exp(mean_log), float("nan"), float("nan")
    sd_log = statistics.stdev(logs)
    # Two-sided 95% t critical values by degrees of freedom (df = n-1),
    # for the small sample sizes used here (n-1 in 2..19).
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093,
    }
    t = t_table[n - 1]
    half = t * sd_log / math.sqrt(n)
    return math.exp(mean_log), math.exp(mean_log - half), math.exp(mean_log + half)


def check(label: str, got: float, want: float, tol: float) -> None:
    ok = abs(got - want) <= tol
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label}: recomputed {got:.4f}, paper {want:.4f} (tol {tol})")
    if not ok:
        FAILURES.append(f"{label}: recomputed {got!r} vs paper {want!r} (tol {tol})")


def geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(v) for v in values))


def run_confirmation(name: str, path: pathlib.Path, paper_geomean: float,
                      paper_ci: tuple[float, float]) -> dict:
    print(f"\n=== {name} ===")
    status = load(path)
    pairs = status["pairs"]
    control_s = [p["traffic_seconds"] for p in pairs]
    h65_s = [p["candidate_seconds"] for p in pairs]
    ratios = [c / h for c, h in zip(control_s, h65_s)]

    check(f"{name}: n pairs", len(pairs), len(pairs), 0)
    check(f"{name}: recorded geometric_mean_speedup vs pairs",
          geomean(ratios), status["geometric_mean_speedup"], 1e-6)
    check(f"{name}: geometric speedup vs paper Table 7",
          geomean(ratios), paper_geomean, 0.002)

    gm, lo, hi = paired_log_ratio_ci(ratios)
    check(f"{name}: 95% CI lower vs paper Table 7", lo, paper_ci[0], 0.01)
    check(f"{name}: 95% CI upper vs paper Table 7", hi, paper_ci[1], 0.01)

    control_vram = [p["traffic_whole_process_peak_vram_gb"] for p in pairs]
    h65_vram = [p["candidate_whole_process_peak_vram_gb"] for p in pairs]
    vram_ratio = geomean([c / h for c, h in zip(control_vram, h65_vram)])
    print(f"  peak-VRAM geometric ratio (control/H6.5): {vram_ratio:.4f}"
          f"  (max control {max(control_vram):.3f} GB,"
          f" max H6.5 {max(h65_vram):.3f} GB)")

    control_geo = geomean(control_s)
    h65_geo = geomean(h65_s)
    print(f"  request-time geometric means: control {control_geo:.2f}s,"
          f" H6.5 {h65_geo:.2f}s")

    return {
        "pairs": len(pairs), "speedup": gm, "ci": (lo, hi),
        "control_s": control_s, "h65_s": h65_s,
    }


def run_pooled(d1: dict, d2: dict, paper_geomean: float,
               paper_ci: tuple[float, float]) -> None:
    print("\n=== Pooled (D1 + D2, 20 pairs) ===")
    ratios = ([c / h for c, h in zip(d1["control_s"], d1["h65_s"])]
              + [c / h for c, h in zip(d2["control_s"], d2["h65_s"])])
    check("Pooled: n pairs", len(ratios), 20, 0)
    check("Pooled: geometric speedup vs paper", geomean(ratios), paper_geomean, 0.002)
    gm, lo, hi = paired_log_ratio_ci(ratios)
    check("Pooled: 95% CI lower vs paper", lo, paper_ci[0], 0.01)
    check("Pooled: 95% CI upper vs paper", hi, paper_ci[1], 0.01)


def run_d4_greedy_mechanism() -> None:
    print("\n=== D4: schedule-aware placement vs. isolated-cost greedy rule ===")
    status = load(LLAMA / "D4-greedy-mechanism" / "status.json")
    contrasts = status["contrasts"]
    # Paper Table 8 / text: H6.5 vs. greedy ("knapsack" arm here) is about
    # 16% lower latency and 20% lower peak incremental GPU memory.
    latency = contrasts["endpoint_vs_knapsack_latency"]["geometric_mean"]
    vram = contrasts["endpoint_vs_knapsack_vram"]["geometric_mean"]
    check("D4: H6.5-vs-greedy latency reduction",
          1 - 1 / latency, 0.16, 0.01)
    check("D4: H6.5-vs-greedy peak-VRAM reduction",
          1 - 1 / vram, 0.20, 0.01)

    rows = status["results_by_block"]
    for arm, paper_seconds, paper_peak in (
        ("traffic", 47.90, 10.257),
        ("knapsack", 46.31, 9.868),
        ("h65-endpoint", 38.72, 7.937),
    ):
        seconds = geomean([r["seconds"][arm] for r in rows])
        peak = max(r["peak_vram_gb"][arm] for r in rows)
        check(f"D4: {arm} request latency vs paper Table 8", seconds, paper_seconds, 0.5)
        check(f"D4: {arm} peak incremental GPU vs paper Table 8", peak, paper_peak, 0.05)


def run_four_token_ablation() -> None:
    print("\n=== Secondary: four-token representation ablation ===")
    status = load(LLAMA / "four-token-ablation" / "status.json")
    speedups = status["geometric_speedups"]
    # Bundle README's "Benefit at four tokens": 1.041x, placement-only vs.
    # traffic (the primary secondary contrast declared in the frozen
    # protocol, docs/h65/protocols/PAPER1_5090_LLAMA_H65_4TOKEN_ABLATION.md).
    check("4-token: placement-only vs traffic",
          speedups["placement_speedup_vs_traffic"], 1.041, 0.002)
    print("  full_speedup_vs_traffic:", speedups["full_speedup_vs_traffic"])
    print("  full_speedup_vs_placement:", speedups["full_speedup_vs_placement"])
    print("  NOTE: this script checks the mechanism-report figure carried in")
    print("  paper1-5090-bundle-20260911/README.md. It does not check a specific")
    print("  arXiv appendix table -- confirm the appendix table number by hand")
    print("  before citing this contrast in a table.")


def run_method_history_confirmation(name: str, path: pathlib.Path,
                                     paper_geomean: float,
                                     paper_ci: tuple[float, float]) -> None:
    """Recompute an earlier (pre-head-seed) confirmation, not cited by the
    2026-09-26 arXiv draft. These are disclosed in docs/h65/README.md's method-history
    section: both completed and passed their own frozen protocol's gates on an
    earlier plan, built before the 2026-09-09 candidate-retention fix
    (commit cb9e320) and the 2026-09-10 output-head search seed (commit
    1c6a02c). Kept here, not deleted, per this evidence folder's own policy of
    keeping invalid/superseded runs visible.
    """
    print(f"\n=== {name} (method history, not in the arXiv draft) ===")
    status = load(path)
    methods = status["live_comparison"]["methods"]
    control_s = methods["traffic-placement"]["block_median_seconds_per_token"]
    h65_s = methods["h65-placement-only"]["block_median_seconds_per_token"]
    ratios = [c / h for c, h in zip(control_s, h65_s)]
    check(f"{name}: n blocks", len(ratios), len(ratios), 0)
    gm, lo, hi = paired_log_ratio_ci(ratios)
    check(f"{name}: geometric speedup", gm, paper_geomean, 0.002)
    check(f"{name}: 95% CI lower", lo, paper_ci[0], 0.01)
    check(f"{name}: 95% CI upper", hi, paper_ci[1], 0.01)
    print("  peak VRAM (median whole-cell): traffic %.2f GB, H6.5 %.2f GB" % (
        methods["traffic-placement"]["median_whole_cell_peak_vram_gb"],
        methods["h65-placement-only"]["median_whole_cell_peak_vram_gb"]))


def main() -> int:
    check_manifest()
    check_frozen_inputs()
    d1 = run_confirmation(
        "D1: Llama confirmation 1 (n=8)",
        LLAMA / "D1-confirmation-1" / "status.json",
        paper_geomean=1.018, paper_ci=(0.884, 1.173))
    d2 = run_confirmation(
        "D2: Llama confirmation 2 (n=12)",
        LLAMA / "D2-confirmation-2" / "status.json",
        paper_geomean=1.239, paper_ci=(1.128, 1.361))
    run_pooled(d1, d2, paper_geomean=1.145, paper_ci=(1.052, 1.247))
    run_d4_greedy_mechanism()
    run_four_token_ablation()
    run_method_history_confirmation(
        "2026-08-31, 8 blocks, before the output-head seed",
        LLAMA / "method-history" / "2026-08-31-8block-before-head-seed.json",
        paper_geomean=1.066, paper_ci=(0.949, 1.198))
    run_method_history_confirmation(
        "2026-09-02, 12 pairs, before the output-head seed",
        LLAMA / "method-history" / "2026-09-02-12pair-before-head-seed.json",
        paper_geomean=1.052, paper_ci=(1.000, 1.107))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
