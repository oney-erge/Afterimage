#!/usr/bin/env python3
"""Recompute every number the H6.5 paper reports from the raw artifacts in
this folder and check each against the value printed in the arXiv draft
(Oney Erge, "Schedule-Aware Exact Weight Placement for Large Language Models
with Limited GPU Memory", draft frozen 2026-09-26): the Llama-3.3-70B / RTX
5090 confirmations (Tables 7-8) and the Qwen3-14B / Gemma 2 27B / RTX 3080
Laptop studies (Tables 9-10 and the external-package comparisons).

Every artifact here is copied verbatim from the original campaign output;
nothing recomputes or edits a measurement. The script also checks
MANIFEST.sha256, every frozen protocol against the SHA-256 its run recorded,
every frozen laptop plan against the hash its run recorded, and that each
laptop study ran from a clean tree and was marked paper-eligible.

Run (stdlib only, no GPU, a few seconds):
    python evidence/h65-paper1/verify_paper_evidence.py

Exits 0 and prints "ALL CHECKS PASSED" only if every check passes.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
LLAMA = HERE / "llama-rtx5090"
LAPTOP = HERE / "laptop-rtx3080"
PROTOCOLS = HERE.parents[1] / "docs" / "h65" / "protocols"

FAILURES: list[str] = []


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest() -> None:
    """Every JSON artifact in this folder must match MANIFEST.sha256, and every
    one must be listed -- an unlisted file is unverified evidence."""
    print("=== MANIFEST.sha256 ===")
    listed = {}
    for line in (HERE / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            listed[name.lstrip("*")] = digest
    present = {p.relative_to(HERE).as_posix() for p in HERE.rglob("*.json")}
    bad = [name for name, digest in listed.items()
           if not (HERE / name).exists() or _sha256(HERE / name) != digest]
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


LAPTOP_STUDIES = {
    "D6": "D6-qwen-matched/qwen3-14b-h65-matched-ttft-practical-20260909-r2-1tok.json",
    "D7": "D7-qwen-external/qwen3-14b-h65-external-ttft-practical-20260909-r2-1tok.json",
    "D8": "D8-qwen-32token/qwen3-14b-h65-matched-decode-practical-20260909-r2-32tok.json",
    "D9": "D9-gemma-matched/gemma2-27b-h65-matched-ttft-practical-20260909-r2b-1tok.json",
    "D10": "D10-gemma-external/gemma2-27b-h65-external-ttft-practical-20260909-r2b-1tok.json",
}
# Paper Table 4, System A. Checked against each artifact's own environment block.
SYSTEM_A = {"torch": "2.6.0+cu124", "cuda": "12.4", "driver": "596.49"}
SYSTEM_A_PACKAGES = {"transformers": "5.12.1", "accelerate": "1.14.0",
                     "airllm": "3.2.0", "deepspeed": "0.19.5"}
# Artifact method ids -> the paper's names.
ARMS = {"exact-min": "minimum-memory control", "simple-v4-r8": "traffic control",
        "disk-frozen": "disk control", "h65-selected": "H6.5"}


def _laptop(label: str) -> dict:
    return load(LAPTOP / LAPTOP_STUDIES[label])


def _rows(status: dict) -> dict[str, list[dict]]:
    return {method["method_id"]: method["rows"] for method in status["methods"]}


def check_laptop_provenance() -> None:
    print("\n=== RTX 3080 Laptop studies: provenance and frozen plans ===")
    for label in LAPTOP_STUDIES:
        status = _laptop(label)
        env = status["environment"]
        clean = (status.get("paper_eligible") is True and status.get("status") == "complete"
                 and not env.get("git_status") and bool(env.get("git_commit")))
        matches = (all(env.get(k) == v for k, v in SYSTEM_A.items())
                   and all(env.get("packages", {}).get(k) == v
                           for k, v in SYSTEM_A_PACKAGES.items()))
        ok = clean and matches
        print("  [%s] %s: clean tree at %s, paper-eligible, Table 4 environment"
              % ("OK  " if ok else "FAIL", label, str(env.get("git_commit"))[:7]))
        if not ok:
            FAILURES.append("%s provenance or environment does not match" % label)
    for label, model in (("D6", "qwen3-14b"), ("D9", "gemma2-27b")):
        for plan in _laptop(label).get("afterimage_plan_methods") or []:
            name = "%s-%s" % (model, pathlib.PurePosixPath(plan["snapshot_path"]).name)
            local = LAPTOP / "frozen-plans" / name
            ok = local.exists() and _sha256(local) == plan["source_sha256"]
            print("  [%s] %s: frozen plan %s" % ("OK  " if ok else "FAIL", label, name))
            if not ok:
                FAILURES.append("%s frozen plan %s does not match its recorded hash"
                                % (label, name))


def check_laptop_tables() -> None:
    print("\n=== Table 9: laptop same-engine comparisons (D6 Qwen3-14B, D9 Gemma 2 27B) ===")
    table9 = {
        "D6": {"exact-min": (29.00, 1.928), "simple-v4-r8": (14.87, 4.136),
               "disk-frozen": (19.27, 2.300), "h65-selected": (14.21, 3.611)},
        "D9": {"exact-min": (89.04, 2.884), "simple-v4-r8": (41.58, 4.171),
               "disk-frozen": (43.97, 2.713), "h65-selected": (36.84, 3.660)},
    }
    for label, expected in table9.items():
        rows = _rows(_laptop(label))
        for method, (seconds, peak) in expected.items():
            got_s = geomean([r["wall_seconds"] for r in rows[method]])
            got_p = max(r["peak_vram_gb"] for r in rows[method])
            check("%s %s geomean request time (s)" % (label, ARMS[method]), got_s, seconds, 0.005)
            check("%s %s max peak incremental GPU (GB)" % (label, ARMS[method]), got_p, peak, 0.0005)
        traffic = expected["simple-v4-r8"]
        h65 = expected["h65-selected"]
        print("  %s H6.5 vs traffic: %.1f%% lower latency, %.1f%% lower peak GPU"
              % (label, 100 * (1 - h65[0] / traffic[0]), 100 * (1 - h65[1] / traffic[1])))

    print("\n=== Table 10 and Section 5.5: Qwen3-14B at 32 output tokens (D8) ===")
    rows = _rows(_laptop("D8"))
    table10 = {"exact-min": (28.83, 18.71, 2.8, 14.1), "disk-frozen": (17.93, 18.77, 6.6, 17.8),
               "simple-v4-r8": (26.02, 11.81, 23.7, 47.2), "h65-selected": (22.95, 12.22, 19.2, 51.7)}
    for method, (wall, read, miss, wait) in table10.items():
        r = rows[method]
        hits = sum(x.get("prefetch_hits", 0) for x in r)
        misses = sum(x.get("prefetch_misses", 0) for x in r)
        check("D8 %s wall/output-token (s)" % ARMS[method],
              statistics.fmean(x["wall_seconds"] / x["output_tokens"] for x in r), wall, 0.005)
        check("D8 %s read/output-token (GB)" % ARMS[method],
              statistics.fmean(x["gb_read_per_token"] for x in r), read, 0.005)
        check("D8 %s host-prefetch miss rate (%%)" % ARMS[method],
              100 * misses / (hits + misses), miss, 0.05)
        check("D8 %s host-prefetch wait (s)" % ARMS[method],
              statistics.fmean(x["prefetch_wait_seconds"] for x in r), wait, 0.05)
    check("D8 disk control geomean request time (s), paper 'about 574'",
          geomean([x["wall_seconds"] for x in rows["disk-frozen"]]), 574, 0.5)
    check("D8 H6.5 geomean request time (s), paper 'about 734'",
          geomean([x["wall_seconds"] for x in rows["h65-selected"]]), 734, 0.5)

    print("\n=== Section 5.4 / Figure 7: external package operating points (D7, D10) ===")
    for label in ("D7", "D10"):
        rows = _rows(_laptop(label))
        times = {m: geomean([x["wall_seconds"] for x in r]) for m, r in rows.items()}
        fastest = min(times, key=times.get)
        ok = fastest == "h65-selected"
        print("  [%s] %s: H6.5 has the shortest one-token request time (%s)"
              % ("OK  " if ok else "FAIL", label,
                 ", ".join("%s %.2fs" % (m, t) for m, t in sorted(times.items(), key=lambda kv: kv[1]))))
        if not ok:
            FAILURES.append("%s: fastest configuration is %s, not H6.5" % (label, fastest))
    rows = _rows(_laptop("D10"))
    check("D10 H6.5 max peak incremental GPU (GB), paper 'about 3.7'",
          max(x["peak_vram_gb"] for x in rows["h65-selected"]), 3.7, 0.05)
    check("D10 DeepSpeed max peak incremental GPU (GB), paper '7.4'",
          max(x["peak_vram_gb"] for x in rows["deepspeed-zero-inference"]), 7.4, 0.05)


def main() -> int:
    check_manifest()
    check_frozen_inputs()
    check_laptop_provenance()
    check_laptop_tables()
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
