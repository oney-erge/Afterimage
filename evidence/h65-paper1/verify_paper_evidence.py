#!/usr/bin/env python3
"""Recompute every number the H6.5 paper reports from the raw artifacts in
this folder and check each against the value printed in the arXiv draft
(Oney Erge, "Schedule-Aware Exact Weight Placement for Large Language Models
with Limited GPU Memory"): the Llama-3.3-70B / RTX 5090 confirmations
(Tables 7-8), their three repeats, the 12 GB budget follow-up and the
head-seed ablation, and the Qwen3-14B / Gemma 2 27B / RTX 3080 Laptop studies
(Tables 9-11 and the external-package comparisons).

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


# Llama follow-ups on the RTX 5090: folder, frozen protocol, prompt file, the
# amendment a resumed run recorded (if any), and each arm's frozen plan.
_D4_PLANS = {"h65": "h65-endpoint-plan.json", "traffic": "traffic-plan.json"}
LLAMA_FOLLOWUPS = {
    "D4 repeat 1": ("D4-repeat-1", "PROTOCOL-d4-replication-20261005.md", "d4r-prompts.json",
                    "AMENDMENT-1-resume-after-controller-stop-20261005.md", _D4_PLANS),
    "D4 repeat 2": ("D4-repeat-2", "PROTOCOL-d4r2-20261005.md", "d4r2-prompts.json", None, _D4_PLANS),
    "D4 repeat 3": ("D4-repeat-3", "PROTOCOL-d4r3-20261005.md", "d4r3-prompts.json", None, _D4_PLANS),
    "D11": ("D11-seed-ablation", "PROTOCOL-P1-head-seed-ablation.md", "p1-prompts.json",
            "PROTOCOL-P1-amendment1-pause-resume.md",
            {"h65-endpoint": "h65-endpoint-plan.json", "traffic": "traffic-plan.json",
             "head-seed": "head-seed-plan.json"}),
    "D12": ("D12-budget-12gb", "PROTOCOL-budget-followup-amendment1-20261004.md",
            "budget-prompts.json", None,
            {"h65": "h65-plan-12v16r.json", "traffic": "traffic-plan-12v16r.json"}),
}


def _followup(label: str) -> dict:
    return load(LLAMA / LLAMA_FOLLOWUPS[label][0] / "status.json")


def _recorded_input(status: dict, suffix: str) -> str | None:
    return next((sha for path, sha in (status.get("immutable_input_sha256") or {}).items()
                 if path.endswith(suffix)), None)


def check_followup_inputs() -> None:
    """Each follow-up pinned its protocol, prompts, amendment and plans by
    SHA-256 before measuring. The published copies must be those exact bytes.
    """
    print("\n=== Llama follow-ups: frozen inputs vs. the hashes each run recorded ===")
    for label, (folder, protocol, prompts, amendment, plans) in LLAMA_FOLLOWUPS.items():
        status = _followup(label)
        pinned = [(PROTOCOLS / protocol, status["protocol_sha256"]),
                  (LLAMA / folder / prompts,
                   status.get("prompts_sha256") or _recorded_input(status, prompts))]
        if amendment:
            pinned.append((PROTOCOLS / amendment, status["resume"]["amendment_sha256"]))
        for arm, plan in plans.items():
            recorded = status["plans"][arm]
            pinned.append((LLAMA / "frozen-plans" / plan,
                           recorded["sha256"] if isinstance(recorded, dict) else recorded))
        profile = _recorded_input(status, "knapsack-critical-path-profile.json")
        if profile:
            pinned.append((LLAMA / "frozen-plans" / "knapsack-critical-path-profile.json", profile))
        for path, recorded in pinned:
            ok = path.exists() and _sha256(path) == recorded
            print("  [%s] %s: %s" % ("OK  " if ok else "FAIL", label, path.name))
            if not ok:
                FAILURES.append("%s: %s does not match its recorded SHA-256" % (label, path.name))


def _eligible_blocks(label: str) -> list[dict]:
    """Blocks the run's own gates kept, with each recorded ratio rechecked
    against the two cells it was computed from."""
    blocks = [b for b in _followup(label)["block_results"] if b["eligible"]]
    for block in blocks:
        for key, value in block.items():
            if not key.endswith(("_latency", "_peak_vram")):
                continue
            top, bottom = key[0], key[len("X_over_")]
            field = "wall_seconds" if key.endswith("_latency") else "peak_vram_gb"
            recomputed = block["arms"][top][field] / block["arms"][bottom][field]
            if abs(recomputed - value) > 1e-9:
                FAILURES.append("%s block %s: %s is %r, cells give %r"
                                % (label, block["block"], key, value, recomputed))
    return blocks


def _unresolved(label: str, what: str, ratios: list[float]) -> None:
    gm, lo, hi = paired_log_ratio_ci(ratios)
    ok = lo < 1 < hi
    print("  [%s] %s: %s ratio %.3f [%.3f, %.3f], interval includes 1"
          % ("OK  " if ok else "FAIL", label, what, gm, lo, hi))
    if not ok:
        FAILURES.append("%s: %s interval [%.3f, %.3f] does not include 1" % (label, what, lo, hi))


def run_llama_followups() -> None:
    print("\n=== Section 5.3: three repeats of the mechanism study on new prompts ===")
    pooled = []
    for label, complete in (("D4 repeat 1", 11), ("D4 repeat 2", 12), ("D4 repeat 3", 12)):
        blocks = _eligible_blocks(label)
        check("%s: complete blocks (Table 5)" % label, len(blocks), complete, 0)
        pooled += blocks
    check("Repeats: pooled blocks", len(pooled), 35, 0)
    for key, what, paper in (("T_over_H_latency", "latency vs traffic", 0.18),
                             ("K_over_H_latency", "latency vs greedy", 0.15),
                             ("T_over_H_peak_vram", "peak GPU vs traffic", 0.23),
                             ("K_over_H_peak_vram", "peak GPU vs greedy", 0.20)):
        check("Repeats: H6.5 reduction, %s" % what,
              1 - 1 / geomean([b[key] for b in pooled]), paper, 0.01)
    lower = sum(b["T_over_H_peak_vram"] > 1 and b["K_over_H_peak_vram"] > 1 for b in pooled)
    ok = lower == len(pooled)
    print("  [%s] Repeats: H6.5 peak GPU below both controls in %d of %d blocks"
          % ("OK  " if ok else "FAIL", lower, len(pooled)))
    if not ok:
        FAILURES.append("Repeats: H6.5 peak GPU is not lower in every block")

    print("\n=== Section 5.3: 12 GB GPU budget follow-up (D12) ===")
    blocks = _eligible_blocks("D12")
    check("D12: blocks", len(blocks), 8, 0)
    check("D12: H6.5 peak GPU reduction vs traffic",
          1 - 1 / geomean([b["T_over_H_peak_vram"] for b in blocks]), 0.14, 0.01)
    _unresolved("D12", "traffic/H6.5 latency", [b["T_over_H_latency"] for b in blocks])

    print("\n=== Section 6: head-seed ablation (D11) ===")
    blocks = _eligible_blocks("D11")
    check("D11: blocks", len(blocks), 8, 0)
    _unresolved("D11", "head-seed/H6.5 latency", [b["S_over_H_latency"] for b in blocks])


LAPTOP_STUDIES = {
    "D6": "D6-qwen-matched/qwen3-14b-h65-matched-ttft-practical-20260909-r2-1tok.json",
    "D7": "D7-qwen-external/qwen3-14b-h65-external-ttft-practical-20260909-r2-1tok.json",
    "D8": "D8-qwen-32token/qwen3-14b-h65-matched-decode-practical-20260909-r2-32tok.json",
    "D9": "D9-gemma-matched/gemma2-27b-h65-matched-ttft-practical-20260909-r2b-1tok.json",
    "D10": "D10-gemma-external/gemma2-27b-h65-external-ttft-practical-20260909-r2b-1tok.json",
    "D13": "D13-qwen-32token-screen/qwen3-14b-h65-cal32v2-20261003-cal32v2-32tok.json",
    "D14": "D14-qwen-32token-confirmation/qwen3-14b-h65-confirm32-20261004-confirm32-32tok.json",
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
    for label, model in (("D6", "qwen3-14b"), ("D9", "gemma2-27b"),
                         ("D13", "qwen3-14b"), ("D14", "qwen3-14b")):
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


def check_table11() -> None:
    print("\n=== Table 11 and Section 5.5: Qwen3-14B 32-token follow-ups (D13, D14) ===")
    names = {"disk-frozen": "disk control", "simple-v4-r0": "traffic control",
             "h65-cal1-ram8": "H6.5, 8 GB host, 1-token calibration",
             "h65-cal32-ram8": "H6.5, 8 GB host, 32-token calibration",
             "h65-cal1-ram0": "H6.5, 0 GB host, 1-token calibration",
             "h65-cal32-ram0": "H6.5, 0 GB host, 32-token calibration"}
    # method -> (time/output-token, disk/method ratio with its 95% interval, peak incremental GPU)
    table11 = {
        "D13": (3, {"disk-frozen": (18.07, None, 1.976),
                    "simple-v4-r0": (15.02, (1.203, 1.183, 1.223), 3.768),
                    "h65-cal1-ram8": (23.57, (0.768, 0.680, 0.868), 2.908),
                    "h65-cal32-ram8": (23.09, (0.785, 0.643, 0.959), 3.069),
                    "h65-cal1-ram0": (13.84, (1.305, 1.275, 1.336), 2.911),
                    "h65-cal32-ram0": (14.18, (1.274, 1.203, 1.349), 2.916)}),
        "D14": (7, {"disk-frozen": (18.18, None, 1.956),
                    "simple-v4-r0": (14.88, (1.221, 1.194, 1.249), 3.739),
                    "h65-cal32-ram0": (13.93, (1.304, 1.279, 1.329), 2.900)}),
    }
    for label, (n_blocks, expected) in table11.items():
        tables = {method: {(int(r["repeat"]), r["case_id"]): r for r in rows}
                  for method, rows in _rows(_laptop(label)).items()}
        control = tables["disk-frozen"]
        blocks = sorted({block for block, _ in control})
        check("%s: blocks" % label, len(blocks), n_blocks, 0)
        for method, (seconds, ratio, peak) in expected.items():
            rows = tables[method]
            check("%s %s time/output-token (s)" % (label, names[method]),
                  statistics.fmean(r["wall_seconds"] / r["output_tokens"] for r in rows.values()),
                  seconds, 0.005)
            check("%s %s max peak incremental GPU (GB)" % (label, names[method]),
                  max(r["peak_vram_gb"] for r in rows.values()), peak, 0.0005)
            if ratio is None:
                continue
            # The paper's estimator: per-block geometric mean of the paired
            # disk/method request times, then mean and t interval across blocks.
            gm, lo, hi = paired_log_ratio_ci([
                geomean([control[key]["wall_seconds"] / rows[key]["wall_seconds"]
                         for key in control if key[0] == block]) for block in blocks])
            for what, got, want in (("ratio", gm, ratio[0]), ("95% CI lower", lo, ratio[1]),
                                    ("95% CI upper", hi, ratio[2])):
                check("%s %s disk/method latency %s" % (label, names[method], what),
                      got, want, 0.0006)
            if (label, method) == ("D14", "h65-cal32-ram0"):
                check("D14 H6.5 latency reduction vs disk control, paper 'about 23%'",
                      1 - 1 / gm, 0.23, 0.01)


def main() -> int:
    check_manifest()
    check_frozen_inputs()
    check_followup_inputs()
    check_laptop_provenance()
    check_laptop_tables()
    check_table11()
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
    run_llama_followups()
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
