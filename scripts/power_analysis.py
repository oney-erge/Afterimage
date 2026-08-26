#!/usr/bin/env python3
"""Estimate the paired-sample count an L3 confirmatory run needs, from the
run-to-run variance already observed in existing L1/L2 regulated-pair
results.

This is pure analysis on data already committed under results/ -- it needs
no hardware and produces no new experiment result, so it does not write to
results/ (see results/README.md's contract for what belongs there).

L3's decision rule (protocols.py's assess_paired_effect) requires a 95%
paired-bootstrap lower bound above zero and the point effect at or above the
hypothesis's registered minimum_effect. A 95% two-sided CI excluding zero on
one side is equivalent to a one-sided test at alpha=0.025, so this uses the
standard paired-sample-size formula on the log scale:

    n = ((z_alpha + z_beta) * sigma / delta) ** 2

where sigma is the observed standard deviation of the per-pair log-ratio
log(control_seconds) - log(candidate_seconds), and delta = ln(1 + minimum_effect)
is the log-scale effect the protocol requires detecting.

Caveats stated deliberately, not hidden in a footnote: sigma is estimated
from 2-4 blocks per hypothesis here, so the estimate itself has wide
uncertainty (few degrees of freedom) and should be treated as a rough
planning number, not a precise requirement. A prospective power calculation
belongs in the L3 protocol's preregistration, computed once per hypothesis
before that hypothesis's confirmatory run, not assumed transferable between
hypotheses with different mechanisms.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics

Z_ALPHA_ONE_SIDED_025 = 1.9599639845400545  # 95% two-sided CI lower bound > 0
Z_BETA_80_PERCENT_POWER = 0.8416212335729143
Z_BETA_90_PERCENT_POWER = 1.2815515655446004

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def paired_log_ratios(result: dict) -> list[float]:
    """log(control_seconds_per_token) - log(candidate_seconds_per_token) for
    every (block, case_id) pair present in both arms. Positive means the
    candidate was faster, matching protocols.assess_paired_effect's sign
    convention for a latency metric."""
    by_arm_key: dict[tuple[str, int, str], float] = {}
    for trial in result.get("trials", []):
        arm = trial["arm"]
        block = trial["block"]
        for row in trial.get("rows", []):
            key = (arm, block, row["case_id"])
            by_arm_key[key] = float(row["seconds_per_token"])
    pairs = set((block, case_id) for (arm, block, case_id) in by_arm_key
               if arm == "control") & set(
        (block, case_id) for (arm, block, case_id) in by_arm_key if arm == "candidate")
    ratios = []
    for block, case_id in sorted(pairs):
        control_s = by_arm_key[("control", block, case_id)]
        candidate_s = by_arm_key[("candidate", block, case_id)]
        ratios.append(math.log(control_s) - math.log(candidate_s))
    return ratios


def required_pairs(sigma: float, minimum_effect: float, *, power_z: float) -> float:
    delta = math.log(1.0 + minimum_effect)
    if delta <= 0 or sigma <= 0:
        return float("nan")
    return ((Z_ALPHA_ONE_SIDED_025 + power_z) * sigma / delta) ** 2


def retrospective_power(n_pairs: int, sigma: float, minimum_effect: float) -> float:
    """Power actually achieved by the n this hypothesis already ran, at its
    own registered gate -- lets you see how far short of 80% a completed L1/L2
    screen was, not just what a future L3 run would need."""
    if sigma <= 0 or n_pairs <= 0:
        return float("nan")
    delta = math.log(1.0 + minimum_effect)
    z = delta * math.sqrt(n_pairs) / sigma - Z_ALPHA_ONE_SIDED_025
    # standard normal CDF via erf, no scipy dependency
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def main() -> int:
    from afterimage.experiments import HYPOTHESES

    rows = []
    for path in sorted((REPO_ROOT / "results").glob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        hypothesis_id = result.get("hypothesis_id")
        if not hypothesis_id or hypothesis_id not in HYPOTHESES:
            continue
        ratios = paired_log_ratios(result)
        if len(ratios) < 2:
            continue
        sigma = statistics.stdev(ratios)
        gate = HYPOTHESES[hypothesis_id].minimum_effect
        rows.append({
            "file": path.name, "hypothesis_id": hypothesis_id,
            "n_pairs": len(ratios), "observed_sigma": sigma,
            "minimum_effect": gate,
            "n_for_80pct_power": required_pairs(sigma, gate, power_z=Z_BETA_80_PERCENT_POWER),
            "n_for_90pct_power": required_pairs(sigma, gate, power_z=Z_BETA_90_PERCENT_POWER),
            "retrospective_power_at_ran_n": retrospective_power(len(ratios), sigma, gate),
        })

    if not rows:
        print("No regulated-pair result files with parseable trials found "
              "under results/.")
        return 1

    header = ("file", "hyp", "n_ran", "sigma(log)", "gate", "n@80%pwr",
              "n@90%pwr", "power@n_ran")
    print(f"{header[0]:<58} {header[1]:<24} {header[2]:>5} {header[3]:>10} "
          f"{header[4]:>6} {header[5]:>9} {header[6]:>9} {header[7]:>12}")
    for row in rows:
        flag = " *" if row["n_pairs"] < 4 else "  "
        print(f"{row['file']:<58} {row['hypothesis_id']:<24} "
              f"{row['n_pairs']:>5} {row['observed_sigma']:>10.4f} "
              f"{row['minimum_effect']:>6.2%} {row['n_for_80pct_power']:>9.1f} "
              f"{row['n_for_90pct_power']:>9.1f} "
              f"{row['retrospective_power_at_ran_n']:>11.1%}{flag}")

    if any(row["n_pairs"] < 4 for row in rows):
        print()
        print("* fewer than 4 pairs: sigma has 1-2 degrees of freedom and "
              "can be wildly off in either direction. Treat these rows as "
              "'we do not yet know the variance,' not as a real n@power "
              "estimate.")
    print()
    print("sigma(log) is the sample standard deviation of the paired "
          "log-ratio observed in that file's own trials; n@80%/90% power is "
          "the paired count an L3 run targeting the registered gate would "
          "need if the true effect equals that gate and future runs show "
          "the same variance. power@n_ran is the power the completed run "
          "actually had at its own n, retrospectively -- a low number here "
          "does not mean the hypothesis is false, it means an L2 screen "
          "at this n cannot reliably detect an effect of exactly the gate "
          "size even if one is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
