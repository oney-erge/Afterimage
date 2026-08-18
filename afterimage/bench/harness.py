"""Run-matrix orchestration with repetition and stability checking
(IMPLEMENTATION_PLAN.md #4.3, #9).

N=5 repeats per configuration, median and IQR (never mean alone -- a single
slow run from thermal throttling or a stray background process should not
silently average away). Execution order across configurations is randomized
and interleaved rather than run in blocks, so a drive warming up over a long
run doesn't bias later-scheduled configurations (IMPLEMENTATION_PLAN.md
#4.2). Any metric whose IQR exceeds 15% of its median is flagged unstable
rather than silently reported -- callers should re-run flagged
configurations, not average through the noise.
"""
from __future__ import annotations

import dataclasses
import random
import statistics
from typing import Callable


@dataclasses.dataclass
class MetricResult:
    values: list[float]
    median: float
    iqr: float
    stable: bool


def _iqr(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    q = statistics.quantiles(sorted(values), n=4)
    return q[2] - q[0]


def aggregate(trials: list[dict[str, float]], instability_threshold: float = 0.15) -> dict[str, MetricResult]:
    if not trials:
        return {}
    keys = trials[0].keys()
    out = {}
    for k in keys:
        vals = [t[k] for t in trials]
        med = statistics.median(vals)
        iqr = _iqr(vals)
        stable = iqr <= instability_threshold * abs(med) if med != 0 else iqr < 1e-9
        out[k] = MetricResult(values=vals, median=med, iqr=iqr, stable=stable)
    return out


def run_matrix(configs: dict[str, Callable[[], dict[str, float]]], n_repeats: int = 5,
               seed: int = 0) -> dict[str, dict[str, MetricResult]]:
    """configs maps a configuration name to a zero-arg callable returning a
    dict of metric -> value for one trial. Execution is interleaved and
    shuffled across (config, repeat) pairs, not run config-by-config."""
    schedule = [(name, i) for name in configs for i in range(n_repeats)]
    rng = random.Random(seed)
    rng.shuffle(schedule)

    raw: dict[str, list[dict[str, float]]] = {name: [] for name in configs}
    for name, _i in schedule:
        raw[name].append(configs[name]())

    return {name: aggregate(trials) for name, trials in raw.items()}


def unstable_configs(results: dict[str, dict[str, MetricResult]]) -> list[tuple[str, str]]:
    """(config_name, metric_name) pairs that failed the stability check and
    should be re-run before being trusted in a report."""
    out = []
    for cfg, metrics in results.items():
        for metric_name, r in metrics.items():
            if not r.stable:
                out.append((cfg, metric_name))
    return out
