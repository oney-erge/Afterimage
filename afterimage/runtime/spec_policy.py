"""Adaptive control over speculative draft length (and, for self-draft, when
to stop drafting) -- docs/PROPOSAL_ADAPTIVE.md mechanism B.

Every policy here answers exactly one question -- "how many tokens should
the next draft chain try for?" -- and is judged purely on wall-clock speed,
never correctness. That is deliberate and it is what makes this a safe place
to explore aggressively: runtime.verify.speculative_sample_step guarantees
the SAME output distribution regardless of k or how the draft was produced
(see its docstring). A bad choice of k costs a slow sweep. It cannot produce
a wrong token. Most RL-for-efficiency work (pruning, quantization search) has
to explore cautiously because a bad choice risks quality; this project's
knobs don't have that failure mode, so a plain bandit/heuristic is enough --
no training, no held-out validation of "did accuracy drop."

FixedPolicy is the control arm every other policy must beat -- per
PROPOSAL_ADAPTIVE.md's test plan, "adaptive beats a tuned constant" is the
actual bar, not "adaptive beats k=8."
"""
from __future__ import annotations

import dataclasses
import json
import pathlib


@dataclasses.dataclass
class SweepRecord:
    k_used: int
    n_accepted: int
    sweep_seconds: float

    @property
    def acceptance(self) -> float:
        return self.n_accepted / max(self.k_used, 1)

    @property
    def tokens_per_second(self) -> float:
        # n_accepted + 1: the accepted draft prefix plus the correction/bonus
        # token every sweep emits regardless (verify.speculative_sample_step
        # always returns at least one token).
        return (self.n_accepted + 1) / max(self.sweep_seconds, 1e-9)


class SpecPolicy:
    name = "base"

    def choose_k(self) -> int:
        raise NotImplementedError

    def update(self, record: SweepRecord) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def save(self, path) -> None:
        pathlib.Path(path).write_text(
            json.dumps({"policy": self.name, "state": self.state_dict()}, indent=2))

    def load(self, path) -> None:
        p = pathlib.Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text())
        if data.get("policy") != self.name:
            raise ValueError(
                "spec_policy_state at %s was saved by policy %r, this is %r "
                "-- policies keep separate state files, they don't share one"
                % (path, data.get("policy"), self.name))
        self.load_state_dict(data.get("state", {}))


class FixedPolicy(SpecPolicy):
    """Returns a constant k forever. The control arm."""
    name = "fixed"

    def __init__(self, k: int):
        self.k = k

    def choose_k(self) -> int:
        return self.k

    def update(self, record: SweepRecord) -> None:
        pass


class GammaTunePolicy(SpecPolicy):
    """EWMA over recent acceptance; expands k by 1 when the smoothed
    acceptance rate is high, contracts it by 1 when low. Training-free by
    construction (https://arxiv.org/pdf/2504.00030), which matters here
    because a ~20-token answer is only ~6 sweeps -- far too few to fit
    anything with real parameters within a single run. State persists
    across runs via EngineConfig.spec_policy_state precisely because of that.
    """
    name = "gamma"

    def __init__(self, k_init: int = 8, k_min: int = 1, k_max: int = 16,
                 ewma_alpha: float = 0.3, high: float = 0.7, low: float = 0.4):
        self.k = k_init
        self.k_min = k_min
        self.k_max = k_max
        self.ewma_alpha = ewma_alpha
        self.high = high
        self.low = low
        self.mean_acceptance = None

    def choose_k(self) -> int:
        return self.k

    def update(self, record: SweepRecord) -> None:
        a = record.acceptance
        self.mean_acceptance = (
            a if self.mean_acceptance is None
            else self.ewma_alpha * a + (1 - self.ewma_alpha) * self.mean_acceptance)
        if self.mean_acceptance >= self.high:
            self.k = min(self.k_max, self.k + 1)
        elif self.mean_acceptance <= self.low:
            self.k = max(self.k_min, self.k - 1)

    def state_dict(self) -> dict:
        return {"k": self.k, "mean_acceptance": self.mean_acceptance}

    def load_state_dict(self, state: dict) -> None:
        self.k = state.get("k", self.k)
        self.mean_acceptance = state.get("mean_acceptance")


class ThresholdPolicy(SpecPolicy):
    """SpecDec++ (https://arxiv.org/pdf/2405.19715) proves the OPTIMAL
    stopping rule for a draft chain has a threshold shape: keep drafting
    while confidence stays high, stop (and verify) the moment it drops. This
    uses the draft's own max-softmax-probability at each proposed position as
    a free proxy for accept probability, instead of training a separate
    acceptance-prediction head as the paper does -- the engine already
    computes that probability at every drafted position regardless.

    choose_k returns k_max as an UPPER bound on how many positions to
    propose; trim_by_confidence is the actual mechanism, called by the
    caller as tokens are drafted one at a time (see
    streaming_engine.generate_adaptive), so a low-confidence position stops
    drafting immediately rather than proposing k_max tokens and discarding
    the low-confidence tail afterward.
    """
    name = "threshold"

    def __init__(self, k_min: int = 1, k_max: int = 16,
                 confidence_threshold: float = 0.5, threshold_step: float = 0.05):
        self.k_min = k_min
        self.k_max = k_max
        self.confidence_threshold = confidence_threshold
        self.threshold_step = threshold_step
        self._best_tps = 0.0

    def choose_k(self) -> int:
        return self.k_max

    def trim_by_confidence(self, draft_probs: list) -> int:
        """draft_probs[i] is the draft's own distribution at chain position
        i. Returns how many leading positions to keep -- the first position
        whose max probability drops below the threshold ends the chain
        there (never below k_min, so a chain is never trimmed to nothing)."""
        for i, p in enumerate(draft_probs):
            if i >= self.k_min and float(p.max()) < self.confidence_threshold:
                return i
        return len(draft_probs)

    def update(self, record: SweepRecord) -> None:
        # An untrained early-exit draft's own confidence may not track its
        # actual acceptance rate well; if speed regresses while acceptance
        # is low, raise the bar so the chain stops sooner next time.
        if record.tokens_per_second > self._best_tps:
            self._best_tps = record.tokens_per_second
        elif record.acceptance < 0.3:
            self.confidence_threshold = min(0.95, self.confidence_threshold + self.threshold_step)

    def state_dict(self) -> dict:
        return {"confidence_threshold": self.confidence_threshold, "best_tps": self._best_tps}

    def load_state_dict(self, state: dict) -> None:
        self.confidence_threshold = state.get("confidence_threshold", self.confidence_threshold)
        self._best_tps = state.get("best_tps", 0.0)


def build_policy(policy_name: str, spec_k: int) -> SpecPolicy:
    if policy_name == "fixed":
        return FixedPolicy(spec_k)
    if policy_name == "gamma":
        return GammaTunePolicy(k_init=spec_k)
    if policy_name == "threshold":
        return ThresholdPolicy(k_max=spec_k)
    raise ValueError("unknown spec_k_policy %r" % (policy_name,))
