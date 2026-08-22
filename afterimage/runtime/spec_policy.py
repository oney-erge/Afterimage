"""Adaptive control over speculative draft length (and, for self-draft, when
to stop drafting) -- docs/archive/PROPOSAL_ADAPTIVE.md mechanism B.

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
import math
import pathlib

import numpy as np


@dataclasses.dataclass
class SweepRecord:
    k_used: int
    n_accepted: int
    sweep_seconds: float
    draft_confidences: tuple[float, ...] = ()
    draft_entropies: tuple[float, ...] = ()
    draft_seconds: float | None = None
    target_seconds: float | None = None

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

    def should_stop(self, draft_probs: list) -> bool:
        """Whether to stop before sampling the distribution at the tail.

        The default keeps the existing fixed/gamma behaviour.  Policies
        with per-position evidence override it.
        """
        return False

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def save(self, path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"policy": self.name, "state": self.state_dict()}, indent=2),
            encoding="utf-8")
        tmp.replace(path)

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

    def should_stop(self, draft_probs: list) -> bool:
        return (len(draft_probs) > self.k_min
                and float(draft_probs[-1].max()) < self.confidence_threshold)

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


def _prob_entropy(prob) -> float:
    """Shannon entropy without requiring a particular tensor library."""
    try:
        import torch
        p = prob.detach().to(dtype=torch.float64)
        return float(-(p * p.clamp_min(1e-300).log()).sum())
    except (ImportError, AttributeError):
        values = [max(float(v), 1e-300) for v in prob]
        return -sum(v * math.log(v) for v in values)


class AdaEDLPolicy(SpecPolicy):
    """Training-free entropy-bound early draft stopping.

    Implements Agrawal et al.'s approximate acceptance lower bound
    ``1 - sqrt(gamma * H(p))`` and their acceptance-EWMA threshold update.
    The policy keeps the first ``k_min`` positions and stops when the bound
    falls below the current threshold.
    """
    name = "adaedl"

    def __init__(self, k_max: int = 16, k_min: int = 1,
                 gamma: float = 0.2, threshold: float = 0.5,
                 target_acceptance: float = 0.9, threshold_step: float = 0.01,
                 acceptance_ewma: float = 0.5, threshold_ewma: float = 0.9):
        self.k_max = k_max
        self.k_min = k_min
        self.gamma = gamma
        self.threshold = threshold
        self.target_acceptance = target_acceptance
        self.threshold_step = threshold_step
        self.acceptance_ewma = acceptance_ewma
        self.threshold_ewma = threshold_ewma
        self.mean_acceptance: float | None = None

    def choose_k(self) -> int:
        return self.k_max

    def should_stop(self, draft_probs: list) -> bool:
        if len(draft_probs) <= self.k_min:
            return False
        lower_bound = 1.0 - math.sqrt(max(0.0, self.gamma *
                                          _prob_entropy(draft_probs[-1])))
        return lower_bound < self.threshold

    def update(self, record: SweepRecord) -> None:
        acceptance = record.acceptance
        self.mean_acceptance = (
            acceptance if self.mean_acceptance is None else
            self.acceptance_ewma * self.mean_acceptance
            + (1.0 - self.acceptance_ewma) * acceptance)
        proposal = self.threshold
        if self.mean_acceptance < self.target_acceptance:
            proposal += self.threshold_step
        elif record.n_accepted != self.k_max:
            proposal -= self.threshold_step
        self.threshold = max(0.0, min(1.0,
            self.threshold_ewma * self.threshold
            + (1.0 - self.threshold_ewma) * proposal))

    def state_dict(self) -> dict:
        return {"threshold": self.threshold,
                "mean_acceptance": self.mean_acceptance,
                "gamma": self.gamma, "k_max": self.k_max}

    def load_state_dict(self, state: dict) -> None:
        if state.get("k_max", self.k_max) != self.k_max:
            raise ValueError("AdaEDL state uses a different k_max")
        if state.get("gamma", self.gamma) != self.gamma:
            raise ValueError("AdaEDL state uses a different gamma")
        self.threshold = float(state.get("threshold", self.threshold))
        self.mean_acceptance = state.get("mean_acceptance")


class HazardCostPolicy(SpecPolicy):
    """Discrete survival model for the first rejected draft position.

    A sweep reveals accepted positions until the first rejection; positions
    after it are censored.  Beta posteriors indexed by position and draft
    confidence learn the conditional survival probability without treating
    censored positions as failures.  Drafting continues only while the
    expected target-sweep time saved exceeds the measured marginal draft
    cost.  Exact verification remains unchanged.
    """
    name = "hazard_cost"

    def __init__(self, k_max: int = 16, k_min: int = 1, n_bins: int = 10,
                 prior_accept: float = 2.0, prior_reject: float = 2.0,
                 ewma: float = 0.2):
        self.k_max = k_max
        self.k_min = k_min
        self.n_bins = n_bins
        self.prior_accept = prior_accept
        self.prior_reject = prior_reject
        self.ewma = ewma
        self.accept = [[0 for _ in range(n_bins)] for _ in range(k_max)]
        self.reject = [[0 for _ in range(n_bins)] for _ in range(k_max)]
        self.draft_token_s = 0.001
        self.target_token_s = 1.0

    def choose_k(self) -> int:
        return self.k_max

    def _bin(self, confidence: float) -> int:
        return min(self.n_bins - 1, max(0, int(confidence * self.n_bins)))

    def conditional_acceptance(self, position: int, confidence: float) -> float:
        pos = min(max(position, 0), self.k_max - 1)
        b = self._bin(confidence)
        yes = self.prior_accept + self.accept[pos][b]
        no = self.prior_reject + self.reject[pos][b]
        return yes / (yes + no)

    def should_stop(self, draft_probs: list) -> bool:
        if len(draft_probs) <= self.k_min:
            return False
        position = len(draft_probs) - 1
        confidence = float(draft_probs[-1].max())
        survive = self.conditional_acceptance(position, confidence)
        expected_saved_s = survive * self.target_token_s
        return expected_saved_s <= self.draft_token_s

    def update(self, record: SweepRecord) -> None:
        confidences = record.draft_confidences
        for position, confidence in enumerate(confidences[:record.k_used]):
            b = self._bin(confidence)
            if position < record.n_accepted:
                self.accept[position][b] += 1
            elif position == record.n_accepted and record.n_accepted < record.k_used:
                self.reject[position][b] += 1
                break
        if record.draft_seconds is not None and record.k_used:
            sample = record.draft_seconds / record.k_used
            self.draft_token_s = ((1 - self.ewma) * self.draft_token_s
                                  + self.ewma * sample)
        if record.target_seconds is not None:
            per_token = record.target_seconds / max(record.n_accepted + 1, 1)
            self.target_token_s = ((1 - self.ewma) * self.target_token_s
                                   + self.ewma * per_token)

    def state_dict(self) -> dict:
        return {"accept": self.accept, "reject": self.reject,
                "draft_token_s": self.draft_token_s,
                "target_token_s": self.target_token_s,
                "n_bins": self.n_bins, "k_max": self.k_max}

    def load_state_dict(self, state: dict) -> None:
        if state.get("n_bins", self.n_bins) != self.n_bins:
            raise ValueError("hazard state uses a different confidence bin count")
        if state.get("k_max", self.k_max) != self.k_max:
            raise ValueError("hazard state uses a different k_max")
        self.accept = state.get("accept", self.accept)
        self.reject = state.get("reject", self.reject)
        self.draft_token_s = state.get("draft_token_s", self.draft_token_s)
        self.target_token_s = state.get(
            "target_token_s", state.get("target_sweep_s", self.target_token_s))


class NeuralUtilityPolicy(SpecPolicy):
    """Tiny censored-survival network with an explicit throughput objective.

    This borrows the cascade-feedback view from information retrieval: the
    first rejected token is observed, while positions after it are censored.
    A six-hidden-unit MLP pools evidence across confidence, entropy and chain
    position instead of maintaining a sparse bin for every combination.  It
    does *not* choose tokens or approximate target probabilities.  It only
    stops drafting when the predicted expected tokens/second would decline;
    the normal exact verifier remains the authority.

    The model is trained only when ``spec_policy_learn`` lets the caller invoke
    update, can be frozen on held-out requests, and serializes as ordinary JSON.
    """
    name = "neural_utility"

    def __init__(self, k_max: int = 16, k_min: int = 1, hidden: int = 6,
                 learning_rate: float = 0.04, training_steps: int = 3,
                 minimum_observations: int = 24, ewma: float = 0.2,
                 seed: int = 0):
        self.k_max = k_max
        self.k_min = k_min
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.minimum_observations = minimum_observations
        self.ewma = ewma
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.08, size=(5, hidden))
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.w2 = rng.normal(0.0, 0.08, size=hidden)
        self.b2 = math.log(0.6 / 0.4)
        self.n_observations = 0
        self.brier_sum = 0.0
        self.positive_labels = 0
        self.draft_token_s = 0.001
        self.target_sweep_s = 1.0
        self.decision_stops = 0
        self.decision_continues = 0
        self.last_stop_position: int | None = None
        self.last_required_survival: float | None = None
        self.min_required_survival: float | None = None
        self.max_required_survival: float | None = None

    def choose_k(self) -> int:
        return self.k_max

    def _features(self, confidence: float, entropy: float, position: int) -> np.ndarray:
        pos = min(max(position / max(self.k_max - 1, 1), 0.0), 1.0)
        ent = max(0.0, float(entropy))
        ent = ent / (1.0 + ent)
        conf = min(max(float(confidence), 0.0), 1.0)
        return np.asarray([conf, ent, pos, conf * pos, 1.0], dtype=np.float64)

    def _predict(self, x: np.ndarray) -> float:
        hidden = np.tanh(x @ self.w1 + self.b1)
        logit = float(hidden @ self.w2 + self.b2)
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp = math.exp(logit)
        return exp / (1.0 + exp)

    def _fit_one(self, x: np.ndarray, label: float) -> None:
        for _ in range(self.training_steps):
            hidden = np.tanh(x @ self.w1 + self.b1)
            prediction = self._predict(x)
            dlogit = prediction - label
            grad_w2 = hidden * dlogit
            grad_b2 = dlogit
            dhidden = self.w2 * dlogit
            dz = dhidden * (1.0 - hidden * hidden)
            grad_w1 = np.outer(x, dz)
            grad_b1 = dz
            self.w2 -= self.learning_rate * grad_w2
            self.b2 -= self.learning_rate * grad_b2
            self.w1 -= self.learning_rate * grad_w1
            self.b1 -= self.learning_rate * grad_b1

    def _expected_tokens_and_seconds(self, draft_probs: list) -> tuple[float, float]:
        survival = 1.0
        accepted = 0.0
        for position, prob in enumerate(draft_probs):
            confidence = float(prob.max())
            entropy = _prob_entropy(prob)
            survival *= self._predict(self._features(confidence, entropy, position))
            accepted += survival
        expected_tokens = 1.0 + accepted
        expected_seconds = self.target_sweep_s + len(draft_probs) * self.draft_token_s
        return expected_tokens, expected_seconds

    def _expected_utility(self, draft_probs: list) -> float:
        tokens, seconds = self._expected_tokens_and_seconds(draft_probs)
        return tokens / max(seconds, 1e-9)

    def should_stop(self, draft_probs: list) -> bool:
        if (len(draft_probs) <= self.k_min
                or self.n_observations < self.minimum_observations):
            self.decision_continues += 1
            return False
        without_tokens, without_seconds = self._expected_tokens_and_seconds(
            draft_probs[:-1])
        with_tokens, with_seconds = self._expected_tokens_and_seconds(draft_probs)
        # The break-even survival probability for the token just drafted,
        # given the currently learned costs: stopping only helps once the
        # network's predicted survival falls below this value. Because
        # draft_token_s is measured in tens of milliseconds against a
        # target_sweep_s measured in seconds for an offloaded target, this
        # threshold is structurally tiny (commonly under 2%) -- exposed here
        # so "zero stop decisions" is diagnosable instead of a mystery. See
        # docs/HYPOTHESIS_LINEAGE.md's H11 correction.
        required = min(1.0, max(
            0.0, without_tokens * self.draft_token_s / max(without_seconds, 1e-9)))
        self.last_required_survival = required
        self.min_required_survival = (required if self.min_required_survival is None
                                      else min(self.min_required_survival, required))
        self.max_required_survival = (required if self.max_required_survival is None
                                      else max(self.max_required_survival, required))
        stop = (with_tokens / max(with_seconds, 1e-9)
               <= without_tokens / max(without_seconds, 1e-9))
        if stop:
            self.decision_stops += 1
            self.last_stop_position = len(draft_probs) - 1
        else:
            self.decision_continues += 1
        return stop

    def update(self, record: SweepRecord) -> None:
        confidences = record.draft_confidences
        entropies = record.draft_entropies
        observed = min(record.k_used, len(confidences), len(entropies))
        for position in range(observed):
            if position < record.n_accepted:
                label = 1.0
            elif position == record.n_accepted and record.n_accepted < record.k_used:
                label = 0.0
            else:
                break  # tail after the first rejection is censored
            x = self._features(confidences[position], entropies[position], position)
            prediction = self._predict(x)
            self.brier_sum += (prediction - label) ** 2
            self.positive_labels += int(label)
            self._fit_one(x, label)
            self.n_observations += 1
            if label == 0.0:
                break
        if record.draft_seconds is not None and record.k_used:
            sample = record.draft_seconds / record.k_used
            self.draft_token_s = ((1.0 - self.ewma) * self.draft_token_s
                                  + self.ewma * sample)
        if record.target_seconds is not None:
            self.target_sweep_s = ((1.0 - self.ewma) * self.target_sweep_s
                                   + self.ewma * record.target_seconds)

    def state_dict(self) -> dict:
        return {
            "k_max": self.k_max, "hidden": self.hidden,
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2,
            "n_observations": self.n_observations,
            "brier_score": (self.brier_sum / self.n_observations
                            if self.n_observations else None),
            "brier_sum": self.brier_sum,
            "positive_labels": self.positive_labels,
            "positive_rate": (self.positive_labels / self.n_observations
                              if self.n_observations else None),
            "draft_token_s": self.draft_token_s,
            "target_sweep_s": self.target_sweep_s,
            "decision_stops": self.decision_stops,
            "decision_continues": self.decision_continues,
            "last_stop_position": self.last_stop_position,
            "last_required_survival": self.last_required_survival,
            "min_required_survival": self.min_required_survival,
            "max_required_survival": self.max_required_survival,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("k_max", self.k_max) != self.k_max:
            raise ValueError("neural utility state uses a different k_max")
        if state.get("hidden", self.hidden) != self.hidden:
            raise ValueError("neural utility state uses a different hidden size")
        self.w1 = np.asarray(state.get("w1", self.w1), dtype=np.float64)
        self.b1 = np.asarray(state.get("b1", self.b1), dtype=np.float64)
        self.w2 = np.asarray(state.get("w2", self.w2), dtype=np.float64)
        self.b2 = float(state.get("b2", self.b2))
        self.n_observations = int(state.get("n_observations", 0))
        self.brier_sum = float(state.get("brier_sum", 0.0))
        self.positive_labels = int(state.get("positive_labels", 0))
        self.draft_token_s = float(state.get("draft_token_s", self.draft_token_s))
        self.target_sweep_s = float(state.get("target_sweep_s", self.target_sweep_s))
        # Decision counters describe this evaluation, not the calibration
        # file being loaded. They are intentionally reset on load.
        self.decision_stops = 0
        self.decision_continues = 0
        self.last_stop_position = None


def build_policy(policy_name: str, spec_k: int) -> SpecPolicy:
    if policy_name == "fixed":
        return FixedPolicy(spec_k)
    if policy_name == "gamma":
        return GammaTunePolicy(k_init=spec_k)
    if policy_name == "threshold":
        return ThresholdPolicy(k_max=spec_k)
    if policy_name == "adaedl":
        return AdaEDLPolicy(k_max=spec_k)
    if policy_name == "hazard_cost":
        return HazardCostPolicy(k_max=spec_k)
    if policy_name == "neural_utility":
        return NeuralUtilityPolicy(k_max=spec_k)
    raise ValueError("unknown spec_k_policy %r" % (policy_name,))
