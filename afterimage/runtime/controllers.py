"""Output-invariant runtime controllers.

Controllers choose *how* exact work is scheduled or which already-validated
profile executes.  They never alter model weights, verification, or routing.
The implementations are intentionally small and serializable so an experiment
can compare them against a tuned constant without adding a training stack.
"""
from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import statistics
from typing import Callable, Iterable

import numpy as np


@dataclasses.dataclass(frozen=True)
class PrefetchObservation:
    ready: bool
    wait_s: float = 0.0
    useful_bytes: int = 0
    wasted_bytes: int = 0
    bandwidth_bytes_s: float = 0.0
    lead_s: float = 0.0
    lead_layers: int = 0


class FixedPrefetchController:
    def __init__(self, depth: int):
        self.depth = depth

    def choose_depth(self) -> int:
        return self.depth

    def update(self, _observation: PrefetchObservation) -> None:
        return None

    def update_compute(self, _seconds: float) -> None:
        return None

    def state_dict(self) -> dict:
        return {"policy": "fixed", "depth": self.depth}


class PIPrefetchController:
    """PI feedback over the recent fraction of layers ready on demand."""

    def __init__(self, initial_depth: int, max_depth: int, target_ready: float = 0.85,
                 kp: float = 2.0, ki: float = 0.25, ewma: float = 0.2):
        self.depth = max(0, min(max_depth, initial_depth))
        self.max_depth = max_depth
        self.target_ready = target_ready
        self.kp = kp
        self.ki = ki
        self.ewma = ewma
        self.ready_rate = target_ready
        self.integral = 0.0

    def choose_depth(self) -> int:
        return self.depth

    def update(self, observation: PrefetchObservation) -> None:
        sample = 1.0 if observation.ready else 0.0
        self.ready_rate = (1.0 - self.ewma) * self.ready_rate + self.ewma * sample
        error = self.target_ready - self.ready_rate
        self.integral = max(-2.0, min(2.0, self.integral + error))
        raw = self.depth + self.kp * error + self.ki * self.integral
        if observation.wasted_bytes > observation.useful_bytes and observation.ready:
            raw -= 1.0
        self.depth = max(0, min(self.max_depth, int(round(raw))))

    def update_compute(self, _seconds: float) -> None:
        return None

    def state_dict(self) -> dict:
        return {"policy": "pi", "depth": self.depth,
                "ready_rate": self.ready_rate, "integral": self.integral}


class MPCPrefetchController:
    """One-step model-predictive controller for bounded prefetch depth.

    It learns read and exposed-wait EWMAs from real layers, evaluates every
    feasible depth, and chooses the lowest predicted stall plus overfetch
    penalty.  This is a deliberately inspectable control baseline before RL.
    """

    def __init__(self, initial_depth: int, max_depth: int, waste_penalty_s_per_mb: float = 0.001,
                 ewma: float = 0.2):
        self.depth = max(0, min(max_depth, initial_depth))
        self.max_depth = max_depth
        self.waste_penalty = waste_penalty_s_per_mb
        self.ewma = ewma
        self.wait_s = 0.0
        self.layer_bytes = 0.0
        self.bandwidth = 0.0
        self.compute_s = 0.0

    def choose_depth(self) -> int:
        best_depth, best_cost = 0, float("inf")
        read_s = self.layer_bytes / self.bandwidth if self.bandwidth > 0 else self.wait_s
        for depth in range(self.max_depth + 1):
            hidden = depth * self.compute_s
            stall = max(0.0, read_s - hidden)
            overfetch_mb = max(0.0, depth - 1) * self.layer_bytes / 1e6
            cost = stall + self.waste_penalty * overfetch_mb
            if cost < best_cost:
                best_depth, best_cost = depth, cost
        self.depth = best_depth
        return best_depth

    def update(self, observation: PrefetchObservation) -> None:
        a = self.ewma
        self.wait_s = (1 - a) * self.wait_s + a * observation.wait_s
        self.layer_bytes = ((1 - a) * self.layer_bytes
                            + a * max(0, observation.useful_bytes))
        if observation.bandwidth_bytes_s > 0:
            self.bandwidth = ((1 - a) * self.bandwidth
                              + a * observation.bandwidth_bytes_s)

    def update_compute(self, seconds: float) -> None:
        a = self.ewma
        self.compute_s = (1 - a) * self.compute_s + a * max(0.0, float(seconds))

    def state_dict(self) -> dict:
        return {"policy": "mpc", "depth": self.depth,
                "wait_s": self.wait_s, "layer_bytes": self.layer_bytes,
                "bandwidth_bytes_s": self.bandwidth,
                "compute_s": self.compute_s}


class _LogNormalPosterior:
    """Normal-inverse-gamma posterior over one log-duration.

    The predictive distribution is technically Student-t.  The controller
    deliberately uses a normal quantile (a probit rule) after a small burn-in:
    it is cheap, inspectable, and its predictive variance still includes both
    observation noise and uncertainty in the posterior mean.
    """

    def __init__(self, prior_variance: float = 0.25):
        self.count = 0
        self.mu = 0.0
        self.kappa = 1e-6
        self.alpha = 2.0
        self.beta = float(prior_variance)

    def update(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            return
        x = math.log(value)
        old_kappa = self.kappa
        new_kappa = old_kappa + 1.0
        delta = x - self.mu
        self.mu = (old_kappa * self.mu + x) / new_kappa
        self.beta += 0.5 * old_kappa * delta * delta / new_kappa
        self.kappa = new_kappa
        self.alpha += 0.5
        self.count += 1

    @property
    def predictive_variance(self) -> float:
        return max(
            1e-9,
            self.beta * (self.kappa + 1.0) /
            (max(self.alpha - 1.0, 1e-9) * self.kappa),
        )

    def state_dict(self) -> dict:
        return {"count": self.count, "log_mean": self.mu,
                "predictive_variance": self.predictive_variance}


class BayesianProbitPrefetchController:
    """Chance-constrained prefetch depth from measured latency distributions.

    Let R be a layer's storage-read time and C the observed lead time per
    decoder layer.  On the log scale the controller models R and C with
    independent normal-inverse-gamma posteriors, then chooses the smallest d
    satisfying the probit approximation

        P(R <= d C) >= target_ready.

    It never predicts which layer is needed: dense execution order is known
    exactly.  Probability is used only to decide how early an exact future
    read should start.
    """

    def __init__(self, initial_depth: int, max_depth: int,
                 target_ready: float = 0.85, minimum_observations: int = 4):
        self.max_depth = max(0, int(max_depth))
        self.initial_depth = max(0, min(self.max_depth, int(initial_depth)))
        self.depth = self.initial_depth
        self.target_ready = float(target_ready)
        self.minimum_observations = max(2, int(minimum_observations))
        self.read = _LogNormalPosterior()
        self.window = _LogNormalPosterior()
        self.last_ready_probability: float | None = None
        self.calibration_count = 0
        self.calibration_brier_sum = 0.0
        self.empirical_ready_count = 0

    def choose_depth(self) -> int:
        if self.max_depth == 0:
            self.depth = 0
            return 0
        if (self.read.count < self.minimum_observations
                or self.window.count < self.minimum_observations):
            self.depth = max(1, self.initial_depth)
            return self.depth
        mean = self.read.mu - self.window.mu
        variance = self.read.predictive_variance + self.window.predictive_variance
        sigma = math.sqrt(max(variance, 1e-12))
        z = statistics.NormalDist().inv_cdf(self.target_ready)
        required = math.exp(min(50.0, mean + z * sigma))
        self.depth = max(1, min(self.max_depth, int(math.ceil(required))))
        standardized = (math.log(self.depth) - mean) / sigma
        self.last_ready_probability = statistics.NormalDist().cdf(standardized)
        return self.depth

    def update(self, observation: PrefetchObservation) -> None:
        if self.last_ready_probability is not None:
            outcome = 1.0 if observation.ready else 0.0
            self.calibration_count += 1
            self.empirical_ready_count += int(observation.ready)
            self.calibration_brier_sum += (
                self.last_ready_probability - outcome) ** 2
        if observation.bandwidth_bytes_s > 0 and observation.useful_bytes > 0:
            self.read.update(observation.useful_bytes /
                             observation.bandwidth_bytes_s)
        if observation.lead_layers > 0 and observation.lead_s > 0:
            self.window.update(observation.lead_s / observation.lead_layers)

    def update_compute(self, _seconds: float) -> None:
        return None

    def state_dict(self) -> dict:
        return {
            "policy": "bayes_probit", "depth": self.depth,
            "target_ready": self.target_ready,
            "last_ready_probability": self.last_ready_probability,
            "calibration_count": self.calibration_count,
            "brier_score": (self.calibration_brier_sum / self.calibration_count
                            if self.calibration_count else None),
            "empirical_ready_rate": (
                self.empirical_ready_count / self.calibration_count
                if self.calibration_count else None),
            "read_posterior": self.read.state_dict(),
            "lead_window_posterior": self.window.state_dict(),
        }


def build_prefetch_controller(policy: str, *, initial_depth: int, max_depth: int,
                              target_ready: float = 0.85, kp: float = 2.0,
                              ki: float = 0.25):
    if policy == "fixed":
        return FixedPrefetchController(initial_depth)
    if policy == "pi":
        return PIPrefetchController(initial_depth, max_depth, target_ready, kp, ki)
    if policy == "mpc":
        return MPCPrefetchController(initial_depth, max_depth)
    if policy == "bayes_probit":
        return BayesianProbitPrefetchController(
            initial_depth, max_depth, target_ready)
    raise ValueError("unknown prefetch policy %r" % policy)


class LinearProfileBandit:
    """LinUCB or linear Thompson sampling over complete method profiles.

    The baseline check is a practical guard, not the cumulative high-
    probability guarantee of Conservative Linear UCB. Experiments label it
    accordingly and must measure baseline violations explicitly.
    """

    def __init__(self, profiles: Iterable[str], context_dim: int, algorithm: str = "linucb",
                 alpha: float = 1.0, seed: int = 0, baseline_profile: str | None = None,
                 conservative_fraction: float = 0.10,
                 calibration_pulls: int = 1):
        self.profiles = tuple(profiles)
        if not self.profiles:
            raise ValueError("at least one profile is required")
        if algorithm not in ("linucb", "thompson"):
            raise ValueError("algorithm must be linucb or thompson")
        self.context_dim = context_dim
        self.algorithm = algorithm
        self.alpha = alpha
        self.baseline_profile = baseline_profile or self.profiles[0]
        if self.baseline_profile not in self.profiles:
            raise ValueError("baseline_profile is not in profiles")
        self.conservative_fraction = conservative_fraction
        self.calibration_pulls = max(1, int(calibration_pulls))
        self._a = {p: np.eye(context_dim, dtype=np.float64) for p in self.profiles}
        self._b = {p: np.zeros(context_dim, dtype=np.float64) for p in self.profiles}
        self._counts = {p: 0 for p in self.profiles}
        self._rng = np.random.default_rng(seed)
        self._baseline_rewards: list[float] = []

    def _prediction(self, profile: str, context: np.ndarray) -> tuple[float, float]:
        inv = np.linalg.inv(self._a[profile])
        theta = inv @ self._b[profile]
        mean = float(theta @ context)
        uncertainty = math.sqrt(max(0.0, float(context @ inv @ context)))
        return mean, uncertainty

    def choose(self, context) -> str:
        x = np.asarray(context, dtype=np.float64)
        if x.shape != (self.context_dim,):
            raise ValueError("context must have shape (%d,)" % self.context_dim)
        under_calibrated = [profile for profile in self.profiles
                            if self._counts[profile] < self.calibration_pulls]
        if under_calibrated:
            # One explicit calibration observation per complete profile is
            # necessary; otherwise deterministic LinUCB ties select the first
            # arm forever and the baseline guard prevents all learning.
            return min(under_calibrated, key=lambda profile: self._counts[profile])
        scores = {}
        for profile in self.profiles:
            mean, uncertainty = self._prediction(profile, x)
            if self.algorithm == "linucb":
                scores[profile] = mean + self.alpha * uncertainty
            else:
                scores[profile] = float(self._rng.normal(mean, self.alpha * uncertainty))
        choice = max(scores, key=scores.get)

        # A conservative guard: once baseline evidence exists, do not select
        # a profile whose pessimistic prediction is materially below it.
        if self._baseline_rewards and choice != self.baseline_profile:
            baseline_mean = float(np.mean(self._baseline_rewards))
            mean, uncertainty = self._prediction(choice, x)
            lower = mean - self.alpha * uncertainty
            if lower < (1.0 - self.conservative_fraction) * baseline_mean:
                choice = self.baseline_profile
        return choice

    def update(self, profile: str, context, reward: float) -> None:
        if profile not in self._a:
            raise ValueError("unknown profile %r" % profile)
        x = np.asarray(context, dtype=np.float64)
        if x.shape != (self.context_dim,):
            raise ValueError("context must have shape (%d,)" % self.context_dim)
        self._a[profile] += np.outer(x, x)
        self._b[profile] += reward * x
        self._counts[profile] += 1
        if profile == self.baseline_profile:
            self._baseline_rewards.append(float(reward))
            self._baseline_rewards = self._baseline_rewards[-128:]

    def reset(self) -> None:
        self._a = {p: np.eye(self.context_dim, dtype=np.float64)
                   for p in self.profiles}
        self._b = {p: np.zeros(self.context_dim, dtype=np.float64)
                   for p in self.profiles}
        self._counts = {p: 0 for p in self.profiles}
        self._baseline_rewards.clear()

    def state_dict(self) -> dict:
        return {
            "profiles": self.profiles, "context_dim": self.context_dim,
            "algorithm": self.algorithm, "alpha": self.alpha,
            "baseline_profile": self.baseline_profile,
            "conservative_fraction": self.conservative_fraction,
            "calibration_pulls": self.calibration_pulls,
            "a": {p: v.tolist() for p, v in self._a.items()},
            "b": {p: v.tolist() for p, v in self._b.items()},
            "counts": self._counts, "baseline_rewards": self._baseline_rewards,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        if tuple(state["profiles"]) != self.profiles or state["context_dim"] != self.context_dim:
            raise ValueError("bandit state belongs to a different profile/context space")
        if state.get("calibration_pulls", self.calibration_pulls) != self.calibration_pulls:
            raise ValueError("bandit state uses a different calibration_pulls setting")
        self._a = {p: np.asarray(v, dtype=np.float64) for p, v in state["a"].items()}
        self._b = {p: np.asarray(v, dtype=np.float64) for p, v in state["b"].items()}
        self._counts = {p: int(v) for p, v in state["counts"].items()}
        self._baseline_rewards = [float(v) for v in state.get("baseline_rewards", [])]
        if "rng_state" in state:
            self._rng.bit_generator.state = state["rng_state"]

    def save(self, path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state_dict()), encoding="utf-8")
        tmp.replace(path)

    def load(self, path) -> None:
        path = pathlib.Path(path)
        if path.exists():
            self.load_state_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclasses.dataclass
class SemiBanditItem:
    key: str
    size: int
    mean_reward: float = 0.0
    pulls: int = 0


class PageHinkley:
    """Small change detector for explicit bandit reset experiments."""

    def __init__(self, threshold: float = 5.0, delta: float = 0.01):
        self.threshold = threshold
        self.delta = delta
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0

    def update(self, value: float) -> bool:
        self.count += 1
        self.mean += (float(value) - self.mean) / self.count
        self.cumulative += float(value) - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        changed = self.cumulative - self.minimum > self.threshold
        if changed:
            self.reset()
        return changed


class KnapsackSemiBandit:
    """UCB item values plus an exact 0/1 knapsack for modest item counts."""

    def __init__(self, items: Iterable[SemiBanditItem], exploration: float = 1.0,
                 size_quantum: int = 1 << 20):
        self.items = {item.key: item for item in items}
        self.exploration = exploration
        self.size_quantum = max(1, size_quantum)
        self.round = 0

    def select(self, budget_bytes: int) -> list[str]:
        self.round += 1
        capacity = max(0, budget_bytes // self.size_quantum)
        keys = list(self.items)
        dp = [0.0] * (capacity + 1)
        chosen: list[set[str]] = [set() for _ in range(capacity + 1)]
        for key in keys:
            item = self.items[key]
            weight = max(1, math.ceil(item.size / self.size_quantum))
            bonus = (self.exploration * math.sqrt(math.log(self.round + 1) /
                                                  max(item.pulls, 1)))
            value = item.mean_reward + bonus
            for cap in range(capacity, weight - 1, -1):
                candidate = dp[cap - weight] + value
                if candidate > dp[cap]:
                    dp[cap] = candidate
                    chosen[cap] = chosen[cap - weight] | {key}
        return sorted(chosen[capacity])

    def update(self, component_rewards: dict[str, float]) -> None:
        for key, reward in component_rewards.items():
            if key not in self.items:
                continue
            item = self.items[key]
            item.pulls += 1
            item.mean_reward += (float(reward) - item.mean_reward) / item.pulls


class ModelBasedProfileController:
    """Receding-horizon planner over a caller-supplied calibrated simulator."""

    def __init__(self, profiles: Iterable[str], simulator: Callable,
                 horizon: int = 3, discount: float = 0.95,
                 baseline_profile: str | None = None, shadow: bool = True):
        self.profiles = tuple(profiles)
        self.simulator = simulator
        self.horizon = max(1, horizon)
        self.discount = discount
        self.baseline_profile = baseline_profile or self.profiles[0]
        self.shadow = shadow
        self.last_recommendation: str | None = None

    def _value(self, state, first_profile: str) -> float:
        value = 0.0
        current = state
        profile = first_profile
        for step in range(self.horizon):
            current, reward = self.simulator(current, profile)
            value += (self.discount ** step) * float(reward)
            if step + 1 < self.horizon:
                profile = max(self.profiles,
                              key=lambda candidate: self.simulator(current, candidate)[1])
        return value

    def choose(self, state) -> str:
        recommendation = max(self.profiles, key=lambda p: self._value(state, p))
        self.last_recommendation = recommendation
        return self.baseline_profile if self.shadow else recommendation
