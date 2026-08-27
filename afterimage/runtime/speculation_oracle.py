"""Offline analysis for the speculation-tree research line's next two gated
hypotheses (see docs/SPECULATION_TREE_RESEARCH.md):

H21 (Multi-Source Oracle Headroom) -- before integrating a second drafter
("Scout") into any runtime mechanism, does it actually cover target
continuations the primary drafter misses? compute_oracle_coverage_stats
answers that from collected (target_rank_under_primary,
target_rank_under_scout) traces, with no live speculative execution
involved.

H22 (Persistent Disagreement State) -- is draft/target disagreement
temporally predictable beyond the current position's own confidence, i.e.
does a hidden-state model of the draft/target relationship have real
predictive power? DiscreteHMM is a from-scratch, log-space, numerically
stable discrete Baum-Welch/forward-backward implementation (no new
dependency), validated in tests/test_speculation_oracle.py against
synthetic sequences generated from a KNOWN HMM -- the standard way to
verify an HMM implementation is correct, since there is no GPU-side
correctness question here at all: this is pure CPU statistics.

Both are deliberately offline: nothing here decides what a live
speculative sweep does. That is the whole point of running them before
building anything that would.
"""
from __future__ import annotations

import dataclasses
import math
import statistics

import numpy as np


def rank_of_token(logits: np.ndarray, token_id: int) -> int:
    """1-indexed rank of token_id in logits' descending order (rank 1 =
    argmax). O(vocab), which is cheap next to the forward pass that
    produced logits."""
    return int((logits > logits[token_id]).sum()) + 1


def disagreement_bucket(rank: int, rank_buckets: tuple[int, ...] = (1, 2, 4, 8)) -> int:
    """Discretizes a token rank into one of len(rank_buckets)+1 ordinal
    buckets: index i means rank <= rank_buckets[i] (first such i), and the
    last index means rank > every listed bucket. rank_buckets=(1,2,4,8)
    matches the user's own rank_buckets: [1, 2, 4, 8, inf] convention.
    """
    for i, bound in enumerate(rank_buckets):
        if rank <= bound:
            return i
    return len(rank_buckets)


def compute_oracle_coverage_stats(rows: list[dict], top_k_values: list[int]) -> dict:
    """H21's actual answer. Each row must carry target_rank_under_primary
    and target_rank_under_scout (1-indexed ranks from rank_of_token).

    For each k in top_k_values, reports:
      primary_coverage    = P(target rank <= k under Primary)
      scout_coverage      = P(target rank <= k under Scout)
      union_coverage      = P(covered by either)
      conditional_rescue_recall
                          = P(target rank <= k under Scout | NOT covered by Primary)
                            -- the number this hypothesis is actually about.
                            None when Primary never missed at this k (nothing
                            to rescue), not 0.0 -- 0.0 would falsely claim the
                            scout failed every rescue opportunity when there
                            were none to take.

    jaccard_overlap_at_collection_k uses each row's own primary_topk/
    scout_topk sets (only meaningful at the k used when the trace was
    collected, hence the separate name).
    """
    if not rows:
        return {"top_k": {}, "jaccard_overlap_at_collection_k": None, "rows": 0}
    top_k_stats = {}
    for k in top_k_values:
        primary_hits = [row["target_rank_under_primary"] <= k for row in rows]
        scout_hits = [row["target_rank_under_scout"] <= k for row in rows]
        union_hits = [p or s for p, s in zip(primary_hits, scout_hits)]
        rescue_opportunities = [s for p, s in zip(primary_hits, scout_hits) if not p]
        top_k_stats[k] = {
            "primary_coverage": statistics.mean(primary_hits),
            "scout_coverage": statistics.mean(scout_hits),
            "union_coverage": statistics.mean(union_hits),
            "primary_miss_count": len(rescue_opportunities),
            "conditional_rescue_recall": (
                statistics.mean(rescue_opportunities) if rescue_opportunities else None),
        }
    jaccard_values = []
    for row in rows:
        if "primary_topk" not in row or "scout_topk" not in row:
            continue
        primary_set, scout_set = set(row["primary_topk"]), set(row["scout_topk"])
        union = primary_set | scout_set
        jaccard_values.append(len(primary_set & scout_set) / len(union) if union else 1.0)
    return {
        "top_k": top_k_stats,
        "jaccard_overlap_at_collection_k": (
            statistics.mean(jaccard_values) if jaccard_values else None),
        "rows": len(rows),
    }


_LOG_EPS = 1e-300  # floor before log() so a zero-probability entry gives a
                   # large finite negative log rather than -inf propagating
                   # through logsumexp and silently poisoning every state.


def _log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, _LOG_EPS, None))


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)  # an all -inf slice must not become NaN
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


@dataclasses.dataclass
class DiscreteHMM:
    """A from-scratch discrete-observation HMM: n_states hidden regimes
    (H22's "aligned / weak disagreement / broad ambiguity / severe
    divergence"), n_symbols discrete observations (disagreement_bucket's
    output). All arithmetic is in log-space throughout fit/forward/
    backward -- a plain-probability implementation underflows silently on
    sequences longer than a few dozen steps, which is exactly wrong for a
    real generation trace.
    """
    n_states: int
    n_symbols: int
    log_pi: np.ndarray       # (n_states,)
    log_A: np.ndarray        # (n_states, n_states) -- log_A[i, j] = log P(state_{t+1}=j | state_t=i)
    log_B: np.ndarray        # (n_states, n_symbols) -- log_B[i, k] = log P(obs=k | state=i)

    @classmethod
    def random_init(cls, n_states: int, n_symbols: int, seed: int) -> "DiscreteHMM":
        rng = np.random.default_rng(seed)
        pi = rng.dirichlet(np.ones(n_states))
        A = rng.dirichlet(np.ones(n_states), size=n_states)
        B = rng.dirichlet(np.ones(n_symbols), size=n_states)
        return cls(n_states, n_symbols, _log(pi), _log(A), _log(B))

    def _forward_log(self, obs: np.ndarray) -> np.ndarray:
        """log alpha[t, i] = log P(obs[0..t], state_t=i)."""
        T = len(obs)
        log_alpha = np.empty((T, self.n_states))
        log_alpha[0] = self.log_pi + self.log_B[:, obs[0]]
        for t in range(1, T):
            # log_alpha[t-1] (n_states,) broadcast against log_A (n_states, n_states):
            # entry [i, j] = log_alpha[t-1, i] + log_A[i, j], summed over i.
            log_alpha[t] = _logsumexp(
                log_alpha[t - 1][:, None] + self.log_A, axis=0) + self.log_B[:, obs[t]]
        return log_alpha

    def _backward_log(self, obs: np.ndarray) -> np.ndarray:
        """log beta[t, i] = log P(obs[t+1..T-1] | state_t=i)."""
        T = len(obs)
        log_beta = np.empty((T, self.n_states))
        log_beta[T - 1] = 0.0
        for t in range(T - 2, -1, -1):
            log_beta[t] = _logsumexp(
                self.log_A + self.log_B[None, :, obs[t + 1]] + log_beta[t + 1][None, :],
                axis=1)
        return log_beta

    def log_likelihood(self, obs: np.ndarray) -> float:
        if len(obs) == 0:
            return 0.0
        return float(_logsumexp(self._forward_log(obs)[-1], axis=0))

    def fit(self, sequences: list[np.ndarray], max_iterations: int = 50,
           tol: float = 1e-4) -> list[float]:
        """Baum-Welch (EM) over multiple independent observation sequences
        (one per collected generation trace -- they do not share state
        across sequence boundaries). Returns the total log-likelihood after
        each iteration so a caller can confirm monotonic improvement (a
        real, cheap correctness check: EM must never decrease the
        likelihood it is optimizing).
        """
        history = []
        for _ in range(max_iterations):
            pi_num = np.zeros(self.n_states)
            A_num = np.full((self.n_states, self.n_states), _LOG_EPS)
            A_den = np.full(self.n_states, _LOG_EPS)
            B_num = np.full((self.n_states, self.n_symbols), _LOG_EPS)
            B_den = np.full(self.n_states, _LOG_EPS)
            total_ll = 0.0

            for obs in sequences:
                if len(obs) == 0:
                    continue
                log_alpha = self._forward_log(obs)
                log_beta = self._backward_log(obs)
                seq_ll = float(_logsumexp(log_alpha[-1], axis=0))
                total_ll += seq_ll

                log_gamma = log_alpha + log_beta - seq_ll
                gamma = np.exp(log_gamma)
                pi_num += gamma[0]
                for k in range(self.n_symbols):
                    mask = (obs == k)
                    if mask.any():
                        B_num[:, k] += gamma[mask].sum(axis=0)
                B_den += gamma.sum(axis=0)

                T = len(obs)
                if T > 1:
                    # xi[t, i, j] = P(state_t=i, state_{t+1}=j | obs, model),
                    # accumulated per-step rather than as one 3D array --
                    # clearer than the vectorized version, and T is at most
                    # a few hundred positions per generation trace.
                    for t in range(T - 1):
                        log_xi_t = (log_alpha[t][:, None] + self.log_A
                                   + self.log_B[:, obs[t + 1]][None, :]
                                   + log_beta[t + 1][None, :] - seq_ll)
                        xi_t = np.exp(log_xi_t)
                        A_num += xi_t
                    A_den += gamma[:-1].sum(axis=0)

            pi_new = pi_num / max(pi_num.sum(), _LOG_EPS)
            A_new = A_num / A_den[:, None]
            B_new = B_num / B_den[:, None]
            A_new = A_new / A_new.sum(axis=1, keepdims=True)
            B_new = B_new / B_new.sum(axis=1, keepdims=True)

            self.log_pi = _log(pi_new)
            self.log_A = _log(A_new)
            self.log_B = _log(B_new)
            history.append(total_ll)
            if len(history) >= 2 and abs(history[-1] - history[-2]) < tol:
                break
        return history

    def filtered_state_posterior(self, obs: np.ndarray) -> np.ndarray:
        """P(state_t | obs[0..t]) for every t -- the ONLINE posterior a
        live system could actually use (unlike the smoothed gamma used
        during fit(), which looks at the whole sequence including the
        future). Shape (T, n_states)."""
        log_alpha = self._forward_log(obs)
        return np.exp(log_alpha - _logsumexp(log_alpha, axis=1)[:, None])

    def one_step_predictive_distribution(self, state_posterior: np.ndarray) -> np.ndarray:
        """P(obs_{t+1} | obs[0..t]) given the filtered state posterior at
        t: propagate through the transition matrix, then emit."""
        next_state_dist = state_posterior @ np.exp(self.log_A)
        return next_state_dist @ np.exp(self.log_B)


def evaluate_predictive_nll(hmm: DiscreteHMM, held_out_sequences: list[np.ndarray]) -> float:
    """Average per-step negative log-likelihood of the ACTUAL next
    observation under the HMM's one-step-ahead predictive distribution,
    over every (t -> t+1) transition in every held-out sequence. This is
    the number G4's gate compares against the memoryless baseline below --
    lower means the model's predictions were closer to what actually
    happened next.
    """
    total_nll = 0.0
    count = 0
    for obs in held_out_sequences:
        if len(obs) < 2:
            continue
        posteriors = hmm.filtered_state_posterior(obs)
        for t in range(len(obs) - 1):
            predicted = hmm.one_step_predictive_distribution(posteriors[t][None, :])[0]
            total_nll += -math.log(max(predicted[obs[t + 1]], _LOG_EPS))
            count += 1
    return total_nll / count if count else float("nan")


def memoryless_baseline_nll(train_sequences: list[np.ndarray],
                            held_out_sequences: list[np.ndarray],
                            n_symbols: int) -> float:
    """H22's actual control: predict the next observation from the
    CURRENT observation alone (an empirical order-1 table on the observed
    bucket, fit on train), with no hidden state at all -- "current draft
    confidence alone," exactly as the hypothesis names it. If the HMM
    cannot beat this, hidden-state modeling is not earning its complexity.
    """
    counts = np.full((n_symbols, n_symbols), 1e-6)  # additive smoothing: an
    # unseen (obs_t, obs_{t+1}) pair in a short calibration set must not
    # get an infinite penalty on held-out data.
    for obs in train_sequences:
        for t in range(len(obs) - 1):
            counts[obs[t], obs[t + 1]] += 1
    table = counts / counts.sum(axis=1, keepdims=True)

    total_nll = 0.0
    count = 0
    for obs in held_out_sequences:
        for t in range(len(obs) - 1):
            total_nll += -math.log(max(table[obs[t], obs[t + 1]], _LOG_EPS))
            count += 1
    return total_nll / count if count else float("nan")
