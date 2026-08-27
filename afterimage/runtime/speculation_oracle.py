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


def entropy_of_logits(logits: np.ndarray) -> float:
    """Shannon entropy (nats) of the softmax distribution over logits --
    a cheap, ONLINE-available-before-the-target-sweep feature (H22b's own
    requirement: usable before the next target observation exists, unlike
    target_rank_under_primary which needs the target's answer to compute
    at all)."""
    shifted = logits - logits.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    return float(-(probs * np.log(np.clip(probs, _LOG_EPS_ENTROPY, None))).sum())


_LOG_EPS_ENTROPY = 1e-12  # smaller than _LOG_EPS (defined below for HMM log-space
                          # arithmetic) since this only guards one log(), not an
                          # accumulated recursion -- named separately so a future
                          # change to the HMM's epsilon does not silently change
                          # entropy's numerical floor too.


def margin_of_logits(logits: np.ndarray) -> float:
    """Top-1 minus top-2 logit gap -- another cheap pre-sweep feature.
    Small margin means the source was nearly torn between two tokens;
    large margin means it was confident, for whatever that confidence is
    worth."""
    top2 = np.partition(logits, -2)[-2:]
    return float(max(top2) - min(top2))


def js_divergence(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """Jensen-Shannon divergence (nats) between two sources' softmax
    distributions over the SAME vocabulary -- symmetric and bounded
    (unlike raw KL), which is what makes it usable as a single scalar
    P/S-disagreement feature rather than needing to pick a direction."""
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max()
        probs = np.exp(shifted)
        return probs / probs.sum()

    p, q = _softmax(logits_a), _softmax(logits_b)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / np.clip(b[mask], _LOG_EPS_ENTROPY, None))).sum())

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def approximate_js_divergence_from_topk(
        ids_a: list[int], probs_a: list[float],
        ids_b: list[int], probs_b: list[float]) -> float:
    """Jensen-Shannon divergence approximated from two SPARSE top-k
    distributions rather than the full vocabulary softmax.

    This exists because H21's reusable Stage B/C pipeline (scripts/
    run_h21_score_source.py) scores exactly ONE candidate source at a
    time -- that is what makes adding a new candidate cheap -- so no
    single stage ever has both Primary's and Scout's full logit vectors
    in memory together to compute a true divergence from. Storing full
    per-position vocabulary distributions (151936-wide for this
    project's target) to enable an exact computation later would be
    enormous; storing each source's own top-k token IDs and probability
    MASSES (already collected for coverage/rank purposes) is cheap, and
    is enough for a real, if approximate, divergence signal.

    The approximation: build the union of both sources' stored top-k
    token IDs, assign each source's recorded probability mass to the IDs
    it actually reported and 0.0 to IDs only the other source reported,
    then compute standard JS divergence over that reduced support. This
    UNDERCOUNTS true divergence whenever the two sources disagree mainly
    in their tails (outside both stored top-k sets) -- a real limitation,
    not hidden here, and the reason this function has "approximate" in
    its name rather than being called js_divergence like the exact
    version above (which needs full logits and is used where both
    sources' logits genuinely are both in memory, e.g. a future
    single-process ablation, not the default reusable pipeline).
    """
    support = sorted(set(ids_a) | set(ids_b))
    a_lookup = dict(zip(ids_a, probs_a))
    b_lookup = dict(zip(ids_b, probs_b))
    p = np.array([a_lookup.get(tok, 0.0) for tok in support])
    q = np.array([b_lookup.get(tok, 0.0) for tok in support])
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum <= 0.0 or q_sum <= 0.0:
        return 0.0  # no recorded mass to compare -- nothing to say about divergence
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / np.clip(b[mask], _LOG_EPS_ENTROPY, None))).sum())

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


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


def compute_equal_budget_metrics(rows: list[dict], total_budget: int,
                                 scout_slots: list[int]) -> dict:
    """The actual Scout question, corrected: union coverage at P-top8 vs
    S-top8 (compute_oracle_coverage_stats) lets Scout add candidate slots
    for free -- of course an 8+8=16-candidate union covers the target at
    least as often as either 8-candidate source alone. That is not
    evidence a second model is worth having; it is evidence that more
    candidates help, which nobody disputed.

    The real question holds TOTAL candidate budget fixed and asks whether
    trading Primary slots for Scout slots is worth it: coverage of
    (P_top(total_budget - s) UNION S_top(s)) for s in scout_slots, each
    compared against the s=0 baseline (P_top(total_budget) alone, no
    Scout at all). This requires each row's own primary_topk/scout_topk
    lists to be at least `total_budget` long -- shorter lists silently
    truncate the comparison, which is treated as a caller error (a
    mismatched --stored-top-k), not something to guess around here.
    """
    if not rows:
        return {"total_budget": total_budget, "points": {}, "rows": 0}
    for row in rows:
        if len(row.get("primary_topk", [])) < total_budget:
            raise ValueError(
                "row's primary_topk has %d entries, fewer than total_budget=%d "
                "-- collect traces with --stored-top-k >= total_budget" %
                (len(row.get("primary_topk", [])), total_budget))
        if len(row.get("scout_topk", [])) < total_budget:
            raise ValueError(
                "row's scout_topk has %d entries, fewer than total_budget=%d "
                "-- collect traces with --stored-top-k >= total_budget" %
                (len(row.get("scout_topk", [])), total_budget))

    def _covered(row: dict, primary_n: int, scout_n: int) -> bool:
        target = row["target_token"]
        if target in row["primary_topk"][:primary_n]:
            return True
        return scout_n > 0 and target in row["scout_topk"][:scout_n]

    baseline_hits = [_covered(row, total_budget, 0) for row in rows]
    baseline_coverage = statistics.mean(baseline_hits)

    points = {}
    for s in scout_slots:
        if s < 0 or s > total_budget:
            raise ValueError("scout_slots entries must be in [0, total_budget]")
        primary_n = total_budget - s
        hits = [_covered(row, primary_n, s) for row in rows]
        coverage = statistics.mean(hits)
        unique_scout_hits = [
            (row["target_token"] not in row["primary_topk"][:primary_n])
            and (s > 0 and row["target_token"] in row["scout_topk"][:s])
            for row in rows]
        points[s] = {
            "primary_slots": primary_n,
            "scout_slots": s,
            "coverage": coverage,
            "equal_budget_union_gain": coverage - baseline_coverage,
            "marginal_scout_coverage_per_slot": (
                (coverage - baseline_coverage) / s if s > 0 else 0.0),
            "unique_scout_hit_rate": statistics.mean(unique_scout_hits),
        }

    overlap_at_budget = []
    for row in rows:
        primary_set = set(row["primary_topk"][:total_budget])
        scout_set = set(row["scout_topk"][:total_budget])
        union = primary_set | scout_set
        overlap_at_budget.append(len(primary_set & scout_set) / len(union) if union else 1.0)

    return {
        "total_budget": total_budget,
        "baseline_coverage": baseline_coverage,
        "points": points,
        "primary_scout_overlap": statistics.mean(overlap_at_budget) if overlap_at_budget else None,
        "rows": len(rows),
    }


def compute_sustained_rescue_depth(traces: list[dict], k: int,
                                   depths: tuple[int, ...] = (1, 2, 4, 8)) -> dict:
    """How far a Scout rescue actually extends, not just whether the
    single position after a Primary miss is covered. Teacher-forced on
    the TARGET's own real trajectory (the traces already are, since both
    Primary and Scout were scored against the target's actual next
    token at every position) -- so this measures sustained coverage along
    the real path, which is the oracle a tree/candidate speculation
    scheme that verifies every node against the true continuation
    actually cares about. It is not free-running Scout-generates-its-own-
    tokens agreement length (a different, harder-to-define quantity that
    needs new generation, not a re-read of this data) -- see
    docs/SPECULATION_TREE_RESEARCH.md's H21 section for that distinction.

    For every position where Primary misses (target rank > k) but Scout
    covers it (target rank <= k), the "rescue depth" is the number of
    consecutive following positions, within the same trace, that Scout
    also covers -- capped at the trace's own remaining length. depth=1
    means only the miss position itself was rescued and the immediately
    next real-trajectory position is not (or the trace ends there).

    Every qualifying position is counted as its own independent
    opportunity with its own forward-looking depth, even if it falls
    inside another opportunity's rescued run (two adjacent positions that
    both independently miss-then-rescue produce two entries, with
    overlapping but different depths). This is deliberate, not
    double-counting: each is a genuinely separate point where a tree
    expansion would have needed Scout's guidance, and "how far can I
    trust Scout starting HERE" is a different question at each of them.
    """
    lengths: list[int] = []
    for trace in traces:
        rows = trace["rows"]
        for i, row in enumerate(rows):
            primary_covers = row["target_rank_under_primary"] <= k
            scout_covers = row["target_rank_under_scout"] <= k
            if primary_covers or not scout_covers:
                continue  # not a rescue opportunity, or Scout didn't rescue it
            depth = 0
            for j in range(i, len(rows)):
                if rows[j]["target_rank_under_scout"] <= k:
                    depth += 1
                else:
                    break
            lengths.append(depth)

    if not lengths:
        return {"k": k, "rescue_opportunities": 0, "mean_depth": None,
               "median_depth": None, "depth_probabilities": {d: None for d in depths}}

    return {
        "k": k,
        "rescue_opportunities": len(lengths),
        "mean_depth": statistics.mean(lengths),
        "median_depth": statistics.median(lengths),
        "depth_probabilities": {
            d: sum(length >= d for length in lengths) / len(lengths) for d in depths},
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


def constant_frequency_baseline_nll(train_sequences: list[np.ndarray],
                                    held_out_sequences: list[np.ndarray],
                                    n_symbols: int) -> float:
    """B0: predict the next observation from its unconditional frequency
    alone, ignoring every previous observation entirely. The floor every
    other baseline (and the HMM) must clear -- if B1 (current observation)
    cannot beat B0, the "current confidence" signal itself carries no
    temporal information, before the HMM is even considered.
    """
    counts = np.full(n_symbols, 1e-6)
    for obs in train_sequences:
        for value in obs:
            counts[value] += 1
    dist = counts / counts.sum()

    total_nll, count = 0.0, 0
    for obs in held_out_sequences:
        for t in range(len(obs) - 1):
            total_nll += -math.log(max(dist[obs[t + 1]], _LOG_EPS))
            count += 1
    return total_nll / count if count else float("nan")


@dataclasses.dataclass
class MultinomialLogisticRegression:
    """A from-scratch softmax classifier: no new project dependency for
    what is a small, well-understood model, matching DiscreteHMM's own
    precedent of implementing standard statistics locally rather than
    pulling in scikit-learn for one algorithm. Trained by full-batch
    gradient descent on the cross-entropy loss with L2 regularization
    (the closed-form IRLS update is not worth the complexity here; the
    feature counts and dataset sizes this project's H22b baselines use
    are small enough that plain gradient descent converges in well under
    a second).

    Exists specifically for H22 baselines B2/B3/B4 (cheap-feature
    prediction of the next disagreement bucket): B2 uses Primary-only
    features (entropy, margin), B3 adds Scout features and P/S
    divergence, B4 adds recent observation history on top of B3. All
    three are the SAME model class with different input feature columns
    -- the ladder's point is isolating which features carry the signal,
    not comparing different algorithms.
    """
    n_features: int
    n_classes: int
    weights: np.ndarray  # (n_features + 1, n_classes), last row is bias

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, n_classes: int,
           learning_rate: float = 0.1, l2: float = 1e-3,
           max_iterations: int = 500, seed: int = 0) -> "MultinomialLogisticRegression":
        n_samples, n_features = X.shape
        X_bias = np.hstack([X, np.ones((n_samples, 1))])
        rng = np.random.default_rng(seed)
        weights = rng.normal(scale=0.01, size=(n_features + 1, n_classes))
        y_onehot = np.eye(n_classes)[y]
        for _ in range(max_iterations):
            logits = X_bias @ weights
            logits -= logits.max(axis=1, keepdims=True)  # numerical stability
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            grad = X_bias.T @ (probs - y_onehot) / n_samples
            grad[:-1] += l2 * weights[:-1]  # do not regularize the bias row
            weights -= learning_rate * grad
        return cls(n_features, n_classes, weights)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        X_bias = np.hstack([X, np.ones((n_samples, 1))])
        logits = X_bias @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)


def logistic_regression_baseline_nll(
        X_train: np.ndarray, y_train: np.ndarray,
        X_held_out: np.ndarray, y_held_out: np.ndarray, n_classes: int,
        seed: int = 0) -> float:
    """Fits MultinomialLogisticRegression on (X_train, y_train) -- y is
    the discretized NEXT observation's bucket, X is whatever features are
    available BEFORE that next observation (this is what makes it a fair
    online-predictability baseline, unlike feeding it the next
    observation's own rank). Returns held-out NLL in the same units as
    evaluate_predictive_nll/memoryless_baseline_nll, so all of B0-B5 are
    directly comparable on one axis.
    """
    if len(X_train) == 0 or len(X_held_out) == 0:
        return float("nan")
    model = MultinomialLogisticRegression.fit(X_train, y_train, n_classes, seed=seed)
    probs = model.predict_proba(X_held_out)
    row_probs = probs[np.arange(len(y_held_out)), y_held_out]
    return float(-np.mean(np.log(np.clip(row_probs, _LOG_EPS, None))))


def select_n_states_by_held_out_nll(
        train_sequences: list[np.ndarray], validation_sequences: list[np.ndarray],
        n_symbols: int, candidate_n_states: tuple[int, ...] = (2, 3, 4, 5, 6),
        restarts: int = 3, max_iterations: int = 100, seed: int = 0) -> dict:
    """Selects the hidden-state count on a VALIDATION split, never on the
    final held-out test split -- state count is a hyperparameter of the
    model, and choosing it by looking at the number that makes the test
    result look best is the exact goalpost-moving the pre-registered
    gates in docs/SPECULATION_TREE_RESEARCH.md exist to prevent. Reports
    every candidate's validation NLL so the selection is auditable, not
    just its winner.
    """
    scored = []
    for n_states in candidate_n_states:
        best_ll, best_hmm = -np.inf, None
        for offset in range(restarts):
            hmm = DiscreteHMM.random_init(n_states, n_symbols, seed=seed + offset)
            history = hmm.fit(train_sequences, max_iterations=max_iterations)
            if history and history[-1] > best_ll:
                best_ll, best_hmm = history[-1], hmm
        validation_nll = evaluate_predictive_nll(best_hmm, validation_sequences)
        scored.append({"n_states": n_states, "validation_nll": validation_nll,
                       "train_log_likelihood": best_ll})
    finite = [entry for entry in scored if not math.isnan(entry["validation_nll"])]
    selected = min(finite, key=lambda entry: entry["validation_nll"]) if finite else None
    return {
        "candidates": scored,
        "selected_n_states": selected["n_states"] if selected else None,
    }


def grouped_k_fold(n_groups: int, k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Splits n_groups group indices (one group = one whole trajectory,
    never split across folds -- splitting positions within a trajectory
    across train/test would leak temporal information across the
    boundary, the same leakage split_by_trace already guards against for
    a single train/held-out split) into k folds. Returns a list of
    (train_group_indices, test_group_indices) pairs.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    if n_groups < k:
        raise ValueError("n_groups (%d) must be >= k (%d)" % (n_groups, k))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_groups)
    folds = [order[i::k].tolist() for i in range(k)]
    splits = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = [idx for j in range(k) if j != i for idx in folds[j]]
        splits.append((train_idx, test_idx))
    return splits


def bootstrap_nll_difference_ci(paired_differences: list[float], n_resamples: int = 2000,
                                seed: int = 0, confidence: float = 0.95) -> dict:
    """Bootstrap confidence interval on the mean of paired
    (baseline_nll - model_nll) differences (one per CV fold, or per
    trajectory), used because a single point estimate of "the HMM won by
    X" says nothing about whether that margin could plausibly be zero
    given the fold-to-fold variance -- see G4a/G4b's own CI-excludes-zero
    requirement in docs/SPECULATION_TREE_RESEARCH.md. Positive values
    mean the model beat the baseline (lower NLL is better, so
    baseline - model > 0 favors the model).
    """
    if not paired_differences:
        return {"mean": None, "ci_low": None, "ci_high": None, "excludes_zero": None,
               "n_resamples": n_resamples, "n_paired_observations": 0}
    values = np.asarray(paired_differences, dtype=float)
    rng = np.random.default_rng(seed)
    resample_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_resamples)])
    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(resample_means, alpha))
    ci_high = float(np.quantile(resample_means, 1.0 - alpha))
    return {
        "mean": float(values.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "n_paired_observations": len(values),
    }


def minimum_positions_for_hmm(n_states: int, n_symbols: int,
                              observations_per_free_parameter: int = 20) -> int:
    """A derived, not hand-picked, minimum total observation count for
    fitting an n_states/n_symbols DiscreteHMM. Free parameters: n_states
    rows of a (n_symbols-1)-dimensional simplex for emissions, n_states
    rows of a (n_states-1)-dimensional simplex for transitions, and one
    (n_states-1)-dimensional simplex for the initial distribution.
    observations_per_free_parameter=20 is a conventional rule-of-thumb
    floor for stable maximum-likelihood estimation, not a guarantee of
    a good fit -- more data is always better, this is only the point
    below which the fit should not be trusted at all.
    """
    free_parameters = (n_states * (n_symbols - 1) + n_states * (n_states - 1)
                       + (n_states - 1))
    return free_parameters * observations_per_free_parameter
