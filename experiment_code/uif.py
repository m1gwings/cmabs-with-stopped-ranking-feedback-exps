"""Real-pivot UIF; journal algs/top_k_uif.tex and algs/subroutines.tex.

Adapted from the conference implementation at 0764019ad043b05ec2c9a58306da0de591c43fe5.
"""

import numpy as np


def _comparison_counts(lower, upper):
    """Count strict certified comparisons; close cross-pivot chains exactly."""
    if len(lower) == 1:
        # An interval order is already transitive: L_i > U_j >= L_j > U_h.
        lo, hi = lower[0], upper[0]
        worse = np.searchsorted(np.sort(hi), lo, side="left")
        better = len(lo) - np.searchsorted(np.sort(lo), hi, side="right")
        return worse, better
    precedes = np.any(lower[:, :, None] > upper[:, None, :], axis=0)
    for j in range(precedes.shape[0]):
        precedes |= precedes[:, j, None] & precedes[j, None, :]
    np.fill_diagonal(precedes, False)
    return precedes.sum(axis=1), precedes.sum(axis=0)


class RealPivotUIF:
    """Journal Algorithm 1 and its subroutines, using pre-round A, U and P.

    Rows of counts/wins correspond to real pivots. No-choice is never a pivot.
    Confidence radii are sqrt(log(2*n*T/delta)/N). Guards follow
    Appendix C, using ascending arm order; activations are counted in checkpoints.
    """

    supports_extension = False  # Confidence bounds depend on the fixed horizon.

    def __init__(self, *, rng, n, k, m=1, horizon=1000000, delta=0.05):
        if not (2 <= k <= n and 1 <= m <= k and 0 < delta <= 1 and horizon >= 1):
            raise ValueError("Invalid UIF parameters")
        self.rng, self.n, self.k, self.m = rng, n, k, m
        self.log_bonus = np.log(2 * n * horizon / delta)
        self.counts = np.zeros((n, n))
        self.wins = np.zeros((n, n))
        np.fill_diagonal(self.counts, np.inf)
        self.accepted = np.zeros(n, dtype=bool)
        self.unknown = np.ones(n, dtype=bool)
        self.pivot = int(rng.integers(n))
        self.round = 0
        self.completed_round = 0
        self.guard_events = 0

    def act(self, context=None):
        accepted = np.flatnonzero(self.accepted)
        if len(accepted) == self.k:
            return accepted
        pivots = np.union1d(accepted, [self.pivot])
        candidates = np.setdiff1d(np.flatnonzero(self.unknown), [self.pivot])
        size = self.k - len(pivots)
        if not 0 <= size <= len(candidates):
            raise RuntimeError("UIF invariant failed: cannot sample a size-k action")
        sampled = self.rng.choice(candidates, size=size, replace=False)
        return np.sort(np.concatenate((pivots, sampled)))

    def observe(self, action, feedback):
        self.round += 1
        if self.completed_round:
            return
        old_a, old_u, old_p = self.accepted.copy(), self.unknown.copy(), self.pivot
        pivots = np.union1d(np.flatnonzero(old_a), [old_p])
        items = np.asarray(action)
        ranks = np.asarray(feedback)
        pi = np.ix_(pivots, items)
        comparable = ranks[pivots, None] != ranks[items]
        self.counts[pi] += comparable
        self.wins[pi] += (ranks[items] < ranks[pivots, None]) & comparable

        unknown = np.flatnonzero(old_u)
        counts = self.counts[np.ix_(pivots, unknown)]
        means = np.full(counts.shape, 0.5)
        observed = (counts > 0) & np.isfinite(counts)
        np.divide(self.wins[np.ix_(pivots, unknown)], counts, out=means, where=observed)
        with np.errstate(divide="ignore"):
            bonus = np.sqrt(self.log_bonus / counts)
        lower, upper = means - bonus, means + bonus
        worse, better = _comparison_counts(lower, upper)
        reject = better >= self.k - old_a.sum()
        accept = (~reject) & (worse >= (old_a | old_u).sum() - self.k)
        # The displayed pseudocode omits the Appendix C cardinality guards.
        for arm, rejected, accepted in zip(unknown, reject, accept):
            if rejected:
                if (self.accepted | self.unknown).sum() > self.k:
                    self.unknown[arm] = False
                else:
                    self.guard_events += 1
            elif accepted:
                if self.accepted.sum() < self.k:
                    self.accepted[arm], self.unknown[arm] = True, False
                else:
                    self.guard_events += 1

        pivot_lower = lower[np.flatnonzero(pivots == old_p)[0]]
        better = pivot_lower > 0.5
        if old_a.any():
            # All of U_t participates, including arms just removed from U.
            for arm in unknown[better]:
                if self.accepted[arm]:
                    continue
                if self.accepted.sum() < self.k:
                    self.accepted[arm], self.unknown[arm] = True, False
                else:
                    self.guard_events += 1
        elif better.any():
            best = better & (pivot_lower == np.max(pivot_lower[better]))
            self.pivot = int(self.rng.choice(unknown[best]))

        if self.accepted.sum() > self.k or np.any(self.accepted & self.unknown):
            raise RuntimeError("UIF accept/reject invariant failed")
        if self.accepted.sum() == self.k:
            self.unknown[:] = False
            self.completed_round = self.round

    def state_dict(self):
        return {
            key: getattr(self, key)
            for key in (
                "counts",
                "wins",
                "accepted",
                "unknown",
                "pivot",
                "round",
                "completed_round",
                "guard_events",
            )
        }

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value.copy() if isinstance(value, np.ndarray) else value)
