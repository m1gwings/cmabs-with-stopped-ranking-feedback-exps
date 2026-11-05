"""Unchanged NeurIPS AOA-RB baselines.

Source: gianmarcogenalti/combinatorial-bandits-relative-feedback,
commit 0764019ad043b05ec2c9a58306da0de591c43fe5, experiment_code/algorithms.py.
Original experiment authors: Cristiano Migali, Gianmarco Genalti,
Alberto Maria Metelli, and Marco Mussi.
"""

import numpy as np


class AOARB:
    """Saha-Gaillard pairwise-ratio optimism with optional intermediate pivots.

    Saturated probability bounds yield infinite ratios.
    The no-choice ratio to itself is one.
    """

    supports_extension = False

    def __init__(self, *, rng, n, k, horizon=1000000, adaptive=True, x_scale=2.0):
        self.rng, self.n, self.k, self.adaptive = rng, n, k, adaptive
        self.x = x_scale * np.log(max(horizon, 2))
        self.wins = np.zeros((n + 1, n + 1))

    def act(self, context=None):
        counts = self.wins + self.wins.T
        mean = np.divide(
            self.wins, counts, out=np.full(counts.shape, 0.5), where=counts > 0
        )
        safe_counts = np.maximum(counts, 1)
        upper = (
            mean
            + np.sqrt(2 * mean * (1 - mean) * self.x / safe_counts)
            + 3 * self.x / safe_counts
        )
        upper[counts == 0] = 1.0
        ratios = np.divide(
            upper, 1 - upper, out=np.full(upper.shape, np.inf), where=upper < 1
        )
        np.fill_diagonal(ratios, 1.0)
        scores = ratios[: self.n, self.n]
        if self.adaptive:
            scores = np.min(ratios[: self.n] * ratios[:, self.n], axis=1)
        permutation = self.rng.permutation(self.n)
        order = np.argsort(-scores[permutation], kind="stable")
        return np.sort(permutation[order[: self.k]])

    def observe(self, action, feedback):
        items = np.append(action, self.n)
        winner = items[np.argmin(np.asarray(feedback)[items])]
        self.wins[winner, items[items != winner]] += 1

    def state_dict(self):
        return {"wins": self.wins}

    def load_state_dict(self, state):
        self.wins = np.array(state["wins"], copy=True)
