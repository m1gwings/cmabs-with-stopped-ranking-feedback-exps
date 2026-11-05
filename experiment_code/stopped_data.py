"""Stopped top-m PL feedback, with an explicit observed prefix length."""

import numpy as np
from experiments_wo_stress import Feedback, Instance


class StoppedRankingBandit:
    supports_extension = True

    @classmethod
    def create_instance(cls, *, rng, theta, k, theta0, m):
        theta = np.asarray(theta, dtype=float)
        if (theta.ndim != 1 or not np.isfinite(theta).all()
                or np.any(theta <= 0) or not 2 <= k <= len(theta)
                or not 1 <= m <= k or not np.isfinite(theta0)
                or not 0 <= theta0 <= 1):
            raise ValueError("Invalid stopped-ranking instance")
        return Instance(kind="stopped_ranking",
                        metadata={"k": k, "m": m, "theta0": theta0},
                        arrays={"theta": theta})

    def __init__(self, *, rng, instance, theta, k, theta0, m):
        self.rng, self.k, self.m, self.theta0 = rng, k, m, theta0
        self.theta = instance.arrays["theta"]
        self.weights = np.append(self.theta, theta0)

    def generate(self, action):
        action = np.asarray(action, dtype=np.int64)
        n = len(self.theta)
        if (action.shape != (self.k,) or len(np.unique(action)) != self.k
                or np.any(action < 0) or np.any(action >= n)):
            raise ValueError("Action must contain k distinct real arms")
        items = np.append(action, n) if self.theta0 else action
        order = items[np.argsort(self.rng.exponential(1 / self.weights[items]))]
        prefix = []
        for item in order:
            if item == n or len(prefix) == self.m:
                break
            prefix.append(item)
        ranks = np.full(n + 1, len(prefix) + 1, dtype=np.int64)
        ranks[prefix] = np.arange(1, len(prefix) + 1)
        return Feedback(ranks, {"action": action, "length": len(prefix)})

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("Unexpected environment state")
