"""MNL choice feedback sampled exactly as in the conference experiment."""

import numpy as np
from experiments_wo_stress import Feedback, Instance


class MNLChoiceBandit:
    """Own the fixed instance and data RNG, keeping preferences hidden from UIF.

    Independent exponential draws for the offered real arms and outside option
    produce a winner with probability proportional to its preference. Arm n is
    the outside option. All unreported items receive rank 2, including on a
    no-choice round, so UIF observes no real-arm comparisons on those rounds.
    """

    supports_extension = True

    @classmethod
    def create_instance(cls, *, rng, theta, k, theta0):
        theta = np.asarray(theta, dtype=float)
        if (
            theta.ndim != 1
            or not np.isfinite(theta).all()
            or np.any(theta <= 0)
            or not 2 <= k <= len(theta)
            or not np.isfinite(theta0)
            or not 0 <= theta0 <= 1
        ):
            raise ValueError("Invalid MNL instance")
        return Instance(
            kind="mnl_choice",
            metadata={"k": k, "m": 1, "theta0": theta0},
            arrays={"theta": theta},
        )

    def __init__(self, *, rng, instance, theta, k, theta0):
        self.rng = rng
        self.theta = instance.arrays["theta"]
        self.k = k
        self.theta0 = theta0
        self.weights = np.append(self.theta, theta0)

    def generate(self, action):
        action = np.asarray(action, dtype=np.int64)
        if (
            action.shape != (self.k,)
            or len(np.unique(action)) != self.k
            or np.any(action < 0)
            or np.any(action >= len(self.theta))
        ):
            raise ValueError("Action must contain k distinct real arms")
        items = np.append(action, len(self.theta)) if self.theta0 else action
        race = self.rng.exponential(1 / self.weights[items])
        winner = items[np.argmin(race)]
        ranks = np.full(len(self.weights), 2, dtype=np.int64)
        ranks[winner] = 1
        return Feedback(ranks, {"action": action})

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("Unexpected environment state")
