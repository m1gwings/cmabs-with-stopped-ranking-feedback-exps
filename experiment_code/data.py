"""PL feedback samplers and the fixed-assortment comparison diagnostic."""

import itertools
from statistics import NormalDist

import numpy as np
from experiments_wo_stress import Feedback, Instance


class RankingBandit:
    """Save fixed preferences and sample top-m feedback by an exponential race."""

    supports_extension = True

    @classmethod
    def create_instance(cls, *, rng, theta, k, m, mnl=False, theta0=1.0):
        theta = np.asarray(theta, dtype=float)
        if (
            theta.ndim != 1
            or not np.isfinite(theta).all()
            or np.any(theta <= 0)
            or not 1 <= m <= k <= len(theta)
            or theta0 <= 0
            or not np.isfinite(theta0)
        ):
            raise ValueError("Invalid ranking bandit instance")
        if mnl and m != 1:
            raise ValueError("MNL uses winner feedback")
        return Instance(
            kind="pl_ranking",
            metadata={"k": k, "m": m, "mnl": mnl, "theta0": theta0},
            arrays={"theta": theta},
        )

    def __init__(self, *, rng, instance, theta, k, m, mnl=False, theta0=1.0):
        self.rng, self.theta, self.k, self.m, self.mnl = (
            rng,
            instance.arrays["theta"],
            k,
            m,
            mnl,
        )
        self.weights = np.append(self.theta, theta0) if mnl else self.theta

    def generate(self, action):
        action = np.asarray(action, dtype=np.int64)
        if (
            len(action) != self.k
            or len(np.unique(action)) != self.k
            or np.any(action < 0)
            or np.any(action >= len(self.theta))
        ):
            raise ValueError("Action must contain k distinct real arms")
        items = np.append(action, len(self.theta)) if self.mnl else action
        race = self.rng.exponential(1 / self.weights[items])
        order = np.argsort(race)[: self.m]
        ranks = np.full(len(self.weights), min(len(items), self.m + 1), dtype=np.int64)
        ranks[items[order]] = np.arange(1, len(order) + 1)
        return Feedback(ranks, {"action": action})

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("Unexpected environment state")


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


def exact_probabilities(theta, theta0, pivot, item, cap):
    """Enumerate PL permutations independently of the exponential sampler."""
    n = len(theta)
    weights = list(theta) + ([theta0] if theta0 > 0 else [])
    observed = wins = 0.0
    for order in itertools.permutations(range(len(weights))):
        probability = 1.0
        remaining = sum(weights)
        prefix = []
        for arm in order:
            probability *= weights[arm] / remaining
            remaining -= weights[arm]
        for arm in order:
            if arm == n or len(prefix) == cap:
                break
            prefix.append(arm)
        if pivot in prefix or item in prefix:
            observed += probability
            if item in prefix and (
                pivot not in prefix or prefix.index(item) < prefix.index(pivot)
            ):
                wins += probability
    return min(1.0, max(0.0, observed)), wins / observed


def simulate_counts(theta, theta0, pivot, item, caps, samples, rng, batch_size=20000):
    """Couple caps through the same races, keeping draws independent across rounds."""
    theta = np.asarray(theta, dtype=float)
    weights = np.append(theta, theta0) if theta0 > 0 else theta
    counts = {cap: [0, 0] for cap in caps}
    competitors = [j for j in range(len(theta)) if j not in (pivot, item)]
    for start in range(0, samples, batch_size):
        race = rng.exponential(
            1 / weights, size=(min(batch_size, samples - start), len(weights))
        )
        first = np.minimum(race[:, pivot], race[:, item])
        position = 1 + (race[:, competitors] < first[:, None]).sum(axis=1)
        before_stop = (
            first < race[:, -1] if theta0 > 0 else np.ones(len(first), dtype=bool)
        )
        item_first = race[:, item] < race[:, pivot]
        for cap in caps:
            seen = before_stop & (position <= cap)
            counts[cap][0] += int(seen.sum())
            counts[cap][1] += int((seen & item_first).sum())
    return counts


def wilson_interval(successes, trials, confidence=0.95):
    if trials == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = (
        z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def comparison_trial(
    *,
    algorithm,
    data,
    rng,
    theta,
    pivot,
    item,
    caps,
    outside_weights,
    samples_per_outside_weight,
    batch_size,
    confidence_level,
    sampling_seed,
):
    """Return comparison/win counts; caps share races within each outside weight."""
    counts = []
    streams = np.random.SeedSequence(sampling_seed).spawn(len(outside_weights))
    for theta0, stream in zip(outside_weights, streams):
        result = simulate_counts(
            theta,
            theta0,
            pivot,
            item,
            caps,
            samples_per_outside_weight,
            np.random.default_rng(stream),
            batch_size,
        )
        counts.extend(result[cap] for cap in caps)
    return {"counts": np.asarray(counts, dtype=np.int64).ravel()}
