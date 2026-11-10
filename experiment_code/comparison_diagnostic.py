"""Fixed-assortment comparison diagnostic; exact PL enumeration and simulation."""

import itertools
from statistics import NormalDist

import numpy as np


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
):
    """Return comparison/win counts; caps share races within each outside weight."""
    counts = []
    for theta0 in outside_weights:
        result = simulate_counts(
            theta,
            theta0,
            pivot,
            item,
            caps,
            samples_per_outside_weight,
            rng,
            batch_size,
        )
        counts.extend(result[cap] for cap in caps)
    return {"counts": np.asarray(counts, dtype=np.int64).ravel()}


class ComparisonCounts:
    """Expose the trial's count vector to EWS analysis without averaging trials."""

    required_fields = ("counts",)

    def compute(self, result):
        from experiments_wo_stress import MetricResult

        counts = np.asarray(result["counts"][-1]).ravel()
        return MetricResult(x=np.arange(len(counts)), values=counts)
