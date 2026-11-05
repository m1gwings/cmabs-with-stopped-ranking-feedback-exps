"""Scientific feedback, real pivots, decision rules, and checkpoint recovery."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from experiments_wo_stress import load_config, run_experiment
from experiments_wo_stress.storage import iter_completed_runs

from experiment_code.baselines import AOARB
from experiment_code.data import MNLChoiceBandit
from experiment_code.uif import RealPivotUIF, _comparison_counts

ROOT = Path(__file__).resolve().parents[1]


def make_data(theta=(1.0, 0.5, 0.2, 0.1), k=3, theta0=0.3, seed=2):
    params = {"theta": list(theta), "k": k, "theta0": theta0}
    instance = MNLChoiceBandit.create_instance(rng=np.random.default_rng(1), **params)
    return MNLChoiceBandit(rng=np.random.default_rng(seed), instance=instance, **params)


def advance(learner, data):
    action = learner.act()
    feedback = data.generate(action).value
    learner.observe(action, feedback)
    return action.copy(), feedback.copy()


class FeedbackTests(unittest.TestCase):
    """Validate the choice law and the information available to the learner."""

    def test_mnl_frequencies_include_outside_choice(self):
        data = make_data()
        counts = np.zeros(5)
        for _ in range(30000):
            counts[np.argmin(data.generate([0, 1, 2]).value)] += 1
        expected = np.array([1.0, 0.5, 0.2, 0.0, 0.3]) / 2
        np.testing.assert_allclose(counts / counts.sum(), expected, atol=0.01)

    def test_zero_outside_weight_always_reports_a_real_arm(self):
        data = make_data(theta0=0)
        for _ in range(100):
            self.assertIn(int(np.argmin(data.generate([0, 1, 2]).value)), [0, 1, 2])

    def test_no_choice_never_updates_real_arm_comparisons(self):
        learner = RealPivotUIF(rng=np.random.default_rng(3), n=4, k=3)
        self.assertLess(learner.pivot, 4)
        self.assertEqual(learner.counts.shape, (4, 4))
        before = learner.counts.copy()
        learner.observe(learner.act(), np.array([2, 2, 2, 2, 1]))
        np.testing.assert_array_equal(learner.counts, before)
        self.assertFalse(learner.wins.any())

    def test_reported_winner_breaks_only_observed_pairs(self):
        learner = RealPivotUIF(rng=np.random.default_rng(3), n=4, k=3)
        learner.pivot = 1
        learner.observe(np.array([0, 1, 2]), np.array([1, 2, 2, 2, 2]))
        self.assertEqual(learner.counts[1, 0], 1)
        self.assertEqual(learner.wins[1, 0], 1)
        self.assertEqual(learner.counts[1, 2], 0)


class DecisionTests(unittest.TestCase):
    """Check acceptance and the exact transitive comparison relation."""

    def test_single_pivot_fast_counts_equal_literal_transitive_closure(self):
        rng = np.random.default_rng(5)
        for n in (1, 10, 30):
            for _ in range(30):
                lower = rng.normal(size=n)
                upper = lower + rng.exponential(size=n)
                lower[0] = upper[0] = 0.5
                if n > 1:
                    lower[1], upper[1] = -np.inf, np.inf
                edges = lower[:, None] > upper[None, :]
                for j in range(n):
                    edges |= edges[:, j, None] & edges[j, None, :]
                worse, better = _comparison_counts(lower[None, :], upper[None, :])
                np.testing.assert_array_equal(worse, edges.sum(axis=1))
                np.testing.assert_array_equal(better, edges.sum(axis=0))

    def test_cross_pivot_chain_certifies_acceptance(self):
        learner = RealPivotUIF(rng=np.random.default_rng(3), n=5, k=3)
        learner.pivot = 0
        learner.accepted[:2] = True
        learner.unknown[:2] = False
        learner.counts[0, [2, 3]] = 100000
        learner.wins[0, [2, 3]] = [40000, 20000]
        learner.counts[1, [3, 4]] = 100000
        learner.wins[1, [3, 4]] = [40000, 20000]
        learner.observe(np.array([0, 1, 2]), np.array([2, 2, 1, 2, 2, 2]))
        self.assertEqual(learner.accepted.tolist(), [True, True, True, False, False])
        self.assertFalse(learner.unknown.any())

    def test_pivot_shortcut_accepts_last_arm_and_stops(self):
        learner = RealPivotUIF(rng=np.random.default_rng(3), n=4, k=2)
        learner.pivot = 0
        learner.accepted[0], learner.unknown[0] = True, False
        learner.counts[0, 1], learner.wins[0, 1] = 100000, 80000
        learner.observe(np.array([0, 1]), np.array([2, 1, 2, 2, 2]))
        np.testing.assert_array_equal(learner.act(), [0, 1])
        self.assertEqual(learner.completed_round, 1)
        self.assertFalse(learner.unknown.any())


class RecoveryTests(unittest.TestCase):
    """Prove exact replay for component state and for the complete EWS pipeline."""

    def test_checkpoint_replays_actions_and_feedback_for_every_algorithm(self):
        for cls, params in (
            (RealPivotUIF, {}),
            (AOARB, {"adaptive": True}),
            (AOARB, {"adaptive": False}),
        ):
            with self.subTest(algorithm=cls.__name__, params=params):
                data = make_data()
                learner = cls(rng=np.random.default_rng(3), n=4, k=3, **params)

                for _ in range(100):
                    advance(learner, data)
                state = copy.deepcopy(learner.state_dict())
                algorithm_rng = copy.deepcopy(learner.rng.bit_generator.state)
                data_rng = copy.deepcopy(data.rng.bit_generator.state)
                expected = [advance(learner, data) for _ in range(200)]
                learner = cls(rng=np.random.default_rng(999), n=4, k=3, **params)
                learner.load_state_dict(state)
                learner.rng.bit_generator.state = algorithm_rng
                data.rng.bit_generator.state = data_rng
                for action, feedback in expected:
                    actual_action, actual_feedback = advance(learner, data)
                    np.testing.assert_array_equal(actual_action, action)
                    np.testing.assert_array_equal(actual_feedback, feedback)

    def test_ews_resumption_and_worker_count_preserve_every_record(self):
        settings = yaml.safe_load((ROOT / "experiment.yml").read_text())
        settings["runs"][0]["repetitions"] = 1
        settings["runs"][0]["budget"]["steps"] = 300
        settings["runs"][0]["grid"]["data.params.theta0"] = [0.01]
        settings["recording"]["retention"] = "keep"
        settings["execution"]["checkpoint_steps"] = 50
        runtime = ROOT / ".runtime"
        runtime.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as directory:
            directory = Path(directory)
            path = directory / "verification.yml"
            path.write_text(yaml.safe_dump(settings))
            config = load_config(path)
            direct, resumed = directory / "direct", directory / "resumed"
            report = run_experiment(config, direct, workers=1)
            self.assertEqual((report.completed, report.failed), (3, 0), report.errors)
            report = run_experiment(config, resumed, workers=1, max_steps=137)
            self.assertEqual((report.paused, report.failed), (3, 0), report.errors)
            report = run_experiment(config, resumed, workers=2)
            self.assertEqual((report.completed, report.failed), (3, 0), report.errors)
            expected = {
                spec.run_id: result for spec, result in iter_completed_runs(direct)
            }
            actual = {
                spec.run_id: result for spec, result in iter_completed_runs(resumed)
            }
            self.assertEqual(expected.keys(), actual.keys())
            for run_id, result in expected.items():
                for field in result:
                    np.testing.assert_array_equal(result[field], actual[run_id][field])
                np.testing.assert_array_equal(actual[run_id]["step"], np.arange(1, 301))
            report = run_experiment(config, resumed, workers=1)
            self.assertEqual((report.skipped, report.failed), (3, 0), report.errors)


if __name__ == "__main__":
    unittest.main()
