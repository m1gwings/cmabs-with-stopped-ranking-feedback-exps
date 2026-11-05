"""Stopped-feedback laws and exact recovery of the new EWS study."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from experiments_wo_stress import load_config, run_experiment
from experiments_wo_stress.storage import iter_completed_runs
from experiment_code.stopped_data import StoppedRankingBandit
from experiment_code.data import MNLChoiceBandit

ROOT = Path(__file__).resolve().parents[1]


def data(m, theta0, seed=7):
    params = dict(theta=[1., .7, .4, .2], k=3, theta0=theta0, m=m)
    instance = StoppedRankingBandit.create_instance(rng=None, **params)
    return StoppedRankingBandit(rng=np.random.default_rng(seed), instance=instance, **params)


class StoppedTests(unittest.TestCase):
    def test_mnl_real_comparisons_match_existing_sampler_exactly(self):
        for theta0 in [0., .3, 1.]:
            stopped = data(1, theta0)
            params = dict(theta=[1., .7, .4, .2], k=3, theta0=theta0)
            instance = MNLChoiceBandit.create_instance(rng=None, **params)
            original = MNLChoiceBandit(rng=np.random.default_rng(7), instance=instance, **params)
            for _ in range(300):
                a=stopped.generate([0,1,2]).value[:4]
                b=original.generate([0,1,2]).value[:4]
                np.testing.assert_array_equal(a[:,None] < a, b[:,None] < b)

    def test_stopped_pair_law_and_empty_probability(self):
        for m in [1,2,3]:
            env=data(m, .6); observed=wins=empty=0
            for _ in range(15000):
                r=env.generate([0,1,2]).value
                observed += r[0] != r[1]
                wins += r[0] < r[1]
                empty += np.all(r[:3] == r[0])
            self.assertAlmostEqual(wins / observed, 1 / 1.7, delta=.02)
            self.assertAlmostEqual(empty / 15000, .6 / 2.7, delta=.015)

    def test_full_and_penultimate_caps_reveal_identical_pairs(self):
        for theta0 in [0., .3, 1.]:
            partial, full = data(2,theta0), data(3,theta0)
            for _ in range(300):
                a=partial.generate([0,1,2]).value[:3]
                b=full.generate([0,1,2]).value[:3]
                np.testing.assert_array_equal(a[:,None] < a,b[:,None] < b)

    def test_direct_and_resumed_study_records_match(self):
        settings=yaml.safe_load((ROOT/'experiment_stopped.yml').read_text())
        for run in settings['runs']:
            run['repetitions']=1
            run['budget']['steps']=100
            run['grid']['data.params.theta0']=[.3]
        settings['recording']['retention']='keep'
        settings['execution']['checkpoint_steps']=20
        runtime=ROOT/'.runtime'; runtime.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as directory:
            directory=Path(directory); config_path=directory/'test.yml'
            config_path.write_text(yaml.safe_dump(settings)); config=load_config(config_path)
            direct,resumed=directory/'direct',directory/'resumed'
            report=run_experiment(config,direct,workers=1)
            self.assertEqual((report.completed,report.failed),(4,0),report.errors)
            report=run_experiment(config,resumed,workers=1,max_steps=37)
            self.assertEqual((report.paused,report.failed),(4,0),report.errors)
            report=run_experiment(config,resumed,workers=2)
            self.assertEqual((report.completed,report.failed),(4,0),report.errors)
            expected={spec.run_id:result for spec,result in iter_completed_runs(direct)}
            actual={spec.run_id:result for spec,result in iter_completed_runs(resumed)}
            self.assertEqual(expected.keys(),actual.keys())
            for run_id,result in expected.items():
                for field in result:
                    np.testing.assert_array_equal(result[field],actual[run_id][field])

if __name__ == '__main__':
    unittest.main()
