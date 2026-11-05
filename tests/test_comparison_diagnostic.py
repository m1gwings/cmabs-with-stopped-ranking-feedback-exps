import unittest
import numpy as np
from experiment_code.comparison_diagnostic import exact_probabilities, simulate_counts
from experiment_code.stopped_data import StoppedRankingBandit


class DiagnosticTests(unittest.TestCase):
    def test_exact_probability_endpoints_and_pair_mean(self):
        theta=[1.,.8,.5,.2,.1]
        for theta0 in [0.,.1,1.]:
            for cap in [1,2,3,4,5]:
                frequency,mean=exact_probabilities(theta,theta0,3,4,cap)
                self.assertAlmostEqual(mean,1/3)
                if cap==1:
                    self.assertAlmostEqual(frequency,.3/(sum(theta)+theta0))
                if cap>=4:
                    self.assertAlmostEqual(frequency,.3/(.3+theta0))

    def test_vectorized_counts_match_existing_environment(self):
        theta=[1.,.8,.5,.2,.1]
        for theta0 in [0.,.3,1.]:
            counts=simulate_counts(theta,theta0,3,4,[1,2,3,4],300,np.random.default_rng(71),batch_size=77)
            for cap in [1,2,3,4]:
                params=dict(theta=theta,k=5,theta0=theta0,m=cap)
                instance=StoppedRankingBandit.create_instance(rng=None,**params)
                env=StoppedRankingBandit(rng=np.random.default_rng(71),instance=instance,**params)
                seen=wins=0
                for _ in range(300):
                    ranks=env.generate(np.arange(5)).value
                    seen+=int(ranks[3]!=ranks[4]);wins+=int(ranks[4]<ranks[3])
                self.assertEqual(counts[cap],[seen,wins])
