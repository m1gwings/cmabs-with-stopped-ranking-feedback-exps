#!/usr/bin/env python3
"""Run the comparison-frequency/conditional-mean diagnostic and exact checks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_code.comparison_diagnostic import exact_probabilities, simulate_counts, wilson_interval


def run(config_path, destination):
    settings=json.loads(config_path.read_text())
    theta=settings['theta']; pivot=settings['pivot']; item=settings['item']
    caps=settings['caps']; samples=settings['samples_per_outside_weight']
    assert 0 <= pivot < len(theta) and 0 <= item < len(theta) and pivot != item
    assert all(0 < x <= 1 for x in theta) and max(theta)==1
    assert all(1 <= m <= len(theta) for m in caps)
    assert samples > 0 and all(0 <= x <= 1 for x in settings['outside_weights'])
    true_mean=theta[item]/(theta[item]+theta[pivot])
    streams=np.random.SeedSequence(settings['seed']).spawn(len(settings['outside_weights']))
    rows=[]
    for theta0, stream in zip(settings['outside_weights'], streams):
        counts=simulate_counts(theta,theta0,pivot,item,caps,samples,np.random.default_rng(stream),settings['batch_size'])
        for cap in caps:
            observed,wins=counts[cap]
            exact_frequency,exact_mean=exact_probabilities(theta,theta0,pivot,item,cap)
            assert abs(exact_mean-true_mean)<1e-12
            if cap>=len(theta)-1:
                assert abs(exact_frequency-(theta[pivot]+theta[item])/(theta[pivot]+theta[item]+theta0))<1e-12
            frequency=observed/samples; mean=wins/observed
            frequency_se=np.sqrt(exact_frequency*(1-exact_frequency)/samples)
            mean_se=np.sqrt(true_mean*(1-true_mean)/observed)
            assert abs(frequency-exact_frequency)<=6*frequency_se+1/samples
            assert abs(mean-true_mean)<=6*mean_se+1/observed
            fl,fu=wilson_interval(observed,samples,settings['confidence_level'])
            ml,mu=wilson_interval(wins,observed,settings['confidence_level'])
            rows.append(dict(theta0=theta0,m=cap,samples=samples,comparisons=observed,wins=wins,
                frequency=frequency,frequency_ci_lower=fl,frequency_ci_upper=fu,
                conditional_mean=mean,mean_ci_lower=ml,mean_ci_upper=mu,
                exact_frequency=exact_frequency,exact_conditional_mean=exact_mean))
        print(f'theta0={theta0:g}: all caps completed and exact-probability checks passed',flush=True)
    destination.mkdir(parents=True,exist_ok=True)
    output=destination/'comparison_diagnostic.csv'
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    provenance=dict(settings,independent_rankings=samples*len(streams),numpy_version=np.__version__,
        cap_coupling='Caps share exponential races within each outside weight; outside weights use separate SeedSequence child streams.',
        intervals='Pointwise Wilson 95% binomial intervals; conditional-mean trials are observed comparisons.',
        validation='Exact PL permutation enumeration; pair mean checked analytically; all Monte Carlo estimates within six binomial standard errors of exact values.',
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        files={output.name:hashlib.sha256(output.read_bytes()).hexdigest()},
        source_modules_sha256={p:hashlib.sha256((Path(__file__).resolve().parents[1]/p).read_bytes()).hexdigest() for p in ['scripts/run_comparison_diagnostic.py','experiment_code/comparison_diagnostic.py']})
    (destination/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args();run(args.config,args.destination)
