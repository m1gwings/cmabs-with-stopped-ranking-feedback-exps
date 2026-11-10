#!/usr/bin/env python3
"""Validate diagnostic counts and generate native plots with exact references."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_code.comparison_diagnostic import exact_probabilities, wilson_interval

COLORS=['blue!75!black','orange!90!black','teal!80!black','violet!85!black']
MARKS=['*','square*','triangle*','diamond*']


def generate(source):
    provenance=json.loads((source/'provenance.json').read_text())
    data=source/'comparison_diagnostic.csv'
    assert hashlib.sha256(data.read_bytes()).hexdigest()==provenance['files'][data.name]
    rows=list(csv.DictReader(data.open()))
    assert len(rows)==len(provenance['caps'])*len(provenance['outside_weights'])
    assert {(int(r['m']),float(r['theta0'])) for r in rows}=={(m,t) for m in provenance['caps'] for t in provenance['outside_weights']}
    theta=provenance['theta']; pivot=provenance['pivot']; item=provenance['item']
    for row in rows:
        n=int(row['samples']); seen=int(row['comparisons']); wins=int(row['wins'])
        assert n==provenance['samples_per_outside_weight'] and 0<=wins<=seen<=n
        frequency,mean=exact_probabilities(theta,float(row['theta0']),pivot,item,int(row['m']))
        expected={'frequency':seen/n,'conditional_mean':wins/seen,'exact_frequency':frequency,'exact_conditional_mean':mean}
        for prefix,successes,trials in [('frequency',seen,n),('mean',wins,seen)]:
            lower,upper=wilson_interval(successes,trials,provenance['confidence_level'])
            expected[prefix+'_ci_lower']=lower;expected[prefix+'_ci_upper']=upper
        for key,value in expected.items():
            assert abs(float(row[key])-value)<1e-12,(key,row,value)
    out=source/'figures';out.mkdir(exist_ok=True)
    common=r'width=\linewidth,height=.74\linewidth,font=\small,grid=major,grid style={gray!20},xlabel={Outside weight $\theta_0$},xmin=0,xmax=1,xtick={0,.25,.5,.75,1}'
    frequency_lines=['% Exact PL references and empirical pointwise 95% Wilson intervals.',r'\begin{tikzpicture}',r'\begin{axis}['+common+r',ylabel={Comparison frequency},ymin=0,ymax=1.04,legend pos=north east,legend style={font=\small,cells={anchor=west}}]']
    mean_lines=['% Same colors/markers as the frequency panel; dashed black line is 1/3.',r'\begin{tikzpicture}',r'\begin{axis}['+common+r',ylabel={Conditional win fraction},ymin=0,ymax=1.04,extra y ticks={0.333333333333},extra y tick labels={$1/3$}]',r'\addplot[black,dashed,thick,domain=0:1,samples=2] {1/3};']
    for cap,color,mark in zip(provenance['caps'],COLORS,MARKS):
        selected=sorted([r for r in rows if int(r['m'])==cap],key=lambda r:float(r['theta0']))
        exact=' '.join(f'({t:.12g},{exact_probabilities(theta,float(t),pivot,item,cap)[0]:.12g})' for t in np.linspace(0,1,101))
        frequency_lines += [r'\addplot['+color+r',thick,no marks] coordinates {'+exact+'};',r'\addlegendentry{$m='+str(cap)+'$}']
        for lines,field,lower,upper in [(frequency_lines,'frequency','frequency_ci_lower','frequency_ci_upper'),(mean_lines,'conditional_mean','mean_ci_lower','mean_ci_upper')]:
            coordinates=' '.join(f"({float(r['theta0']):.12g},{float(r[field]):.12g}) += (0,{float(r[upper])-float(r[field]):.12g}) -= (0,{float(r[field])-float(r[lower]):.12g})" for r in selected)
            lines.append(r'\addplot['+color+r',only marks,mark='+mark+r',mark size=1.7pt,forget plot,error bars/.cd,y dir=both,y explicit] coordinates {'+coordinates+'};')
    for name,lines in [('comparison_frequency',frequency_lines),('comparison_conditional_mean',mean_lines)]:
        lines += [r'\end{axis}',r'\end{tikzpicture}','']
        (out/(name+'.tikz')).write_text('\n'.join(lines))
    provenance['validation']='All 36 settings independently rechecked: count identities, exact PL permutation probabilities, conditional means, and Wilson confidence limits.'
    provenance['plot_generator']='scripts/plot_comparison_diagnostic.py'
    provenance['plot_generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for p in out.glob('*.tikz'):
        provenance['files'][str(p.relative_to(source))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (source/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Validated all 36 settings; generated exact-reference and empirical plots.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args();generate(args.results)
