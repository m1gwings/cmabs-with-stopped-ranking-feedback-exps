#!/usr/bin/env python3
"""Generate native PGFPlots from validated stopped-ranking aggregates (stdlib only)."""
import argparse
import csv
from pathlib import Path

# Two-sided 95% Student-t critical value for the study's 20 repetitions.
T_19 = 2.093024054408263
COLORS = ['blue!75!black', 'orange!90!black', 'teal!80!black', 'violet!85!black']


def coordinates(rows, xkey, ykey):
    return ' '.join(f"({float(r[xkey])/1000000:.12g},{float(r[ykey])/1000:.12g})" for r in rows)


def generate(source):
    with (source/'final_regret.csv').open() as f:
        final=list(csv.DictReader(f))
    with (source/'regret.csv').open() as f:
        curves=list(csv.DictReader(f))
    assert len(final)==16 and all(int(r['count'])==20 for r in final)
    out=source/'figures';out.mkdir(exist_ok=True)
    common=r'width=\linewidth,height=.72\linewidth,grid=major,grid style={gray!20},legend style={font=\small,cells={anchor=west}},tick label style={font=\small},label style={font=\small},scaled y ticks=false,ylabel={Cumulative regret ($\times 10^3$)}'
    lines=['% Pointwise 95% Student-t confidence intervals over 20 independent runs.',r'\begin{tikzpicture}',r'\begin{axis}['+common+r',xlabel={Ranking cap $m$},xtick={1,2,3,4},legend pos=north east]']
    for theta,color in zip([0.,.1,.3,1.],COLORS):
        selected=sorted([r for r in final if float(r['data.params.theta0'])==theta],key=lambda r:int(r['data.params.m']))
        coords=' '.join(f"({r['data.params.m']},{float(r['mean'])/1000:.12g}) +- (0,{(float(r['ci95_upper'])-float(r['mean']))/1000:.12g})" for r in selected)
        lines += [r'\addplot['+color+r',thick,mark=*,error bars/.cd,y dir=both,y explicit] coordinates {'+coords+'};',r'\addlegendentry{$\theta_0='+f'{theta:g}'+'$}']
    lines += [r'\end{axis}',r'\end{tikzpicture}','']
    (out/'final_regret_vs_m.tikz').write_text('\n'.join(lines))
    lines=['% Requires the PGFPlots fillbetween library.',r'\begin{tikzpicture}',r'\begin{axis}['+common+r',xlabel={Round $t$ ($\times 10^6$)},scaled x ticks=false,xtick={0,.25,.5,.75,1},xmin=0,xmax=1,ymin=0,legend pos=south east]']
    for m,color in zip([1,2,3,4],COLORS):
        selected=sorted([r for r in curves if int(r['data.params.m'])==m and float(r['data.params.theta0'])==.3],key=lambda r:float(r['x']))
        for r in selected:
            assert int(r['count'])==20 and r['uncertainty_kind']=='standard_error'
            r['lower']=float(r['mean'])-T_19*float(r['uncertainty'])
            r['upper']=float(r['mean'])+T_19*float(r['uncertainty'])
        for bound in ['lower','upper']:
            lines.append(r'\addplot[name path='+f'{bound}{m}'+r',draw=none,forget plot] coordinates {'+coordinates(selected,'x',bound)+'};')
        lines += [r'\addplot['+color+r',fill opacity=.12,forget plot] fill between[of='+f'lower{m} and upper{m}'+'];',r'\addplot['+color+r',thick,no marks] coordinates {'+coordinates(selected,'x','mean')+'};',r'\addlegendentry{$m='+str(m)+'$}']
    lines += [r'\end{axis}',r'\end{tikzpicture}','']
    (out/'cumulative_regret_theta0_0_3.tikz').write_text('\n'.join(lines))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args();generate(args.results)
