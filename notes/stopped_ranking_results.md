# Stopped-ranking results

All 320 requested trials completed without failed or corrupt runs. Independent
validation checked every retained per-run metric checksum, horizon, repetition
identity and completion state, and recomputed all aggregate means and sample SEMs.
Intervals are pointwise 95% Student-t intervals over 20 repetitions.

| Outside parameter | Final mean regret, m=1 | Final mean regret, m=4 | Reduction |
| --- | ---: | ---: | ---: |
| 0 | 34,337 | 22,279 | 35.1% |
| 0.1 | 34,188 | 23,381 | 31.6% |
| 0.3 | 36,065 | 24,893 | 31.0% |
| 1 | 40,614 | 28,979 | 28.6% |

At every tested outside parameter, mean final regret decreases with each cap
increase from 1 to 4. The largest reduction is from cap 1 to 2. The cap-1 and
cap-4 mean-regret intervals are disjoint at all four outside parameters. This is a
descriptive comparison; no multiple-comparison-adjusted test is claimed.
Learning curves do not yet plateau at the chosen horizon.

Validated aggregates, intervals and scientific provenance are in
`reference_results/stopped_ranking/`. Generate native plots with:

```sh
python3 scripts/plot_stopped_results.py reference_results/stopped_ranking
```

The two figures show final regret versus cap at all outside parameters, and regret
trajectories versus time at outside parameter 0.3. Curve bands use the same pointwise
Student-t intervals. The latter requires PGFPlots' fillbetween library.
