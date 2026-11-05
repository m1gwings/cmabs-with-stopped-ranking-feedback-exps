# Comparison diagnostic

All nine million fixed-assortment rankings completed. The offered parameters are
[1, 0.8, 0.5, 0.2, 0.1], with pivot parameter 0.2 and comparison-arm parameter 0.1.
The diagnostic uses caps 1–4 and nine outside parameters, with one million independent
rankings per outside parameter. Caps share races; outside parameters have independent
SeedSequence child streams. Array indices in the configuration are zero-based.

The exact conditional win probability is 1/3 for every cap and outside parameter.
Observed win fractions range from 0.33007 to 0.33460. At cap 4, exact comparison
frequency falls from 1 at outside parameter 0 to 3/13 at outside parameter 1; empirical
frequencies are 1 and 0.230567. The figures separate loss of comparisons from
preservation of their conditional mean. They illustrate the proved PL property,
not an empirical proof of unbiasedness.

Counts, ratios, exact PL permutation probabilities and all pointwise Wilson 95%
confidence limits were independently rechecked. The conditional-mean intervals
use the observed number of comparisons as their denominator. They are pointwise,
not simultaneous bands; no assertion that all intervals cover their reference is
made. The empirical sampler was also tested for exact agreement with the original
stopped-ranking environment under common RNG draws.

Results and scientific provenance: `reference_results/comparison_diagnostic/`.
Regenerate native figures with:

```sh
python3 scripts/plot_comparison_diagnostic.py reference_results/comparison_diagnostic
```
