# Experiments for Combinatorial Bandits with Stopped Ranking Feedback

Cristiano Migali, Gianmarco Genalti, Alberto Maria Metelli, and Marco Mussi

Code and aggregate results for *Worst-Case Regret Bounds for Combinatorial Bandits
with Stopped Ranking Feedback*.

## How to Run the Experiments

The experiments use [Experiments W/O Stress (EWS)](https://github.com/m1gwings/experiments-wo-stress).
You need Python 3.12 and Git. Clone this repository and install the pinned
packages in a virtual environment:

```bash
git clone https://github.com/m1gwings/cmabs-with-stopped-ranking-feedback-exps.git
cd cmabs-with-stopped-ranking-feedback-exps
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the MNL and stopped-ranking studies with:

```bash
ews run experiment.yml --output outputs/mnl --workers 2
ews run experiment_stopped.yml --output outputs/stopped_ranking --workers 2
```

EWS runs the simulations and computes regret. Change `--workers` to choose the
number of parallel workers. To resume an interrupted experiment, repeat the same
command with the same output folder. To inspect saved progress, use:

```bash
ews inspect outputs/mnl
```

Run the fixed-length ranking and feedback-level studies from `fixed_length/`:

```bash
cd fixed_length
ews run experiment.yml --output ../outputs/fixed_length --workers 2
ews run experiment_epsilon_0_05.yml --output ../outputs/epsilon_0_05 --workers 2
ews run experiment_epsilon_0_2.yml --output ../outputs/epsilon_0_2 --workers 2
cd ..
```

Run the fixed-assortment comparison diagnostic with:

```bash
ews run experiment_diagnostic.yml --output outputs/comparison_diagnostic
```

Every command runs the simulations, analysis and configured TikZ figures. The
diagnostic is one atomic trial: caps share rankings within each outside weight,
and an interrupted trial restarts from the beginning.

## Experiments

The YAML configurations specify instances, algorithm parameters and root seed
104729. Each learning setting uses 20 independent trials of 1,000,000 rounds,
for a total of 760 learning runs. The diagnostic samples nine million rankings.

| Experiment | Setting | Algorithms | Runs |
| --- | --- | --- | ---: |
| Fixed-length ranking | 400 arms, k=m=2; ranking gaps 0.05, 0.1 and 0.2 | UIF and RecMaxMinUCB | 120 |
| Feedback level | 16 arms, k=6, m in {1,2,3,6} | UIF | 80 |
| MNL feedback | 30 arms, k=5, m=1, outside parameter in {0.001,0.003,0.01,0.03} | UIF, AOA-RB-PL and AOA-RB-PL-Adaptive | 240 |
| Stopped ranking | 30 arms, k=5, m in {1,2,3,4}, outside parameter in {0,0.1,0.3,1} | UIF | 320 |
| Comparison diagnostic | Five offered arms; caps 1–4 and nine outside parameters | Exact PL probabilities and empirical comparisons | 9 million rankings |

The MNL and stopped-ranking studies share the real-arm parameters
`[1.00,0.95,0.90,0.85,0.80,0.79,0.78,0.77,0.76,0.75,0.50,0.47,0.44,0.41,0.38,0.35,0.32,0.29,0.26,0.23,0.20,0.18,0.16,0.14,0.12,0.10,0.08,0.06,0.04,0.02]`.
The gap between the fifth and sixth arms is 0.01.

Regret is cumulative mean-preference top-k regret. Learning curves and error
bars show pointwise 95% Student-t confidence intervals for the mean over 20
trials. The diagnostic reports pointwise 95% Wilson intervals for frequencies
and conditional win probabilities.

## Results and Figures

Validated aggregate curves, confidence limits, source checksums and native
TikZ figures are included in:

- [`fixed_length/reference_results/`](fixed_length/reference_results/): ranking comparison, feedback levels and gap variants.
- [`reference_results/`](reference_results/): MNL study.
- [`reference_results/stopped_ranking/`](reference_results/stopped_ranking/): stopped-ranking study.
- [`reference_results/comparison_diagnostic/`](reference_results/comparison_diagnostic/): comparison diagnostic.

These are aggregate exports; unaggregated per-run trajectories are not included.
Figures are written under `OUTPUT/analysis/figures/`. To regenerate them from
saved EWS output without repeating the simulations, use the matching configuration:

```bash
ews plot experiment.yml --output outputs/mnl
ews plot experiment_stopped.yml --output outputs/stopped_ranking
ews plot experiment_diagnostic.yml --output outputs/comparison_diagnostic
```

For fixed-length plots, run `ews plot experiment.yml --output ../outputs/fixed_length`
from `fixed_length/`, or substitute the configuration and output for a gap variant.
The included reference CSVs and TikZ files are standalone aggregate exports, not
EWS output directories. Their provenance records describe the original runs;
the diagnostic now uses EWS random streams, so its Monte Carlo counts can differ.

The TikZ plots inherit the fonts of the LaTeX document. Load `pgfplots`, its
`fillbetween` library, and `fix-cm`, and set `\pgfplotsset{compat=1.18}` before
using `\input{PATH/NAME.tikz}`.

## Cite This Work

If you use this code in a scientific publication, please cite the manuscript:

```bibtex
@unpublished{migali2026worst,
  title={Worst-Case Regret Bounds for Combinatorial Bandits with Stopped Ranking Feedback},
  author={Migali, Cristiano and Genalti, Gianmarco and Metelli, Alberto Maria and Mussi, Marco},
  year={2026},
  note={Manuscript}
}
```

## How to Contact Us

You can write an e-mail to cristiano.migali@polimi.it.
