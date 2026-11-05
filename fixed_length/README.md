# Fixed-length ranking experiments

Run from this directory after installing the root `requirements.lock.txt`:

```bash
ews run experiment.yml --output ../outputs/fixed_length --workers 2
ews run experiment_epsilon_0_05.yml --output ../outputs/epsilon_0_05 --workers 2
ews run experiment_epsilon_0_2.yml --output ../outputs/epsilon_0_2 --workers 2
```

The three configurations specify 120, 40, and 40 trials respectively. Their
scientific run groups, parameters, seed and component sources are preserved.
Only fixed-length tasks are selected; the main repository's real-pivot UIF is
used for its MNL and stopped-ranking studies.

Source attribution and checksums are in `reference_results/*provenance.json`
and its subdirectories. The implementation is reproduced from the authors'
[ranking-feedback experiments](https://github.com/gianmarcogenalti/combinatorial-bandits-relative-feedback).
The aggregate CSVs retain the fixed-length rows and original numeric values.
Install the root `requirements-dev.txt` and run component checks from this
directory with `python -m pytest tests -q`.
