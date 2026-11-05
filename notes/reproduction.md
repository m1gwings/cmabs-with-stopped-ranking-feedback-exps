# Scientific reproduction specification

## Instance and statistical design

`experiment.yml` specifies n=30, k=5, m=1, T=1,000,000, 20 independent
repetitions, root seed 104729 and outside parameters {0.001,0.003,0.01,0.03}.
The 30 real-arm parameters are
`[1.00,0.95,0.90,0.85,0.80,0.79,0.78,0.77,0.76,0.75,0.50,0.47,0.44,0.41,0.38,0.35,0.32,0.29,0.26,0.23,0.20,0.18,0.16,0.14,0.12,0.10,0.08,0.06,0.04,0.02]`.
The fifth/sixth-arm gap is 0.01. There are 240 full runs.

## Algorithm and measurement contracts

- Real-pivot UIF follows journal Algorithm 1 and its Appendix C guards: uniform
  initial real pivot, uniform exploration subset, real-arm rank breaking,
  transitive closure across pivots, and simultaneous decisions from pre-round
  accepted/unknown/pivot state. Item 0 is never used as a comparison pivot.
- UIF uses sqrt(log(2*n*T/delta)/N), delta=0.05 and algorithm horizon 10^6.
- The AOA-RB-PL baselines use x=2*log(T), with separate adaptive/nonadaptive modes.
  Their implementation is sourced from conference experiment commit
  `0764019ad043b05ec2c9a58306da0de591c43fe5`.
- The journal algorithm source is the manuscript at
  `https://github.com/m1gwings/cmabs-with-stopped-ranking-feedback`.
  Implementation and recovery checks are in `tests/test_science.py`.
- Regret per round is (sum of the k best parameters minus sum of selected parameters)/k.
  Figures show mean cumulative regret with pointwise 95% Student-t intervals
  computed from sample SEM and t(.975,19).
- Independent injected algorithm/data RNGs and complete checkpoints make changes
  to operational worker count leave the scientific trajectories unchanged.
- The pilot uses 10,000 rounds and one repetition per cell while keeping the
  algorithm horizon at 10^6. It is diagnostic and isolated from full-study data.
- `experiment.yml` tests MNL feedback (m=1). `experiment_stopped.yml` tests
  intermediate ranking caps with positive outside parameters.
- `fixed_length/` contains the fixed-length configurations and aggregate results,
  preserving scientific run groups, component sources and root seed.

## Software and compute

Python 3.12.3, NumPy 1.26.4, SciPy 1.13.1 and EWS revision
`4ba0dd60e43e8dabae5b5f879cde01f6765418db` are pinned in `requirements.lock.txt`.
