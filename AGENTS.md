# Journal experiment instructions

- This repository owns experiments, independently of the sibling journal paper.
- `notes/reproduction.md` records scientific sources and deliberate deviations.
  Keep the real-pivot journal UIF distinct from the conference MNL-UIF.
- Preserve the four outside weights, instance, horizon, confidence parameters,
  baselines, regret definition, and independent repetitions unless requested.
- Use injected, separate RNGs and complete component checkpoints. Validate
  scientific invariants and exact direct/resumed trajectories before long runs.
- Pin EWS and numerical dependencies. Keep simulation, metrics, and plotting in
  separate modules so analysis changes need not invalidate simulations.


- Exclude generated artifacts and environments from Git. Commit and synchronize
  source changes to the configured remotes. Never discard unrelated work.

- Keep machine-specific hostnames, account names, private paths and access/allocation configuration out of repository content and generated exports. Export only explicitly selected scientific metadata.
