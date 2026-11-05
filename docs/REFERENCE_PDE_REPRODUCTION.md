# Reproducing learned spatial dynamics

Reference: Richardson et al. (2024), [Learning spatio-temporal patterns with Neural Cellular Automata](https://doi.org/10.1371/journal.pcbi.1011589).

```bash
python -m experiments.reference_reaction_diffusion --steps 150
python -m unittest tests.test_reference_reaction_diffusion -v
```

This PR adds a clean Gray-Scott finite-difference world and a tiny differentiable shared-local-rule NCA trained for one-step transitions, then evaluated on a multi-step rollout from a separate initial seed. The teacher's PDE states are training targets and evaluator truth only. No future state enters prediction.

**Reproduction level:** reference-*inspired*, not yet quantitative reproduction of Richardson et al.: the exact PDE, reaction parameters, training curriculum, model size and evaluation settings have not been matched. A future parity run must pin their code and evaluate the published conditions rather than comparing arbitrary normalized field MSE.

This is a useful working baseline for local spatiotemporal computation, especially if video prediction remains difficult to train. Report unsuccessful training or instability plainly, and preserve the explicit PDE solver as a strong known-physics reference.
