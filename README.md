# Predictive Mirror

A minimal executable path toward a **causal predictive mirror**, with an experimentally testable hypothesis about developmental perception. Start with a deterministic low-resolution world, train a locally recurrent perceptual field, and measure forecasts before adding representational sheets, plasticity or learned gaze.

**Read [the charter and PR plan](docs/PLAN.md) before extending the experiment.**

## Quick start (Python 3.11)

```bash
python -m pip install 'jax==0.9.0.1' 'numpy==2.3.5'
python -m unittest discover -s tests -v
python -m train --steps 120 --batch-size 8 --output runs/pilot
```

`runs/pilot/` contains `metrics.json`, `parameters.npz`, and `example.npz`. Pilot metrics compare a recurrent predictor with last-frame persistence on untouched synthetic trajectories. It does **not** prove emergent perception.

## Akash GPU

`Dockerfile` installs JAX CUDA 12 support; `deploy/akash-gpu.yaml` provides an example one-GPU Akash SDL. On launch the container trains, then serves **synthetic-only** results on port 8080. The image in the SDL is **a placeholder until a real GHCR image is published**. Read the [Akash runbook](docs/PLAN.md#akash-training-runbook) before bidding, and **close the lease when done**. No paid job has been started.

## Code conventions

- Functions and explicit arrays, no class hierarchies or framework registries.
- Self-contained experiments; shared code moves to `src/` only when actually reused.
- Physics/simulation ground truth is held out of the model's training inputs.
- Split by complete sequences and random seeds, not individual neighboring frames.
- No claim of emergence without meaningful ablations, generalization and causal tests.
