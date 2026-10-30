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


## Stacked implementation PRs

Review and merge PRs #1–#9 in order, retargeting each stacked PR to `main` once its parent is merged. PR #2 covers durability, #3 curriculum, #4 strong feedforward control, #5 a two-sheet model, #6 exploratory representation probes, #7 analytical gaze, #8 plasticity toy, and #9 GPU sweep harness. The original PR estimates in `docs/PLAN.md` were a scope sketch, not actual GitHub PR numbering.

```bash
python -m unittest discover -s tests -v
python -m sweep --steps 2 --batch-size 2 --seeds 3 --architectures recurrent feedforward multisheet --curricula baseline --episodes 2 --output runs/sweep-smoke
```

**No GPU lease has run**. Before Akash, verify Docker build, NVIDIA JAX, immutable GHCR image pull, SDL validity, checkpoint off-provider export, and closed lease readback; see `docs/TRAINING.md`.
