# Resumable GPU sweeps

Small jobs:

```bash
python -m sweep --steps 2 --batch-size 2 --seeds 31 --architectures recurrent --curricula baseline --episodes 2 --output runs/preflight
python -m sweep --steps 2 --batch-size 2 --seeds 31 --architectures recurrent --curricula baseline --episodes 2 --output runs/preflight --resume
```

A sweep records a `spec.json` for each condition, checkpoints model and optimizer state, and writes per-run evaluation and aggregate JSON atomically. `--resume` skips completed matching conditions, re-evaluates finished-but-not-scored runs, or continues from a matching checkpoint. Changed seeds, steps, loss, code revision, or evaluation budget are rejected rather than silently mixing results. Source revision is injected by the GHCR build; a local editable tree is `unknown`.

**Important:** A restart *within the same live Akash container* can reuse local checkpoints; closing or moving the lease destroys ephemeral files. Download checkpoints to a trusted local/off-provider destination before closing the lease, verify SHA-256 checksums, and restore them into the corresponding `runs/sweep/<condition>/` directory if starting a fresh container. The HTTP server currently exposes synthetic outputs only; it is not backup. No GPU/Akash run has been verified by this PR.

Build image after code merge; PR CI now exercises the sweep resume path and packages `tests/` for CPU-only container validation. Run on a Docker host: `docker build -t pm:smoke .` then `docker run --rm -e JAX_PLATFORMS=cpu --entrypoint python pm:smoke -m unittest discover -s tests -v`. Run the NVIDIA GPU gate with `--gpus all --entrypoint python ... -m train --require-gpu --steps 2 --batch-size 2`. A passing CPU check does *not* validate GPU support.

Next independent PR: export helper and restore test, then the user can run Docker + Akash preflight. The first paid deployment remains a three-update disposable smoke, not the full 30-run matrix.
