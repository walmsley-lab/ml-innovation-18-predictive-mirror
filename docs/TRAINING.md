# PR 9 — Training matrix and Akash handoff

The GPU image now contains all training models, the evaluator, lightweight experiments and `sweep.py`. It is still intentionally a Python/JAX executable experiment, not an experiment-management framework.

Local smoke (also in CI):

```bash
python -m unittest discover -s tests -v
python -m sweep --steps 2 --batch-size 2 --seeds 3 --architectures recurrent feedforward multisheet --curricula baseline --episodes 2 --output runs/sweep-smoke
```

For a first GPU smoke: use the reviewed immutable digest in `deploy/akash-gpu.yaml`, keep `RUN_MODE=train`, set `STEPS=3`, `BATCH_SIZE=2`, validate/dry-run, inspect interactive bids, launch, confirm JAX GPU logs, download results, **close the lease and query its closed state**. No deployment has been launched by this PR.

After that, use `RUN_MODE=sweep`, `STEPS=120`, `BATCH_SIZE=8` for the first small matrix. Defaults: three architectures × two curriculum orders × five seeds = 30 independent runs; this can be expensive. Start with 1–2 seeds locally. Docker's default GPU entrypoint is intentionally strict. Sweep outputs are in `runs/sweep/<architecture>-<curriculum>-seed<seed>`; a `runs.json` file updates after each completed run, but individual models only checkpoint at configured intervals. The current sweep does not yet automatically resume/skip completed conditions: plan for possible restart and verify exports.

**Artifacts and costs:** HTTP exposes synthetic artifacts while training. Download latest checkpoints/metrics off-provider, verify checksums, and retain them outside the lease. Exposing data via HTTP is not itself backup. Do not deploy private recordings. Rate cap is a bid, not a hard spend cap. Close and read back deployment status even after startup failure.

**Scientific limits:** The 2-step smoke failed object detection in all three architectures (reappearance miss penalty 16 px). These models are not ready to claim predictive object tracking. Real training and stronger objectives may be needed. A sweep summary reports descriptive mean/SD, not CIs or hypothesis confirmation.
