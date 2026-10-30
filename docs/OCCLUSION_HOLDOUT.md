# Multi-step occlusion holdout

The default one-pixel wall remains unchanged; this PR adds configurable 3- and 5-pixel opaque barriers. Train as before, then evaluate on fresh seeds at longer occlusion durations (no hidden labels enter training):

```bash
python -m experiments.occlusion_holdout --architecture multisheet \
 --parameters runs/multisheet/parameters.npz --widths 1 3 5 \
 --episodes 64 --steps 20 --out runs/occlusion-holdout.json
```

Report **actual barrier** reappearance position MAE, number of qualifying events, occluded-frame coverage and plain pixel MSE for model, persistence and velocity baselines. A null MAE indicates zero eligible events, not perfect performance. The test explicitly distinguishes the learned model's generalization to longer hidden-state duration from the one-pixel training task; it does not change the current training curriculum. Do not combine scores with historical pre-PR #15 event definitions. No perception or biological mechanism is claimed without trained-model evidence.
