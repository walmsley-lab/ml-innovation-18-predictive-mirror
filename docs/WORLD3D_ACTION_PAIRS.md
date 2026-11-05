# Controlled counterfactual camera actions

**Why:** The earlier 3D pilot's action-aware vs blind MSE was numerically
indistinguishable even though an intervention could weakly move predictions.
This experiment uses a cleaner causal design: *same past RGB, same world
motion, two opposite **future** camera plans*.

```bash
# Pure world signal, no model or learned result claimed:
python -m experiments.world3d_action_pairs --seeds 11 23 41
# Frozen trained checkpoint, never fitted on the test episodes:
python -m experiments.world3d_action_pairs \
  --checkpoint runs/action-world3d-nca/seed-11-action_aware.npz
# Untrained mechanics smoke only:
python -m experiments.world3d_action_pairs --untrained-smoke --seeds 11
python -m unittest tests.test_world3d_action_pairs -v
```

The **world-action difference** is what the renderer actually changes.
The model-action difference is how much the frozen predictor responds;
sign alignment tests whether it changes in the right direction.
Report per-horizon response and forecast error, not only aggregate RGB MSE.
Future RGB, object states and camera poses stay evaluator-only. This is
not a trained 3D reconstruction or proof of active information gathering.
