# Gaussian-state active view: the *conditional* evidence gate

This is a downstream **evaluation protocol**, not a claim that the
RGB-inferred Gaussian model has learned object permanence or 3D
information seeking. Train at least two independently seeded
`experiments.rgb_gaussian_nca` models and freeze checkpoints
before any actual research evaluation.

```bash
# Wiring only, never science:
python -m experiments.gaussian_active_view --untrained-smoke \
  --seeds 11 --domains ood_depth --steps 3
# Real evaluation on trained independent models:
python -m experiments.gaussian_active_view \
  --checkpoints runs/rgb-gaussian-nca/seed-11.npz \
                runs/rgb-gaussian-nca/seed-23.npz
python -m unittest tests.test_gaussian_active_view -v
```

The model's policy sees observed RGB and known camera commands, never
the world state. Candidate views maximize disagreement across frozen
model forecasts (a heuristic, **not** calibrated information gain).
Compare fixed, random, active and independent-donor replay-yoked
policies on identical worlds and the *same standardized home view*.
All use the same number of observations, but active search spends
more inference compute; record that cost before interpreting gains.

## Real webcam transfer remains a separate gated task

This controlled study assumes camera calibration, a fixed world origin
and known commanded x/yaw actions. A handheld webcam generally has
**none** of those guarantees. Do not pretend unknown motion means
zero motion. For live transfer, establish:
- measured camera intrinsics or an explicit unknown-calibration mode;
- recorded or estimated camera actions with uncertainty;
- monotonic capture, forecast-commit and target timestamps;
- genuine held-out video with static/moving-camera strata;
- no simulator geometry or future observations as inference inputs;
- object localization, per-lead soft mass/occlusion and latency
  versus persistence/CNN/transport/pixel NCA baselines.

**No** webcam adapter, production 3DGS renderer or calibrated
epistemic-uncertainty estimator is implemented by this PR.
