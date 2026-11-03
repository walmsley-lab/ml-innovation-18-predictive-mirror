# Active-gaze evidence and verified live forecast timestamps

This experiment reinforces the original 1D active-gaze *control* baseline.
It pairs fixed, random and information-seeking policies on identical hidden
trajectories; reports action-aware vs action-blind position errors, gaze
coverage, forecast entropy, and episode-bootstrap differences.

```bash
python -m experiments.active_gaze_comparison --episodes 24 --seeds 11 23 37
python -m unittest tests.test_active_gaze_comparison -v
```

**Do not promote active sensing into the RGB mirror yet.** Its current policy
has a known world/sensor model and no learned RGB belief. Replacing it with
a model-driven RGB policy must be a separate experiment with fixed and
replayed-action controls and a held-out viewpoint-based evaluation.

## Verified live RGB timing correction

`experiments.rgb_verified_live` previously called `delayed.issue` with the
camera-read timestamp although model inference took place later. The revised
viewer uses `time.monotonic_ns()` after the forecast has been materialized,
then matches it only to subsequently captured frames. This measures
application-level forecast-commit-to-frame-arrival time, not guaranteed
sensor exposure time: camera driver buffering can still invalidate an
exposure-level causality claim.

## Gate to real 3D active perception

1. Select a frozen RGB model by independent 2D/OOD evaluation; record SHA.
2. Bring up the opt-in 3D generator with camera intrinsics and extrinsics,
   RGB, depth and instance segmentation in the evaluator, and known actions.
3. Train/evaluate perspective and parallax forecasts on held-out scenes.
4. Compare fixed, random, entropy-directed and replay-yoked camera actions,
   at matched sensor and compute budgets.
5. Report RGB per-horizon error, depth/pose probes, hidden-state interventions,
   frame timestamps and latency separately.

Until these gates pass, this PR claims **a stronger active-gaze control and
a more honest RGB live timestamp**, not an integrated active 3D mirror.
