# Perspective-aware 3D benchmark

Depends on the calibrated 3D world generator. Freeze test seed ranges
and OOD depth, camera and appearance families *before* model training.

```bash
python -m experiments.world3d_benchmark --episodes 16 --domain ood_camera
python -m unittest tests.test_world3d_benchmark -v
```

Scoring uses whole-RGB MSE and evaluator-derived visible-geometry MSE.
An **oracle-depth reprojection** control uses true previous depth and
future camera pose; it is **not** a fair RGB-only model baseline. Disocclusion
holes, perspective changes, and object dynamics are expected to limit it.
Per-frame camera calibration is recorded in the generator. Depth RMSE can
be scored from frozen prediction probes, not from training-state access.
Future experiments must use the exact same episode list and action
conventions before comparing 3D world-model claims.
