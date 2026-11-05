# Calibrated perceptual failure diagnosis

The established `forecast_diagnostics.collapse_report` reports
rail saturation, cross-scene dependence, and live/dead/insensitive
positive-control sensitivity. The new additive
`perceptual_failure_report` screens against known RGB synthetic truth
for **black**, **bright**, **diffuse**, **misplaced**, and
**input-independent** forecasts. A correct oracle is the positive
control. Tests include all-zero, all-white, all-red, and spatially
shifted failures.

The diverse-world training runner now writes a `measurement` block
beside its existing per-lead event scores for validation and tests,
without changing the model-training objective or selection metric.

```bash
python -m unittest tests.test_perceptual_failure_diagnostics -v
python -m experiments.rgb_world_ladder_training --models cnn --seeds 11 \
  --curricula linear --steps 1
```

These thresholds are *calibrated to 32px synthetic color discs*,
not universal standards. **A bright image may have full brightness
foreground recall and zero hue-specific object mass.** Report both.
A spatially uniform image repeated across distinguishable scenes
is input-independent even if it does not hit a clip rail.

The independence test is disabled when the batch has one scene or
the underlying truth is effectively identical. A missing hard color
detection receives an image-diagonal penalty in the separate event
evaluator; it is not a measured center offset. This module is a
diagnostic, not proof of learned object identity or causality.
