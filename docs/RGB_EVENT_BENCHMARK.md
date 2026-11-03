# Frozen RGB event benchmark (pilot)

The normal RGB benchmark reports full-frame and thresholded motion MSE. This
independent evaluator adds **foreground error, visible-object localization,
missing detections, and return-after-invisibility** on the same frozen
synthetic episodes. The evaluator reads synthetic colors and visibility
only *after* the forecaster commits its predictions. It does not train or
modify an NCA.

```bash
python -m experiments.rgb_event_benchmark --episodes 16 --balls 2
python -m experiments.rgb_event_benchmark --checkpoint runs/rgb-curriculum-study/seed-11-curriculum.npz --episodes 16 --balls 2
python -m unittest tests.test_rgb_event_metrics -v
```

## Interpretation

- Foreground is thresholded from **target RGB**, hence is evaluator-only.
- A visible synthetic object is detected by matching its known palette
  color. A missing forecast receives an image-diagonal distance penalty.
  This tests simple colored-disc localization, **not object identification**.
- Return-after-invisibility may reflect either an off-screen re-entry or
  barrier occlusion. **Do not present it as a pure occlusion/reappearance score.**
- Missing events produce `null`, not fabricated zero errors.
- Synthetic seed separation from training and checkpoint selection must be
  checked externally; default test seeds are 81000+.
- This first contract does not yet establish OOD robustness, multi-run
  confidence intervals, geometry or spatial memory. Follow-on simulator
  and benchmark PRs should add unambiguous event metadata and stronger
  learned controls, without silently changing the v1 metric meaning.

The underlying `rgb_motion_world.episode` exposes state for diagnostics,
but `src.rgb_benchmark.frames` is the only data passed to the predictor.
