# Synthetic occlusion demo v0

This demo freezes a trained recurrent checkpoint and renders the same three
barrier widths used by the occlusion holdout: 1, 3, and 5 pixels. Each frame
shows the target, recurrent forecast, persistence forecast, and recurrent
absolute error. Occluded frames are marked in the animation.

Generate it from a frozen checkpoint:

```bash
python -m synthetic_demo \
  --parameters runs/demo-r1000-contrast/parameters.npz \
  --output runs/demo-v0
```

The output contains `occlusion-1.gif`, `occlusion-3.gif`, and
`occlusion-5.gif`, plus `manifest.json` and `metrics.json`. The manifest
records the checkpoint SHA-256 and deterministic evaluation seeds. Event
metrics include reappearance event count, detection count/rate, and
localization MAE. A missed detection receives the evaluator's 16-pixel
penalty, so detection and localization are not conflated.

The simulator's hidden state is used only to annotate occlusion/reappearance
events and score the forecast. It is never passed to the recurrent model.
These three fixed-seed clips are a communication artifact, not independent
scientific evidence; use `experiments.occlusion_holdout` for the broader
held-out benchmark.

Rendering requires the optional Pillow package. Use `--no-render` to produce
only the manifest and metrics.
