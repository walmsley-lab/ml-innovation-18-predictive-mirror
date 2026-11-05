# Guaranteed per-frame reappearance benchmark

The prior generic `occlusion_long` task did not guarantee a return
transition within its evaluated horizon, and sparse lead sampling
could skip it entirely. This independent, deterministic test does:

- **Barrier:** starts observed and visible, crosses a known stripe,
  passes through partial visibility, becomes fully occluded for multiple
  future frames, and visibly returns within the 10-frame forecast.
- **Camera exit:** an independently scripted camera excursion makes a
  visible object leave the field of view and return. It is reported
  as `camera_return`, never merged with barrier reappearance.

For every forecast frame, score true visibility class, hue-specific
predicted mass, soft-centroid localization (in pixels when defined),
missing object, premature visibility during hiding and return timing
(in frames). Null means missing/undefined—not zero error.

```bash
# Perfect-future score is a positive-control of the evaluator ONLY:
python -m experiments.guaranteed_reappearance --mode barrier
python -m experiments.guaranteed_reappearance --mode camera_exit
# Real, held-out checkpoint (trained independently of these probe seeds):
python -m experiments.guaranteed_reappearance \
  --model nca_blend --checkpoint runs/rgb-world-ladder-training/nca_blend-seed11-mixed-best.npz
python -m unittest tests.test_guaranteed_reappearance -v
```

The `--checkpoint` flag is needed for any learned-model claim;
default oracle forecasting is a correctness control, not a real
predictive result. Simulator positions and event annotations are
evaluator-only. This benchmark does not claim a learned object
representation or end-to-end webcam transfer.
