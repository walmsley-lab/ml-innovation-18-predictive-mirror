# Integrated predictive 3D mirror demonstration

This is the first **end-to-end, locally viewable** storyboard for the
calibrated 3D world line. Provide trained, frozen action-conditioned RGB
NCA checkpoints to evaluate prediction of future camera views.

```bash
# Explicit untrained wiring smoke (not a result):
python -m experiments.world3d_demo --untrained-smoke --no-active
# Then open runs/world3d-demo/storyboard.html locally.

# Meaningful model evidence requires trained frozen checkpoints:
python -m experiments.world3d_demo \
  --checkpoints runs/a.npz runs/b.npz
python -m unittest tests.test_world3d_demo -v
```

The self-contained HTML slider shows the last observed RGB, forecast RGB
at each future lead, actual later RGB, persistence control and error.
Prediction uses RGB context plus known camera actions, **never future
RGB/depth/poses**. The active-policy comparison (if two checkpoints are
provided) controls fixed/random views on identical hidden world seeds
and evaluates everyone at the same final home view.

This is a **sampled 3D world**, not a live interactive MuJoCo engine or
a webcam implementation. Claims of active improvement require frozen
trained models and replicated held-out episodes. No graphic shown is
evidence of learned depth by itself.
