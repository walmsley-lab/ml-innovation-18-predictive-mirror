# Persistent Gaussian observer with full-history state

The existing Gaussian RGB encoder estimates scene elements from the
last two images when predicting motion. This independent experiment
carries estimated Gaussian **centers and velocities** across every
observed frame, propagates them through missing observations, and
applies confidence-weighted corrections when appearance evidence is
available. Renderer and cellular future dynamics share the same
frozen trained parameters across all comparison arms.

```bash
python -m experiments.persistent_gaussian_observer \
  --checkpoint runs/rgb-gaussian-nca/seed-11.npz \
  --seeds 11 23 41 --hidden-lengths 2 4
# Only a mechanics check, not learned perception:
python -m experiments.persistent_gaussian_observer \
  --untrained-smoke --seeds 11 --hidden-lengths 2
python -m unittest tests.test_persistent_gaussian_observer -v
```

Compare the frozen **two-frame** model, persistent full-history
observer, midpoint and final-frame resets, and shuffled earlier
observations. A controlled target-specific opaque rectangle hides
one object for prescribed observed frames while the simulator's
future target remains unchanged. Longer masks test whether a
carried estimate survives without seeing the object. All methods
get the same observation images and planned camera commands.

Per-frame evaluator-only permutation-matched metric 3D center error
and drift assess accumulated estimation error, while per-lead RGB
and foreground-view MSE check actual predicted pixels. Galleries
show observed target, future truth, two-frame forecast and
persistent forecast.

The persistent filter uses **hand-designed confidence gating**—it
is not a learned general-purpose memory mechanism or demonstrated
visual object permanence. All metric 3D geometry remains
evaluator-only. Inference uses known camera actions; unknown
camera motion and genuine multi-object hidden identity remain open.
