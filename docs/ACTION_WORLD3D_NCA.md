# Action-conditioned NCA on calibrated 3D RGB worlds

Depends on the calibrated world generator and frozen perspective benchmark.
The NCA receives observed RGB context and **known/planned** x/yaw camera
actions, and forecasts future RGB open-loop. Depth, object ID, and camera
poses remain evaluator-only. An action-blind control sees identical RGB
but receives zero actions.

```bash
python -m experiments.action_world3d_nca --steps 100 --seeds 11 23 41
python -m unittest tests.test_action_world3d_nca -v
```

Test held-out camera trajectories from `ood_camera`. Evaluate per-horizon
whole-frame and foreground MSE. Comparing action-aware vs blind is a test
of action-conditioned RGB prediction, **not** evidence of metric 3D
understanding or a learned viewpoint-invariant state. Depth/geometry
probes and independent scene composition are required before that claim.

The `measurement` block is a positive control: feeding the trained aware model
counterfactual action plans moves its forecast monotonically and
sign-sensitively (the pathway is **live**, not a dead wire), while
`world_action_dependence` gives the ceiling on any learnable action effect. The
learned response is real but ~10× below that ceiling and below the aggregate
MSE floor, so aware≈blind is a **genuine but underpowered** effect, not a null.
See the [zero-effect diagnosis](ZERO_EFFECT_DIAGNOSIS.md).
