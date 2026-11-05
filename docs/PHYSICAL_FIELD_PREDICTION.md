# Physical fields: cellular versus numerical, fitted stencil and CNN

This experiment relies on [validated field worlds](PHYSICAL_FIELD_WORLDS.md)
and gives every predictor the complete last-observed physical state:
`(u,v)` for wave; `(u,0)` for diffusion. The numerical integrator
knows the true physical coefficients and is an oracle/reference—not
a fair trainable competitor. The fitted local linear stencil receives
the same states and is trained on the same training distribution.

Neural arms: a small two-layer CNN and the repository's existing
`shared_cellular_core.tick` invoked twice per physical step. Full-state
open-loop predictions are scored per horizon with RMSE, amplitude
growth, finite trajectories and diffusion mass drift.

```bash
python -m experiments.physical_field_prediction --kind diffusion --steps 40
python -m experiments.physical_field_prediction --kind wave --steps 40
python -m unittest tests.test_physical_field_prediction -v
```

This tests an easy-to-falsify task: whether a general shared cellular
rule adds anything beyond a fitted five-feature stencil or tiny CNN.
No GPU/paid runs or empirical advantage is asserted.
