# Gaussian scene representation: privileged-state sanity check

This is the first gated Gaussian-state milestone, **not** a learned
3D perception model. It represents the simulator's known spheres as
small metric 3D Gaussian splats, projects them with calibrated camera
poses, and compares static-scene versus constant-velocity scene
forecasts. The oracle-future-geometry arm is an evaluator-only
rendering ceiling and is NOT a predictive model.

```bash
python -m experiments.gaussian_oracle_baselines --seeds 11 23 41
python -m unittest tests.test_gaussian_splats -v
```

Each arm renders through the **same** soft Gaussian alpha compositing.
Errors against the analytic ray-sphere RGB world retain a representation
mismatch even for the oracle; geometry and same-renderer oracle RGB
differences are therefore also reported. Past simulator positions/radii/
colors are **privileged** for all arms, including static/velocity.
This must never be ranked against an RGB-only CNN or NCA as if the
inputs were equal. A subsequent experiment will infer scene state
from observed pixels and evaluate on matched information.

The renderer is intentionally tiny: isotropic Gaussian covariance only,
no learned opacity or appearance, no GPU 3DGS dependencies. Do not
replace the established calibrated world or historical pixel models.
