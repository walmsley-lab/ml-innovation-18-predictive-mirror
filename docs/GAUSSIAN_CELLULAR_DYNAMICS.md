# Shared local dynamics on 3D Gaussian scene state

**Second Gaussian gate: dynamics, still privileged state.** The scene
contains a small set of metric Gaussian centers with observed position
and velocity at the last context frame. Each element carries hidden
channels; it receives radius-weighted messages from neighboring
Gaussians and applies the **same** neural rule. The model forecasts
the scene elements' positions open-loop, then renders from the known
camera viewpoints.

```bash
python -m experiments.gaussian_oracle_dynamics --steps 80 --seeds 11 23 41
python -m unittest tests.test_gaussian_cellular_dynamics -v
```

Compare against static and constant-velocity Gaussian-state controls,
with equal privileged observed geometry and held-out camera trajectories.
The initial local rule does not yet predict color, scale or opacity.
Report validation-selected steps, world-coordinate error, rendered RGB
error and the Gaussian-vs-analytic renderer gap separately.

**Not visual perception:** past centers and velocities are provided by
the simulator. No image-to-scene encoder, unknown object correspondence
or object discovery is learned here. The next experiment will remove
that privileged inference input. Do not replace any RGB NCA or particle
experiment based on the outcome of this pilot.
