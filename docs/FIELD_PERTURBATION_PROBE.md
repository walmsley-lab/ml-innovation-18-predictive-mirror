# Calibrated propagation and recovery

Before interpreting a neural-field intervention, calibrate the
**physical impulse response** against the numerical heat or wave
integrator. Inject a local `u` perturbation (or `v` perturbation for a
wave), retain the same pre-intervention world, and measure difference
trajectories. Zero-perturbation and equal-energy spatially displaced
impulses are controls.

```bash
python -m experiments.field_perturbation_probe --kind diffusion
python -m experiments.field_perturbation_probe --kind wave
python -m unittest tests.test_field_perturbations -v
```

Report per-time-step response L2, peak amplitude, second spatial
moment, and numerical mass conservation. Separate physical spread
from actual recovery/damping: **pure reflecting diffusion conserves
signed mass**; it redistributes a pulse, not annihilates it. Damped
waves lose energy over time; do not equate shrinking local amplitude
with global energy disappearance.

An optional frozen checkpoint from Spatial #109 permits a learned
perturbation-response comparison. **No checkpoint** means this
reports validated physical positive controls only, not evidence
that an NCA learned propagation.
