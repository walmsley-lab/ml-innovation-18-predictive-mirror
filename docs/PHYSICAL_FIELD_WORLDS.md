# Validated 2D wave and diffusion worlds

Two independent, deterministic reference worlds with physical dt/dx:
`u_t = D Δu` and `u_tt + gamma u_t = c² Δu`.

For 2D explicit five-point diffusion, enforce
`D·dt/dx² <= 1/4`. For undamped 2D wave, enforce
`c·dt/dx <= 1/sqrt(2)`; with damping we additionally constrain
`gamma·dt < 2`. These are necessary discretization safety checks;
they do not guarantee an accurate continuum approximation.

Diffusion stores **u**; the wave stores **u and velocity**, so no
model is allowed to infer its dynamics from a single displacement
frame while being compared to a full-state numerical integrator.
`reflect` uses edge-padded zero normal flux; `zero` applies
zero-valued exterior cells (and pins wave boundary cells).

```bash
python -m experiments.physical_field_worlds
python -m unittest tests.test_physical_fields -v
```

Tests check seeded reproducibility, mass conservation for reflecting
diffusion, spatial propagation/velocity in waves, and instability
rejection. The reference is a finite-difference world, not
a learned model or a result favoring NCAs.
