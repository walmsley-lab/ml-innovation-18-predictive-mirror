# Frozen RGB forecasting benchmark

This is an independent evaluation entry point; existing Step 5/6 code
and its reported MSE are not overwritten or retroactively compared.

- Default test seeds: 71000 onward. Exclude these from training/validation.
- Predictions see only context frames; no world positions, hidden velocities,
  future RGB, or visibility flags are available during inference.
- Evaluate +1/+3/+5/+10 frames at a nominal 10 Hz (ten frames = one second
  only when capture timing confirms it).
- Always compare whole RGB MSE, changed-region MSE and count, against
  persistence and **image-derived** constant velocity. Optical flow is optional.
- Step-zero selections and static-background dominance must be noted.
- Keep model seeds and test scene identities frozen across ablations.

Run: `python -m experiments.rgb_benchmark --checkpoint runs/rgb-motion-distribution/seed-11-distribution.npz`

This does not demonstrate one-second anticipation or generalization on its own.
