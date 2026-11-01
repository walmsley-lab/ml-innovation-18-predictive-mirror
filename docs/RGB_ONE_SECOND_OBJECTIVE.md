# One-second dynamics: matched loss ablation

Contrast **uniform** multi-step training loss with a **late-lead**
(+10-frame focused) loss using exactly the same RGB NCA, initialization seeds,
training episodes, update count, and held-out test episodes.

The evaluation and checkpoint choice are based on plain +10 RGB MSE in
both arms; report changed-region error and velocity controls as well.
The untrained **step-zero persistence checkpoint remains eligible** in
both arms. A selected step of zero is a negative learning result, not a
successful one-second predictor. Report this explicitly.

Run: `python -m experiments.rgb_horizon_objective --steps 100 --horizon 10`

This tests a training-objective hypothesis, not improved biological fidelity,
and does not claim to beat constant velocity until independently measured.
