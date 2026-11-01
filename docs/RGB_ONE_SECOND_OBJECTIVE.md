# One-second dynamics: matched loss ablation

Contrast **uniform** multi-step training loss with a **late-lead**
(+10-frame focused) loss using exactly the same RGB NCA, initialization seeds,
training episodes, update count, and held-out test episodes.

The evaluation and checkpoint choice are based on plain +10 RGB MSE in
both arms; report changed-region error and velocity controls as well.
No step-zero checkpoints are excluded by reporting rules, but this runner
selects among evaluated *trained* checkpoints; compare to a separately
evaluated untrained persistence baseline to detect nonlearning.

Run: `python -m experiments.rgb_horizon_objective --steps 100 --horizon 10`

This tests a training-objective hypothesis, not improved biological fidelity,
and does not claim to beat constant velocity until independently measured.
