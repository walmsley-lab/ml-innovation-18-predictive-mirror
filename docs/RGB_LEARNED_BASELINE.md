# Tiny learned CNN baseline for the predictive mirror

A fixed RGB NCA benchmark should not rely only on persistence, a single
global translation, or optional optical flow. This independent experiment
trains a very small two-frame convolutional residual predictor, entirely
in JAX, and evaluates it on the existing frozen RGB scene protocol.

```bash
python -m experiments.rgb_tiny_cnn_baseline --steps 60 --seeds 11 23 41
python -m unittest tests.test_tiny_cnn_video -v
```

Model: RGB last frame + its difference from previous frame enters two 3x3
convolutions and a 1x1 RGB residual. Forecasts recurse **open-loop** for
the requested horizon. Unlike the NCA, the model has no independently
persistent cellular hidden state. Both are trained on RGB only.

Training, validation, and test worlds have separately seeded trajectories.
Step zero is eligible; checkpoint selection uses validation only.
Final comparisons report all standard leads, persistence, global translation
and a paired bootstrap confidence interval on the difference to velocity.

**Limitations:** This is not a faithful SimVP or ConvLSTM implementation;
a stronger standard architecture and a simulator-state kinematic diagnostic
upper bound remain future work. When supplying `--nca-checkpoint`, check
that its checkpoint was never selected on these test episodes and that the
world distribution and budget are genuinely comparable. Tests are smoke
checks, not measured learning success. Do not replace existing baselines.
