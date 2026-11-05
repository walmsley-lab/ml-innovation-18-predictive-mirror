# Bidirectional multiscale field prediction

A **fine-resolution** field and a **coarse-resolution** sheet exchange
messages in both directions. Coarse-to-fine feedback and fine-to-coarse
aggregation are tested by frozen-weight direction ablations.

Controls: fine-only, equal-core extra iterations, dilated convolution,
and an encoder–decoder coarse branch with one-way top-down fusion.
Every arm gets the same full `(u,v)` physical state and held-out
wave/diffusion trajectories; the inputs and update budgets are matched,
but active parameter counts and FLOPs need independent profiling.
Do not interpret the number of extra channels as perceptual success.

```bash
python -m experiments.field_multiscale_comparison --kind diffusion --steps 30
python -m experiments.field_multiscale_comparison --kind wave --steps 30
python -m unittest tests.test_field_multiscale -v
```

Completion evidence: improvement in held-out per-lead RMSE **over
extra ticks, dilation and encoder–decoder**, *and* both direction
ablations worsen error on a frozen trained predictor. Prediction
changes alone do not establish usefulness. No advantage is claimed
from these smoke-tested implementations.
