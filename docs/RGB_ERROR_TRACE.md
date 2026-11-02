# PR 55 — Within-episode local RGB error traces

A standalone **adaptive state** probe using an already trained and frozen
RGB NCA checkpoint. This does **not** train synaptic weights or implement
true Hebbian plasticity. We explicitly distinguish local transient sensory
error adaptation from changing the neural cell update rule itself.

The three arms share exactly the same checkpoint and observed history:

- **Frozen:** original RGB NCA prediction.
- **Local trace:** exponential moving average of each pixel's observed
  prediction error, applied locally as a decaying forecast correction.
- **Shuffled trace:** same trace values applied at displaced pixels, testing
  whether locality of adaptation matters.

Traces update only when observations arrive. They decay open-loop and
never consume future images. Score the same held-out synthetic scenes
and image-only persistence/velocity controls.

    python -m experiments.rgb_error_trace \
      --checkpoint runs/rgb-nca-1s/parameters.npz \
      --context 10 --horizon 10 --out runs/rgb-error-trace

    python -m unittest tests.test_rgb_error_trace -v

The checkpoint SHA256, forecast differences, per-lead metrics and a saved
example are recorded. The caller must confirm test seeds were not used
for training/selection. A later PR could test learned local Hebbian
parameters, but this experiment makes no such claim.
