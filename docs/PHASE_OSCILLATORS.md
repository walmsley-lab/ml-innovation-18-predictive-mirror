# Phase oscillator ablation: coupled rhythm versus recurrent memory

This experiment adds a **fixed-rule phase channel per pixel** to the 16-channel
webcam residual predictor. Phase evolves with a natural frequency, a weak
observation-driven perturbation, and local Kuramoto-like eight-neighbor coupling.
The phase modulates the amplitude of recurrent cell updates; the output is still
a persistence-plus-residual next-frame forecast. No biological equivalence,
binding, object representation, or synchronization benefit is assumed.

Three paired arms, same parameter count, initial weights, real-data batches,
100-step training objective, and chronological holdout:

1. Original video residual network (no phase state).
2. Oscillatory network with coupling K=0 (phase modulation only).
3. Oscillatory network with K=0.25 (neighbor phase coupling).

The oscillator arms **do add nontrainable recurrent state and operations**;
the comparison is matched on trainable parameter count and updates, not runtime
or state count. Coupling effects may be negligible or negative, especially on
unstructured webcam motion.

```bash
python -m experiments.phase_ablation --input recordings/webcam.npz \
  --steps 100 --coupling 0.25 --out runs/phase-ablation
cat runs/phase-ablation/metrics.json
```

Score overall and changed-pixel MSE against persistence. The real test is
whether a coupling gain reproduces across seeds, multiple independent
recordings, occlusion durations and future horizons. Phase alignment by itself
is not evidence of neural binding or spatial reasoning. The saved oscillator
checkpoints require the oscillator model, not the stock live mirror adapter.
