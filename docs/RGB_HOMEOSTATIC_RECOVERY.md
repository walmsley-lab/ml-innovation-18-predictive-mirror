# RGB NCA with local activity-dependent recovery

**Question:** Does a tiny, bounded local negative-feedback variable help
a predictive cellular field remain stable and improve generalization?

A cell's recovery trace is an exponential average of its squared bounded
activity. On each recurrent tick a local feedback term opposes its present
activation; **no spatial eigenmodes or phase oscillators are injected**.
This is a toy regulatory dynamical system, *not* a biophysical cell model or
a demonstration of chemical equilibrium.

Compare three independently trained arms:
- `plain`: unchanged original RGB cellular update;
- `local_recovery`: per-cell activity trace fed back at the same site;
- `shuffled_recovery`: same trace computation, but feedback spatially rolled
  to the wrong sites (a locality control).

All arms begin with the **same base weights** for each model seed, have the
same trainable parameter count, consume the same training examples in the same
order, and receive the same number of optimizer steps. All arms carry the same nontrainable recovery trace **per pixel and hidden
channel**; the plain arm computes it but ignores it for its update. This
matches state budget while leaving a small feedback-operation cost difference,
which should be profiled before making a compute-matched claim. Step-zero checkpoints remain
eligible during validation-based selection.

```bash
python -m experiments.rgb_homeostatic_recovery \
  --seeds 11 23 41 --steps 60 --context 8 --horizon 10
python -m unittest tests.test_rgb_homeostatic_recovery -v
```

Outputs include independently selected checkpoints, validation and held-out
RGB MSE by lead, last-frame and constant-velocity controls, and diagnostic
hidden/recovery magnitudes **and normalized adjacent-cell spatial\nroughness** (mean squared neighbor difference divided by spatial variance).\nA lower roughness value alone can reflect oversmoothing; interpret it jointly\nwith nonzero spatial variance and held-out forecast accuracy. During open-loop dreaming the model gets no new
sensory evidence or future simulator state.

**Claim gate:** held-out improvements beyond strong forecasting baselines
*and* the spatially scrambled recovery control, repeated across seeds,
with stable bounded dynamics. Smaller hidden magnitude or lower spatial roughness alone is not a win.
No fitted weights or experimental results are claimed by this protocol.

**Related:** [Connectome harmonics](https://github.com/fractastical/connectome_harmonics)
and [cellular-regulation hypothesis](https://github.com/walmsley-lab/ml-innovation-10-connectomics).
Independent of the existing [phase entrainment](RGB_PHASE_ENTRAINMENT.md)
and [error-trace](RGB_ERROR_TRACE.md) experiments.
