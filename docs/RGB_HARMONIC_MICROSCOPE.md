# RGB harmonic microscope

**Question:** Do the trained RGB NCA's hidden states preferentially occupy the
low spatial modes of its **fixed 4-neighbor, non-periodic image lattice**?

Run after an existing RGB NCA checkpoint is available:

```bash
python -m experiments.rgb_harmonic_microscope \
  --checkpoint runs/rgb-intro/seed-11-curriculum.npz \
  --episodes 4 --context 8 --horizon 10 --modes 16
python -m unittest tests.test_rgb_harmonic_microscope -v
```

No retraining. Compares (a) checkpoint hidden states, (b) separately seeded
untrained zero-readout NCA, and (c) a *fixed* spatial shuffle of checkpoint
hidden states. The shuffle is analysis-only; its forecast MSE is intentionally
undefined. Hidden state is recorded after each observation correction and
during prediction-only dreaming; future labels never enter the model.

The modes are exact cosine eigenvectors of the graph Laplacian on an open
rectangular grid, excluding the constant mode and preserving degenerate
eigenvalue groups. Power is calculated after channelwise spatial centering;
the denominator is **all nonconstant spatial energy**, including modes not
explicitly retained, so selected fractions need not sum to one.
`metrics.json` includes the checkpoint SHA and reproducibility settings;
`example.npz` contains a trace for inspection.

**Interpretation:** every field admits a harmonic decomposition. Differences
between trained, untrained and shuffled spectra are descriptive. They do not
show temporal oscillation, representation, or causal contribution to prediction.
Use a separate, matched-energy intervention to test functional necessity.
External checkpoint/train-split provenance is still required for held-out claims.

**Related:** [Connectome Harmonics reanalysis](https://github.com/fractastical/connectome_harmonics)
and our [Connectomics research](https://github.com/walmsley-lab/ml-innovation-10-connectomics).
