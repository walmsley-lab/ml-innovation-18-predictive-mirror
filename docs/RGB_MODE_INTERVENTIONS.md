# Causal intervention on spatial harmonic modes

**Question:** Are low spatial modes causally useful for a frozen RGB NCA's
future-frame forecasts, beyond the effects of deleting an equivalent amount
of hidden-state activity?

```bash
python -m experiments.rgb_mode_interventions \
  --checkpoint runs/rgb-intro/seed-11-curriculum.npz \
  --episodes 12 --context 8 --horizon 10 --mode-count 16
python -m unittest tests.test_rgb_mode_interventions -v
```

- Use the exact **open-grid Laplacian eigenmodes** from the harmonic microscope,
  and keep whole eigenvalue-degenerate groups at the low-mode cutoff.
- Replay the *same* frozen model, episodes, observation prefix and forecasting
  interval for three arms: intact; suppress low-mode content; and suppress a
  random orthonormal spatial subspace. The random control matches the target
  removal's L2 energy **per sample and hidden channel at the intervention**.
- Intervention is applied once, just before forecast lead `intervene_at+1`.
  Later dreams see only earlier forecasts, never future ground truth.
- Report by-horizon RGB test MSE **minus intact MSE**, prediction change,
  persistence/velocity baselines, test-seed provenance and checkpoint SHA.
  A positive delta means worse forecasts. A causal *benefit* requires targeted
  removal to hurt more than the matched random intervention, not merely to
  change the output.

Spatial modes are not necessarily stable physical waves; the lattice is a
computational scaffold, not anatomical connectomics. Compare multiple
checkpoints, seeds, low-mode cutoffs and independent scenes before broad claims.
A zero-readout or persistence-dominated model is an expected **negative
control**: both perturbations can have no effect. This experiment introduces
neither learned dynamics nor new parameters.

**Depends on** [RGB harmonic microscope](RGB_HARMONIC_MICROSCOPE.md) for
eigenmodes and descriptive spectra; the existing RGB NCA remains unchanged.
