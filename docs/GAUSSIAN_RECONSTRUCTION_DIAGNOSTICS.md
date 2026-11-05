# Gaussian representation quality before future prediction

**Question:** can the RGB-inferred Gaussian slots reconstruct the **current**
observation, and how much of the error is due to the renderer versus inference?

Compare one fixed observed frame across three views: analytic ray-sphere RGB
truth, **oracle simulator geometry** rendered with the same soft Gaussian
rasterizer, and the **RGB-inferred Gaussian** rasterization. Report full and
visible-geometry MSE and per-frame permutation-invariant projected-center
errors, with a side-by-side PNG gallery per seed. An oracle reconstruction
is **not a learned model** and has privileged geometry.

```bash
python -m experiments.gaussian_reconstruction_diagnostics \
  --checkpoint runs/rgb-gaussian-nca/seed-11.npz
# Wiring only; do not cite as a learned reconstruction:
python -m experiments.gaussian_reconstruction_diagnostics \
  --untrained-smoke --seeds 11
python -m unittest tests.test_gaussian_reconstruction_diagnostics -v
```

The oracle mismatch measures a representation/rendering limitation for the
simple analytic sphere world. Excess inferred MSE is a **signed comparison**
within the common renderer, not an additive, guaranteed-positive causal
decomposition. Slots are matched *per frame*, not assumed to retain
identity, and their spatial localization is only one aspect of reconstruction.
This experiment uses frame `context-1` and known planned camera actions;
future images and evaluator-only positions are never passed to inference.
