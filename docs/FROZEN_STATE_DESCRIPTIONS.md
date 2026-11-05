# Sparse descriptions of learned cellular dynamics

Requires [PR16](INTERPRETABLE_SINDY.md)'s sparse identification functions.

```bash
python -m experiments.frozen_state_descriptions --untrained-smoke
python -m experiments.frozen_state_descriptions --checkpoint runs/.../seed-11.npz
python -m unittest tests.test_frozen_state_descriptions -v
```

A frozen NCA's hidden channels evolve under constant observed RGB
and zero surprise. We record per-channel spatial mean, fit a four-
coordinate PCA on training traces only, and compare SINDy with affine,
DMD and persistence on disjoint held-out seeds. The fixed coefficient
model is evaluated on ordinary evolution and **hidden perturbations
not seen during fitting**.

Caveats: spatial averaging discards local geometry; PCA is not
guaranteed to yield stable semantic variables; the sparse model
omits the fixed RGB driving signal; and a close fit can be descriptive
without being causally necessary. `--untrained-smoke` is explicitly
not evidence of learned dynamics. For research claims use a training
checkpoint with a verified provenance and independently frozen
evaluation units. A separate intervention on the actual NCA should
confirm any proposed mechanism.
