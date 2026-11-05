# Are inferred Gaussian slots temporally coherent?

This diagnostic repeats the **existing RGB encoder** separately over
each observed frame and matches its scene elements to simulator objects
using *evaluator-only* projected center positions. Assignment is
permutation invariant: a change in raw slot index counts as harmless
reordering when both adjacent localizations remain accurate.

It reports per-frame visible/partial/hidden object support, soft
projected-center errors in pixels, raw slot switches, permutation-only
switches, unresolved switches, and missing/wrongly localized visible
objects. An explicit overlapping-sphere scene supplements ordinary
motion and `ood_camera` views; camera actions are known, not inferred.

```bash
python -m experiments.gaussian_correspondence \
  --checkpoint runs/rgb-gaussian-nca/seed-11.npz
python -m experiments.gaussian_correspondence --untrained-smoke --seeds 11
python -m unittest tests.test_gaussian_correspondence -v
```

This is **not** identity learning: object labels are available to the
evaluator, but never to the encoder. Hidden objects are not treated
as successfully tracked. The primary capability is whether the
inferred representation remains well localized up to permutation,
not whether a particular numeric slot name stays fixed.
