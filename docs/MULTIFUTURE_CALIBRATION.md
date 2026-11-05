# Multiple coherent futures under irreducible visual ambiguity

```bash
python -m experiments.multifuture_calibration
python -m unittest tests.test_multifuture_motion -v
```

A paired toy scene has **identical complete observed RGB histories**
but two possible future vertical turns. The branch is withheld from
the predictor. No deterministic forecaster can always select the
correct continuation, making it an appropriate small test of the
distinction between *prediction accuracy* and *calibration*.

The reference produces two spatially coherent whole-trajectory
hypotheses, with frequencies fitted from training-only branch labels.
This is a **classical multimodal floor**, not a learned latent mixture
inside the NCA. It is compared to single-path constant-velocity
coasting on the same history, and evaluated with:

- Multivariate **energy score** (proper distributional score).
- Top-probability trajectory error and probability-weighted ADE.
- Coverage across future frames and between-hypothesis diversity.
- Branch Brier/calibration error, with independent *paired histories*
  rather than framewise confidence intervals.
- Best-of-many ADE as an **optimistic diagnostic only**, not an
  accepted competitive criterion.

The hypothesis family itself is known and should be disclosed:
high performance here alone does not demonstrate that a model
discovered new motion modes. Next experiments can train a latent
NCA/conditional mixture on these same ambiguous scenes, and require
well-calibrated probabilities without collapse or mode memorization.
