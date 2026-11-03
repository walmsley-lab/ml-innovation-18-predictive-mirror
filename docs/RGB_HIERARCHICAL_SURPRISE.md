# Hierarchical predictive coding: small falsification pilot

This separate RGB model has fast full-resolution hidden cells and a
half-resolution slow sheet. After each observed context frame, the hidden
state is corrected using sensory prediction error. Future steps use only
prior predictions, with no observed correction.

```bash
python -m experiments.rgb_hierarchical_surprise --steps 60 --seeds 11 23 41
python -m unittest tests.test_rgb_hierarchical_surprise -v
```

Ablation arms: flat RGB recurrence, two-scale RGB recurrence,
two-scale plus auxiliary latent alignment, two-scale plus learned
sensory-error gain. The latent objective matches dream-state channels to
a frozen-gradient encoding of future training frames. This uses future
targets **only in the supervised loss**, not in inference. The latent
encoding is not pretrained, semantically interpreted or JEPA-calibrated.

The gain is not a probabilistic variance estimator. Compare measured
RGB forecast errors, seed variation and latency, not visually pleasing
hidden-channel maps. The two-scale arms use additional active compute,
so pixel-MSE improvement alone cannot prove a hierarchical mechanism.
Original RGB NCAs and other phase/fast-slow models remain unchanged.
