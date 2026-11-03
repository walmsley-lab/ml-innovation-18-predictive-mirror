# Frozen RGB training and evidence campaign

Run paired training on **the same RGB episodes** for a local recurrent NCA,
a small CNN, and the earlier transport+residual model. Report multi-lead
scores, paired episode-bootstrap uncertainty relative to image-derived
velocity, checkpoint selection (step zero eligible), parameter count and
wall-clock time including JIT compilation.

```bash
python -m experiments.frozen_rgb_campaign --seeds 11 23 41 --steps 100
python -m unittest tests.test_frozen_rgb_campaign -v
```

This is a reproducible **protocol**, not proof of improvement. Default
steps are a pilot budget; increase and verify convergence for research
claims. Inputs/test seeds are frozen, but architectures differ in loss,
FLOPs, active parameter paths and recursive computation. Do not claim
capacity matching merely because training updates and hidden widths match.
No run results are asserted until actual artifacts exist. Separate
unseen-camera and independent webcam evaluations remain necessary.

Before merging the old developmental curriculum draft, compare its
staged/shuffled data protocol with the same frozen evaluation contract.
