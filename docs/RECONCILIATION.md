# PR 6 — Exploratory reconciliation diagnostics

```bash
python -m experiments.reconciliation --parameters runs/multisheet/parameters.npz
```

Freeze the trained model. Fit independent **post-hoc** ridge probes to fast- and slow-sheet activity on independent synthetic episodes. Evaluate their **next-frame retinal object-x** predictions on unseen episodes. Score both raw agreement and *accuracy-qualified* agreement when the object is not visible. Repeat with feedback cut at test time using the intact probes.

This is an **exploratory diagnostic**, not a causal proof of representational reconciliation. Still missing: no-feedback-trained control; random within-sheet lesion and mean-feedback replacement; single-sheet split-decoder null; matching runtime/compute; within-frame inference-iteration trajectories. These are explicit gates before making a reconciliation claim. At low training steps, both probes may be inaccurate; report that failure rather than presenting agreement alone.

No simulator truth enters the trained network; labels are exposed solely to this post-hoc evaluation and probe fitting. Probe-training seeds and evaluation seeds are disjoint. This PR is intentionally narrow to keep the mathematical and causal questions separate.
