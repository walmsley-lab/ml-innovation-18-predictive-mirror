# Neuromatch bridge: vision and dynamics probes

Four small additions to the frozen RGB NCA research workflow:

1. **Sparse assimilation**: update only observed, high-surprise pixels and report
   update counts alongside dense reconstruction error. This is an observation
   assimilation reference, not yet a sparse JAX convolution.
2. **Divisive normalization**: test a fixed spatial luminance circuit under
   a known gain shift; this measures photometric sensitivity, *not* trained
   forecasting improvements. Train matched models before adoption.
3. **Temporal representation comparison**: velocity-kernel CKA on observed
   hidden-state trajectories, with time-shuffled control. This is a small
   proxy inspired by dynamic similarity, **not DSA**.
4. **Causal channel ablations**: after conditioning on observed RGB, mask
   high-energy channels vs randomly chosen channels with similar summed energy.
   The ranking uses context alone. Repeat across trained checkpoints, held-out
   episodes, and seeds; energy equivalence is approximate.

Run from the repository root:

```bash
python -m unittest discover -s tests -p 'test_neuromatch_vision.py' -v
python -m experiments.neuromatch_vision --out runs/neuromatch-vision/metrics.json
python -m experiments.neuromatch_vision --parameters runs/my-trained-model.npz
```

A random untrained checkpoint is a **smoke control** only; it cannot show
that a learned circuit is causally useful. This bridge complements the existing
`research18_event_updates`, `rgb_hidden_interventions`, and
`rgb_mode_interventions` experiments rather than replacing them.

Reference material: Neuromatch Academy NeuroAI microcircuits, comparing
networks, and computational neuroscience causal inference tutorials.

## November evidence safeguards

- Sparse-change selection currently reads two *entire* RGB frames to find
  changes. Report the full scan alongside selected writes; do not claim actual
  GPU computation or energy savings.
- The closest-energy alternative lesion is not uniformly random and can overlap
  the targeted group. Report the overlap and a separate explicitly *disjoint*
  comparison; different removed-energy totals prevent a causal ranking claim.
- Use `--size 32 --horizon 10 --ticks 2` to match the synthetic RGB task;
  a 16-pixel/3-frame diagnostic cannot explain 32-pixel/10-frame failure.
- Contrast-normalized sensitivity is not a trained predictor improvement.
  Self-CKA is a sanity check; add trained-vs-random-initialized trajectory CKA
  and many independent episodes before interpreting similarity.
- The PNG activity heatmap is now per-episode percentile-scaled and the JSON
  includes absolute activity statistics. A uniform display color is no longer
  mistaken for evidence of uniformly maximal hidden activity.
