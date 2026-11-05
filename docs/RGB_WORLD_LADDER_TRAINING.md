# Matched training on the richer perceptual worlds

This experiment tests **task suitability**, not just architectural
capacity. It runs the existing CNN, transport, pixel NCA, soft-routing
NCA and hierarchical predictors under the **same RGB-only objective**
with the same minibatch/update budgets. No architecture is deleted.

```bash
# Smoke (no substantial research claim):
python -m unittest tests.test_rgb_world_ladder_training -v
# Substantive multi-seed study (not auto-launched by merging):
python -m experiments.rgb_world_ladder_training \
  --seeds 11 23 41 --steps 1200 --context 8 --horizon 10
```

Two curricula: linear-only and varied training on acceleration,
collisions, short occlusions and size/appearance. Independent
validation sets select checkpoints using final-lead foreground MSE,
not image background MSE. Frozen test domains add long occlusion,
camera pan and combined distribution shifts. All arms are *action
blind*; camera motion is deliberately held-out input uncertainty.

Per-lead event scores include soft centroid, predicted mass, hard
object detection, foreground segmentation, re-entry and MSE.
Report step-zero checkpoint eligibility, validation history,
parameter counts, saved SHA and training wall-clock. Last-step
optimizer moments are snapshotted separately from best weights;
a complete deterministic resume/kill-restart test is still required
before using those files as production resumable training.

An optional `nca_blend` arm becomes available **after** the
independent training-budget PR's `rgb_grounded_nca` is merged:
`--models cnn transport nca routing hierarchical nca_blend`.
The initial PR has no dependency on that unmerged source.

**Interpretation:** a CNN beating NCA on short/simple motion does not
justify deleting recurrence. Conversely, longer training and varied
data alone do not establish object permanence. Compare held-out
barrier reappearance, soft mass, latency and compute against each
family's appropriate classical control.
