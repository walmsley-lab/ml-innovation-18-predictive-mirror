# Camera actions from frozen RGB-NCA ensemble disagreement

Depends on action-conditioned 3D NCA training. The policy receives
**only observed RGB frames, past actions, and candidate next actions**.
It chooses the candidate with highest forecast disagreement between
frozen model checkpoints. Fixed/random policies see identical hidden
world dynamics. All three are evaluated on their predicted RGB at the
same *home viewpoint* one step after their final observation.

```bash
# Explicitly untrained wiring test (no research claim):
python -m experiments.active_world3d --untrained-smoke --seeds 11 --steps 3
# Research: train and freeze independent action model checkpoints first.
python -m experiments.active_world3d --checkpoints runs/a.npz runs/b.npz
python -m unittest tests.test_active_world3d -v
```

Disagreement is **not calibrated predictive uncertainty or proven
information gain**. Promoting the policy requires frozen trained
ensembles, paired world seeds, a replay-yoked action control and
standardized-view performance gains. This experiment neither accesses
privileged depth during action selection nor learns a policy via RL.
