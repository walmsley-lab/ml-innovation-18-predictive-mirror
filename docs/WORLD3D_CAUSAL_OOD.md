# Frozen causal tests of action-conditioned 3D world prediction

Depends on the action-conditioned NCA and 3D world generator.
With **frozen checkpoint weights**, compare intact prediction to:
zeroing hidden state, spatially permuting hidden state, hiding camera
actions, and shifting action sequences across episodes. Evaluate each
control on train-family, held-out depth, camera and appearance domains.

```bash
python -m experiments.world3d_causal --untrained-smoke --episodes 2
python -m experiments.world3d_causal \
  --checkpoint runs/action-world3d-nca/seed-11-action_aware.npz
python -m unittest tests.test_world3d_causal -v
```

The untrained mode is a **wiring test only**. Held-out seed and domain
splits must be frozen before model selection. Changing hidden states and
observing different predictions demonstrates dependence, not that the
state encodes causal object properties or learned 3D geometry. Report
paired differences, failure cases, and checks for action-information
leakage before any representation claim.
