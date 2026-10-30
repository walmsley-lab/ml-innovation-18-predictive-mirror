# PR 22 — Active sensing with optional local cellular gain adaptation

This small synthetic system takes a **trained recurrent model** and gives it a movable camera. Compare equal-budget fixed gaze, seeded random gaze and a deterministic model-driven rule: choose the permissible movement maximizing predicted-pixel uncertainty `mean(p*(1-p))` **before** seeing the next observation. No RL and no simulator truth reaches policy selection. Every sensing condition can run with the same frozen recurrent network and with/without an **online per-cell local lateral gain**.

```bash
python -m experiments.active_field --parameters runs/pilot/parameters.npz \
 --episodes 16 --steps 12 --out runs/active-plasticity.json
```

The baseline has gain exactly zero and should reproduce the original recurrent update. The plastic variant adds a bounded online gain updated from nearby activity correlation; frozen shared convolution kernels never change. This is **not a complete neural cellular automaton or individual synapse-level Hebbian plasticity**. Gains introduce additional state, so adaptive and fixed controls differ in effective memory and must be reported as such.

Report both **prediction MSE** and **fraction of frames where the object is visible**. Choosing empty/easy scenes can deceptively improve MSE; this benchmark **does not** establish useful uncertainty reduction or world-state understanding. A stronger subsequent experiment would compare held-out hidden-state inference at a standardized final viewpoint and include optimized-fixed or replay-yoked gaze. No biological or emergent perception claim is justified by these simple exploratory observations.
