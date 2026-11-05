# Interpretable SINDy on observed physical variables

```bash
python -m experiments.interpretable_sindy --seeds 11 23 41
python -m unittest tests.test_interpretable_sindy -v
```

This first SINDy experiment fits a sparse polynomial difference
equation to **provided** object `x,y,vx,vy` trajectories, including
acceleration/collision held-out domains. Controls are least-squares
affine dynamics, linear DMD through the origin and constant-velocity
coasting, all using the same privileged physical variables. The
standardized sparse regression resists feature-scale artifacts.

Training noise, seed and distribution are recorded; forecast errors
are in physical simulator pixels / per-generated-frame units.
The resulting equations are interpretable functions of supplied
state, not explanations of a trained NCA. Do not compare these errors
numerically against RGB-only forecasting models. PR17 tests frozen
learned hidden-state descriptions separately.
