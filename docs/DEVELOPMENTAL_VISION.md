# Developmental vision progression

The developmental experiments are deliberately small and independently runnable.
They add one capability at a time while preserving the existing experiments and
their seeds, manifests, checkpoints, and evaluation contracts.

## Progression

1. `retinal_waves.py` creates bounded traveling-wave observations from the
   checked physical field implementation. Only displacement is exposed.
2. `spatial_plasticity.py` presents visual activity to a local sheet. Surprise
   updates a bounded trace and activity-dependent connections, with population
   competition and a homeostatic firing target; no fixed visual checkpoint is
   required.
3. `developmental_vision.py` runs the same sheet continuously from wave-derived
   retinal RGB into perspective visual experience, measuring whether prediction
   error falls without resetting the learned organization at the boundary.
4. `active_3d_learning.py` replaces the source with perspective-rendered 3D
   RGB observations and camera actions. Camera action signals enter the local
   update and plasticity loop, so changing viewpoint changes what is learned.
   The learner still receives only the 2D
   retinal sheet; depth, pose, identity, and simulator truth remain evaluator
   state. The same online loop can also be used with the webcam source in
   `developmental_interaction_path.py`.

Run smoke versions with:

```bash
python -m experiments.retinal_waves --frames 8 --size 16
python -m experiments.spatial_plasticity --frames 8 --size 16 --sheet 4
python -m experiments.developmental_vision --frames 8 --size 16 --sheet 4
python -m experiments.active_3d_learning --frames 8 --size 16 --sheet 4
```

Each command writes a small `metrics.json` (and the wave stage writes
`observed.npy`) beneath its run directory. Seeds and physical parameters are
recorded in the output. These are developmental smoke experiments, not claims
of webcam generalization or learned metric 3D reconstruction.

The shared `DevelopmentalSheet` remains in the existing interaction experiment
because it is the one reusable model needed to connect the stages. Existing
experiments are not imported into or modified by these new stages except for
that explicitly shared learner and the existing checked world generators.
