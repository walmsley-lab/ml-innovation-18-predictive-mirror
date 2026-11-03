# Synthetic world ladder: 2D -> metric 3D

**PR 2 of the research roadmap** establishes an optional 2D robustness
generator and the data/visibility boundary for future 3D worlds. It does
**not** claim to produce or infer 3D.

## Why add a 3D interface now, not a heavyweight renderer?

A controlled 2D world keeps local CPU experiments fast and falsifiable.
Rendered 3D is warranted once the benchmark and communication curriculum
have reproducible results, preferably around the developmental transfer
experiment. Starting with camera motion, depth ordering and parallax is
more informative than immediately adding complex game-engine physics.

Candidate generator choices:
- MuJoCo: controllable physical interactions and camera movement;
- Kubric: Blender/PyBullet multi-object videos with depth, flow, masks;
- BlenderProc: appearance-rich renders, camera geometry, depth and normals.

Choose based on a **minimal headless smoke with pinned dependency versions**,
not aesthetics. Do not make a heavy simulator a required JAX/test dependency.

## Current CPU generator

```bash
python -m experiments.rgb_ood_world --domain train --episodes 4
python -m experiments.rgb_ood_world --domain ood_motion --episodes 4
python -m experiments.rgb_ood_world --domain ood_appearance --episodes 4
python -m experiments.rgb_ood_world --domain ood_camera --episodes 4
python -m unittest tests.test_rgb_robust_world -v
```

Domains are deterministic by seed, but future camera actions may be random
**from the agent's viewpoint**. The generator exposes `camera_actions[t]`
for the commanded camera displacement between frame t and t+1. RGB-only
predictors must not be penalized as if they knew those random future actions.
An action-conditioned benchmark can legitimately provide commanded actions,
but never the future rendered frames or true object states.

## Frozen 3D world sample contract (proposed, not implemented)

One episode should eventually store:
- model input: `rgb[T,H,W,3]`, and optionally **known/planned**
  `camera_actions[T-1,A]`, with a clear timestamp/frame convention;
- calibration: camera intrinsics `K[3,3]`, extrinsics
  `T_world_camera[T,4,4]` for the benchmark and explicitly authorized
  action-conditioned controls;
- evaluator-only: `depth[T,H,W]` in meters, `instance_id[T,H,W]`,
  object poses/identities, optical flow, contact events and visibility;
- provenance: simulator, version, physical units, seeds, asset hashes,
  fixed episode split, camera FPS and realized timestamps.

Do not send privileged depth/pose to an RGB-only model. A pose-conditioned
or depth-supervised variant is a *separate* experiment with its own controls.
Train on worlds with held-out camera trajectories, shapes, materials and
compositions. Measure parallax, depth ordering, 3D location, reappearance
and metric/relative depth errors before claiming learned perspective.

**Promotion gate:** frozen event benchmark + strong 2D baseline + explicit
input contract + separately verified 3D renderer smoke. Only then promote
3D-rendered sequences as a new training family. Retain the 2D control.

The initial `src.rgb_robust_world` is deliberately a separate generator;
the earlier `src.rgb_motion_world` frozen protocol remains unchanged.
