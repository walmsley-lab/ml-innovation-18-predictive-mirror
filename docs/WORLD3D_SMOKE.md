# Calibrated 3D perception worlds

The initial headless benchmark uses NumPy ray-sphere intersections and
**metric** perspective projection. This is a genuine 3D camera with parallax
and depth ordering, but not a general-purpose simulator. Spheres follow
simple deterministic motion; no contact dynamics or deformable objects.

```bash
python -m experiments.world3d_generate --backend analytic --episodes 4
python -m unittest tests.test_world3d -v
```

An **optional** actual MuJoCo RGB smoke is available only after separately
installing and configuring an appropriate headless OpenGL backend:

```bash
# Optional isolated environment, not in default CI requirements
# python -m pip install mujoco
# MUJOCO_GL=egl python -m experiments.world3d_generate --backend mujoco
```

The two renderer outputs are **not** currently pixelwise equivalent, and
the MuJoCo adapter exports RGB only. Do not mix them as identical training
data. The analytic world stores RGB, metric Z depth, instance IDs, camera
intrinsics, camera-to-world extrinsics, planned x/yaw actions and timestamps.
Only RGB (and optionally *known future actions*) are model inputs.
Depth/poses/identities remain evaluator-only. Test procedural scene splits
before making any 3D-generalization claim.

MuJoCo rendering reference:
https://mujoco.readthedocs.io/en/stable/python.html
