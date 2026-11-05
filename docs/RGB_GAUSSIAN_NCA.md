# RGB-inferred Gaussian-state cellular predictor

**Third Gaussian gate: remove privileged scene-state inputs.** A small
set of *learned color slots* infers soft RGB centroids, approximate depth
from apparent size, color and opacity from past frames. The camera model
integrates **known commanded** x/yaw actions from a fixed origin. A
shared local 3D cellular rule then forecasts these Gaussian centers,
and a differentiable soft Gaussian renderer predicts future RGB.

```bash
python -m experiments.rgb_gaussian_nca --steps 80 --seeds 11 23 41
python -m unittest tests.test_rgb_gaussian_nca -v
```

Inputs at inference: RGB history, fixed intrinsic K, planned camera
actions. **Not** supplied: simulator object centers, radii, IDs, depth,
extrinsics or future RGB. Ground truth and masks enter only training
loss or evaluator after committed predictions. This method does not
guarantee actual object discovery: slots are fixed in number, the
background has a known approximate range, and camera origin/kinematics
are assumed. Test identifiability by swapping appearances, occlusion,
unseen objects and viewpoints.

Training is from pixel reconstruction + prediction, not an oracle
3D-position loss. Compare static-slot, constant-velocity-slot and
cellular-slot dynamics *sharing exactly the same RGB inference*;
also report independent RGB-only persistence, image velocity, CNN,
transport and pixel NCA controls on matching held-out worlds.

**Stop gate:** before any claims, demand non-collapsed objects, stable
slot assignments, nonzero action counterfactual response, reappearance
and held-out-view gains against simple baselines. Do not auto-promote
a low full-frame MSE result. This is a minimal CPU experiment, not a
photorealistic Gaussian Splatting implementation.
