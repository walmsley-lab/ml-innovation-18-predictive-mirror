# Predictive Mirror — forward research timeline

**Planning note, November 2026.** This page records conditional research milestones, not a promise of calendar dates or implemented experiments. See [experiment map](EXPERIMENTS.md) for the actual code, and [historical charter](PLAN.md) for the original hypotheses.

| Sequence | Focus | Decision gate |
|---|---|---|
| 1 · Current | Validate object-centric metrics, collapse diagnostics and non-saturating NCA alternatives | Independent object location/mass, saturation and held-out evidence rather than RGB MSE alone |
| 2 · Next | Expand world diversity: velocities, multiple objects, collisions, occlusion durations, appearance, camera motion | Show which existing CNN, transport, recurrent, routing and hierarchical mechanisms learn/generalize under matched budgets; **preserve all variants** |
| 3 · Next | Strengthen controlled metric-3D prediction and camera-action pathways | Reproducible RGB/depth/pose ground truth for the evaluator; demonstrate action-sensitive forecasts before testing active gaze |
| 4 · Later | **Gaussian-state NCA** (future track, below) | A small, reproducible prototype beats its strong controls on object persistence or novel-view accuracy, not just image MSE |
| 5 · Conditional | Learned scene inference, active perception and transfer | Held-out geometries/camera paths, interventions and compute-matched benefits; only then consider larger renderers or real video |

## Gaussian-state NCA (future track)

**Hypothesis.** Recurrent cellular dynamics may work better on a compact, persistent **3D scene representation** than on an RGB-aligned pixel sheet. Represent a scene by a modest number of Gaussians (e.g. 8–64), each with 3D center, size/covariance, color and opacity; optionally velocity and a learned hidden state. A shared, local or radius-neighbor cellular update predicts how these properties evolve. A renderer produces a *committed* future RGB view from the predicted scene and the **known/planned** camera action.

**Important distinction:** static 3D Gaussian splatting renders a scene; even time-indexed/4D splats do **not automatically constitute a predictive world model**. This experiment specifically trains dynamics to forecast *unobserved* future states/views.

**Minimal progression (separate, KISS experiments):**

1. **Oracle-state sanity check:** start with known simulator Gaussians/primitive geometry and known camera actions. Test 3D Gaussian dynamics + rendering against frozen/static splats and constant-velocity Gaussian movement. Ground-truth positions/depth are allowed **only in this explicitly privileged arm**, never silently handed to an RGB-only predictor.
2. **Observation-to-state learning:** infer/update Gaussians from past RGB frames; train the predictor open-loop on future observations. Add zero/incorrect camera-action and static-state controls, with a distinct comparator trained using privileged depth/poses only if labeled as such.
3. **Reappearance, perspective and action:** evaluate held-out object trajectories, occlusion lengths, geometry, appearances and camera viewpoints. Compare image-space CNN, transport, pixel NCA (including non-saturating variant) and Gaussian-state NCA on **identical observable input and training splits**. Include 2D particle/graph controls only when their extra supplied state is marked privileged.

**Required measurements:** per-horizon foreground IoU/recall, hue-robust soft centroid + predicted object mass, object reappearance, whole-RGB MSE, novel-view/parallax consistency, response to counterfactual camera actions, readout/clip saturation, parameter count, training steps/FLOPs and inference latency. Report uncertainty across seeds and evaluation episodes. Keep withheld object identity, metric depth and future frames evaluator-only for RGB-input arms.

**Promotion gate:** demonstrate a *measured* improvement on at least one held-out object-persistence or viewpoint task over suitably matched pixel and static/constant-motion controls, without regressing badly elsewhere. If no gain, retain the experiment and the negative result; do not replace or delete existing architectures.

**Scope guard:** start in the existing calibrated analytic 3D world at 16–32px with a tiny number of primitives, JAX/NumPy CPU-compatible functions and no mandatory PyTorch/CUDA/3DGS engine. Heavy splat rendering or real-time webcam integration comes **after** this evidence, not before.
