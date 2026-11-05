# Diverse perceptual-world ladder

This is a **separate** deterministic generator and frozen evaluation runner.
The older RGB-motion benchmark and its seed meanings are unchanged.

Question: does more varied training/evaluation demand recurrence, routing,
or persistent scene representations beyond two-frame CNN/transport baselines?

Domains: linear motion, acceleration, scripted two-object collision,
short/long occlusion stripes, prescribed camera pan, changed scene
background/size, and combined changes. Rendered RGB is model input;
positions, events, velocities and visibility reasons are evaluator-only.
Random initialization and held-out episode roots are explicit.

```bash
python -m experiments.rgb_world_ladder --episodes 12
python -m unittest tests.test_rgb_world_ladder -v
```

This PR establishes **worlds and observation-only controls**, not a new
trained world model. To test the repertoire hypothesis, train every
existing architecture on *identical* training scenes and test against
the frozen 91000+ episodes; record per-lead foreground IoU, soft
centroid + mass, barrier reappearance, and latency as well as MSE.
The existing event evaluator may score off-screen reentry as return
after invisibility: use `visibility_reason` to separate true barrier
occlusion from camera exit before claiming object permanence.
No paid compute or larger training run is triggered by this change.
