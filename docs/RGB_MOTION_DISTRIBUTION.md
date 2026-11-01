# Step 6 — RGB motion NCA: distribution training versus memorization

This is a standalone benchmark in experiments/rgb_motion_distribution.py.
The original Step 5 sweep implementation and local
runs/rgb-motion-nca-sweep/metrics.json are not currently present on remote
main, so do not compare raw MSE values across these two benchmarks. This PR
does not edit any prior experiment, recording, or checkpoint.

## Hypothesis

Training a local predictive RGB NCA on many moving-object trajectories,
rather than repeatedly presenting one trajectory, improves generalization
to unseen motions, ideally above both persistence and observed-image
constant-velocity extrapolation at +10 frames.

## Experimental design

- Same RGB NCA (src/predictive_rgb_nca.py) and prediction-correction
  dynamics, parameter shapes, initialization, loss and optimizer in both arms.
- Single: repeat one fixed training trajectory for all updates.
- Distribution: train on independent color, location, size, direction,
  velocity and scenario samples. Scenes include straight motion, wall bounces
  and a stationary occlusion stripe; simulated hidden object position is
  never given to the NCA or its baselines.
- Train on trajectories with seeds 31000+, validate separately on 41000+,
  test separately on 51000+. These splits do not share exact trajectories.
  Three independent model initialization seeds run on exactly the same
  train/validation/test scenes and paired update schedules.
- Keep persistence and image-derived centroid constant-velocity baseline
  (infer velocity from last two observed RGB frames, shift foreground).
  This is not simulator-oracle velocity, optical flow, or pose tracking.
- Validation MSE selects checkpoint, including step 0. Report step-0 picks
  conspicuously; only untouched test scenes determine results.
- Report +1/+3/+5/+10-frame full RGB MSE, changed-region MSE,
  changed pixel counts, +10 seed wins against both controls, and separate
  single/distribution checkpoints.

A shifted foreground can fail on occlusion/edge cases; that is an informative
limitation of this control, not evidence that NCA tracks objects. Future
truth is always scoring/training labels, never an inference input.

## CPU pilot

After reviewing and merging the PR:

    python -m experiments.rgb_motion_distribution --steps 30 --batch-size 2 --horizon 10 --context 8 --out runs/rgb-motion-distribution-smoke

For a longer three-seed run:

    python -m experiments.rgb_motion_distribution --steps 100 --batch-size 3 --train-trajectories 96 --validation-trajectories 18 --test-trajectories 24 --context 8 --horizon 10 --ticks 2 --out runs/rgb-motion-distribution

Settings are designed for CPU but total runtime and learning success are
not established yet. Increasing training compute is not proof of learning.

## Promotion gates

1. Distribution beats same-budget single trajectory on unseen test
   trajectories in all or most seeds.
2. Distribution beats persistence at +10 frames, ideally constant velocity.
3. Report the strong baseline and changed-region performance, not only
   whole-frame MSE dominated by static backgrounds.
4. Replicate on a new frozen test set before RGB webcam transfer.

This step adds no new NCA architecture, learned reference selector, phase
oscillator, or claims about biological equivalence.
