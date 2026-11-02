# PR 48 — Independent particle neural cellular automaton

A small **particle-state** cellular predictor, independent of pixel RGB
NCAs and of the provided-skeleton graph NCA.

Every moving particle is a cell with 2D position and recurrent hidden
channels. Each receives distance-weighted messages from other particles
and applies the same learned update rule. A paired control disables those
messages, retaining trainable parameters and training seeds.

**Important limitation:** The synthetic environment SUPPLIES tracked 2D
particle coordinates and stable identities. We do not extract particles
from camera pixels, learn object identities, or predict full RGB images.

The model gets eight prior position observations and predicts ten future
positions open-loop. Both model arms receive identical examples, steps,
and initialization seeds. The two observation-only baselines are last
position (persistence) and two-frame constant-velocity extrapolation.
Report per-lead (+1/+3/+5/+10) coordinate errors (in world pixels),
and save each arm's parameters. Future coordinates are targets/evaluation
truth, never model inference inputs.

    python -m experiments.particle_nca_motion \
      --steps 100 --seeds 11 23 37 --context 8 --horizon 10 \
      --preview --out runs/particle-nca-motion

    python -m unittest tests.test_particle_nca_motion -v

Optional Pillow creates a position-only GIF comparing observed anchor,
predicted positions, velocity baseline, and actual future positions.
Do not confuse the visual rendering of given particle states with RGB
perception. This adds no changes to the existing webcam, selector,
oscillator, or graph-skeleton models.
