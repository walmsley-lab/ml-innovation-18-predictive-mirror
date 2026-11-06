# Embodied causal curriculum: arithmetic, resonance, and blind touch

**Research question:** Can one compact recurrent model learn the *shared
physical causes* connecting actions, images, body position, contact pressure,
and vibration, and use earlier contacts to predict unexplored geometry?

Inspired by embodied perception and the notion of animation as a body moving
through a world. This is **not** evidence or a simulation of subjective
experience, sentience, qualia, or an immortal soul.

## The executable curriculum

**Stage 1 — coupled mechanics.** A numerical damped spring and an actuator
generate synchronized low-resolution RGB, two motor-proprioceptive channels,
contact force at a mechanical stop, and an audible/vibratory signal. The model
predicts what happens *after* applying a command, not just what it last saw.

**Stage 2 — touch as spatial knowledge.** A simulated fingertip moves through
a bounded 2D field around a randomly shaped fivefold disc. Contact measures
indentation and vibration correlates with changes in pressure. A portion of
training episodes are **eyes-closed**, with no visual input. In every episode
the learner receives fingertip position, the previously observed touch, and
the action it is about to issue, then predicts its next sensations. An
auxiliary contour head is trained to reconstruct the silhouette. Only
ground-truth supervised losses and evaluators access the full shape.

**Stage 3 — compositional transfer.** The motor command is a small arithmetic
AST: `Z`, `S(x)`, `NEG(x)`, `ADD(x,y)`. A differentiable recursive
interpreter learns numeric consequences from a training label and sensor
predictions. Longer *unseen compositions* are reserved for testing.
This is a deliberately tiny compositional action grammar, **not** natural
language. The simulator's true interpreted value is never a model input.

Train two equal-capacity arms: properly aligned channels and a negative
control whose tactile/vibration inputs are temporally shuffled within each
training episode. Both use identical initial weights, update counts and
sampled minibatch IDs.

## Run locally

```bash
python -m unittest discover -s tests -p 'test_embodied_causal_curriculum.py' -v

# Quick CPU trial (validates training, not efficacy)
python -m experiments.embodied_causal_curriculum \
  --seeds 11 --spring-steps 10 --touch-steps 20 \
  --train-count 12 --val-count 4 --test-count 4 \
  --out runs/embodied-causal-smoke

# Replicated introductory training
python -m experiments.embodied_causal_curriculum \
  --seeds 11 23 41 --spring-steps 40 --touch-steps 80 \
  --out runs/embodied-causal
```

All generated episodes are synthetic. A GPU is optional; this does not
require Akash. Evidence and immutable parameter checkpoints are saved
locally at `runs/embodied-causal*/metrics.json` and per-seed NPZ files.

## What the experiment actually measures

The run uses separate train (31k/32k), validation (61k/62k), and test
(91k-96k) episode-seed ranges. Tests include familiar spring motion, new
stiffness, unseen arithmetic compositions, sighted palpation, and blind
palpation. For touch, it reports:

- **Next-contact prediction** on the upcoming motor action, even with no
  visual input.
- **Unprobed contour MSE** after a full tactile history: compare shape
  reconstructions at pixels not visited by the fingertip with a train-set
  average silhouette prior.
- **Sensor ablations**: remove touch, proprioception, vibration, or
  temporally align/misalign touch history; test whether each cause matters.
- **Command generalization**: learned AST interpretation error on new
  compositions and the sensor forecasts those commands cause.

Report paired-seed performance relative to the shuffled control and prior;
do not promote a mechanism for merely producing plausible contour images.
An average-shape prior may be difficult to beat, and the low-capacity model
may ignore touch entirely. That outcome is scientific evidence too.

## Boundaries and future research

These are hand-written dynamics, a handcrafted fivefold contour family,
supervised shape labels, and a known command grammar. They are training
**scaffolds**, not a proof that spatial abstraction, body schema, symbolic
language, or consciousness arose spontaneously. Grammar supervision gives
the model the numeric *meaning* of actions, not access to the simulator's
private dynamics. Memory resides in a small recurrent vector; the cellular
sheet from Predictive Mirror is not yet integrated.

If this reproducible regime succeeds, the next controlled experiment is
to replace the recurrent core with the local cellular update rule, compare
parameter/FLOP budgets and retention, then progress to active policies
that select where to touch under matched contact budgets. No external
body or self-description is implied by these results.
