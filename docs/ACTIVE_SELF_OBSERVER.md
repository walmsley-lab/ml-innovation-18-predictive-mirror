# Active perceptual self-monitoring

**Question:** Can a perceptual agent use *its own* evolving memory and
prediction-error signals to decide what to observe next, and can interventions
on that memory demonstrably change behavior?

This adds a small foveated observer to the existing regulated-plasticity
experiment. It is an operational model of belief, uncertainty, and attention;
**it does not create, detect, or establish subjective qualia or sentience**.

## Demonstration

One continuous 2D world undergoes familiar motion, a surprising reversal,
a temporary hidden-object interval, and a return to familiar motion. The
moving red object and static landmark come from the existing toy world.

At each time step the observer:

1. Predicts an object position from remembered observations.
2. **Chooses one limited circular glimpse before seeing the new frame.**
3. Updates a simple handcrafted red-object tracker and its confidence proxy.
4. Scores its already-issued cellular forecast and regulates local plasticity.
5. Records its belief, innovation, chosen gaze, detection, and learning mode.

World truth is used **only for evaluation**. There is no future-frame input,
privileged simulator-state read, learned object detector, or externally
provided target position.

Four matched one-glimpse policies compare active predictive gaze, fixed
central gaze, scheduled raster, and predictive gaze with the tracking memory
reset at each frame. The retinal/cellular learner is retained in all arms;
the lesion isolates the **explicit tracker**, not every kind of cellular memory.

## Run

```bash
python -m unittest discover -s tests -p 'test_active_self_observer.py' -v
python -m experiments.active_self_observer \
  --seeds 11 23 41 --size 32 --segment-frames 40 --radius 6 \
  --out runs/active-self-observer
```

Read `metrics.json` and per-seed/policy `*.jsonl` traces. Each line keeps
`observer` information separate from `evaluation` truth. The report
contains detection fraction, pre-observation tracking error, error around
the changed motion regime, and regulator occupancy. No external GPU or
paid Akash campaign is required.

## Interpretation and next steps

Inspect **causal function**, not evocative internal-state narratives:

- Does a limited observer recover from occlusion more reliably with memory?
- Does active gaze improve detection and tracking over fixed and raster views?
- Do interventions on memory alter *actions* and objective prediction errors?
- Does regulator activation actually change the cellular forecast or merely
  change reported internal statistics?

The tracker and confidence score are heuristics. The cell network does not
yet own the motion representation or choose its own observation policy; this
is a measurable *scaffold* for later learned sensorimotor representations.
An active policy can easily perform worse than a fixed camera on this small
world, and that would be informative. Do not treat self-reports as evidence
of self-awareness, consciousness, or phenomenology.

Promotion would require an end-to-end learned representation and active
policy, calibrated uncertainties, held-out worlds, and meaningful
outperformance of simple classical filtering/attention baselines.
