# PR 7 — Tiny active-sensing sandbox

An analytical 1D gaze world precedes coupling gaze to the trained visual predictor. The hidden object position is never passed to policy functions; the observer retains a discrete posterior from its observed crop, then selects among left/stay/right sensing moves.

```bash
python -m experiments.gaze --budget 5 --out runs/gaze.json
```

Compare fixed gaze, seeded random gaze, globally optimized-fixed actions (enumeration over admissible sequences) and deterministic belief-guided greedy actions. Every policy receives exactly the same observation count and admissible movement range. This is a **known-model Bayesian control**, not learned active inference, neural cellular plasticity or a biological gaze circuit.

CPU reference at budget 5 over every possible object x: optimized-fixed mean absolute error 0.2px vs greedy 0.6px, random ~2.40px and fixed ~4.36px. This is a deterministic toy simulation, not evidence of learned perceptual improvement. Indeed the greedy policy loses to optimized-fixed here. Next extensions would replace the analytic belief with learned sensory states and add costs, dynamic objects and blinded holdouts. No RL needed.
