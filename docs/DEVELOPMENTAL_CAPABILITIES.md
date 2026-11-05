# Capability-first developmental perception

This charter maps biological/developmental motivations to **engineering tests**,
controls, failure categories, falsifying outcomes and a smallest worthwhile
effect. It does not claim equivalence between an NCA and infant or animal
vision. Species differences and uncertainty are recorded per capability.

```bash
python -m experiments.developmental_capabilities
python -m unittest tests.test_developmental_capabilities -v
```

The machine-readable source is
[`experiments/developmental_capabilities.json`](../experiments/developmental_capabilities.json).
Thresholds are **proposed exploratory gates**. Calibrate feasibility on
development episodes; freeze task conditions, improvement threshold, units,
splits, number of seeds and evaluation count **before test evaluation**.
Use paired episode/session resampling and report uncertainty from training
seeds separately. Numerical acceptance must not be retrofitted to a positive
result; revise the protocol only on a new untouched test set.

A causal forecast has a commit timestamp: no future RGB, visibility,
flow truth or next action may enter the predictor. Report absolute
error alongside skill relative to persistence; avoid near-zero
denominators and positive-looking constant-image oracles.
