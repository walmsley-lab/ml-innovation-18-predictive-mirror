# PR 4 — Matched-history control for recurrence (H2)

The feedforward convolution gets up to 12 previous observed frames and matching action commands, the same sequence length as the recurrent baseline. It has no persistent state between windows. Default 3 output channels gives a similar (not identical) number of trainable parameters; the test enforces <15% difference. Compute cost and hidden-state size are **not matched**, so results must report those separately; this is a baseline, not final causal evidence.

```bash
python -m train --architecture recurrent --steps 120 --batch-size 8 --output runs/recurrent
python -m train --architecture feedforward --steps 120 --batch-size 8 --output runs/feedforward
python -m evaluate --architecture recurrent --parameters runs/recurrent/parameters.npz
python -m evaluate --architecture feedforward --parameters runs/feedforward/parameters.npz
```

The evaluator now adds reappearance location MAE in pixels on episodes in which a visible dot disappeared and later reappeared; a no-detection forecast receives a fixed 16-pixel penalty. The threshold is fixed at 0.5, not fitted on the test set. Only post-prediction **visible-object** masks from simulator truth are used. More formal occlusion localization and timing metrics and multiple-seed uncertainty are future experiments.

Checkpoints now reject architecture, curriculum, replay fraction, and staged/shuffled total-update horizon mismatches on resume. This prevents changing the generated data schedule halfway through a run.
