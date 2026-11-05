# Make forecasting learn, one horizon at a time

```bash
python -m experiments.rgb_staged_learning --seed 11 \
  --horizons 1 3 10 --steps-per-stage 100
python -m unittest tests.test_rgb_staged_learning -v
```

Use the existing `rgb_grounded_nca` **blend** head instead of replacing the model core. Keep optimizer moments and weights across increasing prediction horizons, evaluating each stage on disjoint validation scenes and saving a checkpoint.

This experiment asks whether a curriculum improves training progression over starting immediately at +10, **not** whether the resulting model generalizes or beats a classical baseline. The script does not touch the final-test split. Run an independently trained direct-10 reference with the same total update budget before claiming curriculum benefit.

Inspect forecast galleries and object localization at each stage. If even the +1 model cannot localize moving blobs after adequate training, fix basic learning and output dynamics before adding recurrence, harmonics, or routing.

Milestone data are lightweight: validation RGB MSE and clip-rail fraction plus exact checkpoint. Richer metric suites belong at the frozen comparison, not every gradient step.
