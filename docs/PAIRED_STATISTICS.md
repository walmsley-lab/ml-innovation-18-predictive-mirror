# Paired-history effects and planned precision

The unit for the occlusion task is a **scene seed pair**: two opposing histories from one generated world. The pair members must not count as two independent examples. Pixel values and adjacent forecast frames are correlated; never use them as independent observations for significance.

`src.paired_statistics.paired_effect` expects rows for model-initialization seeds and columns for shared independent scene seed pairs, and computes **candidate minus baseline** for higher-is-better scores. The baseline may be a fixed 1D control shared across model seeds. It uses two-axis cluster bootstrap across seed rows and scene-pair columns, preserving paired comparison covariance.

Example input for `python -m experiments.paired_effect_report --input runs/my-scores.json`:

```json
{
  "candidate": "cellular",
  "baseline": "rgb_history_kalman",
  "metric": "on_time_reappearance_success_rate",
  "model_seeds": [11, 23, 41],
  "scene_pair_ids": [71000, 71001, 71002],
  "candidate_scores": [[0.5, 1, 0], [1, 0, 0.5], [0, 1, 0.5]],
  "baseline_scores": [1, 1, 0.5]
}
```

The example is **illustrative fake data**, not a research result. Do not use it in figures.

Report the number of independent scene pairs, number of model seeds, model-seed effects, absolute effect size, 95% cluster interval, and the preregistered minimum practical improvement. A no-supported-advantage label does not prove architectures equivalent or inferior in every setting.

`pilot_precision` resamples a pilot score distribution at proposed counts of seeds/pairs to estimate plausible interval widths. **It is not a formal power analysis**, not evidence of external validity, and cannot select a margin after final-test inspection. Use pilot variability and the desired accuracy–cost decision to choose a final run budget *before* evaluating the held-out test.

This module is an independent analysis tool until the frozen campaign is completed. Its outputs cannot certify provenance or verify an empirical finding by themselves.
