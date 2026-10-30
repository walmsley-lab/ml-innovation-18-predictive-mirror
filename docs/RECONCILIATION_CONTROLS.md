# PR 20 — Accuracy-qualified reconciliation controls

Use *trained*, frozen parameters from matched-seed, matched-budget runs:

```bash
python -m experiments.reconciliation_controls \
  --multisheet runs/multisheet/parameters.npz \
  --single runs/recurrent/parameters.npz \
  --no-feedback runs/no-feedback/parameters.npz \
  --out runs/reconciliation-controls.json
```

All decoders are fitted **after** perceptual model training on separate probe seeds, scored on untouched test seeds, with **true barrier occlusion** from simulator truth only during evaluation. Report raw compatibility **and** compatibility when *both* sheet decoders are accurate within 1 pixel. No occlusion events yields `null`, not a success.

Controls: intact two-sheet, cut after training, calibration-mean message substitution, 216 randomly selected within-sheet connections severed (matching the number of cross-sheet kernel weights), a split-decoder null on six/two channels of a single recurrent sheet, and independently trained feedback-free control when supplied. Random lesion's cut-count is matched but its magnitude/compute and effective active capacity are **not** matched.

**Limits:** This is a diagnostic across successive video time steps, not yet a within-frame convergence curve across multiple recurrent inference iterations. It does not establish anatomical specialization, conscious perception, or mechanistic causation. Reserve strong causal claims until a matched-compute and same-magnitude lesion study with independent seeds.
