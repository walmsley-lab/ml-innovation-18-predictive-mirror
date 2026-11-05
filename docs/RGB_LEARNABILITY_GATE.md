# Learnability before complexity

```bash
python -m experiments.rgb_learnability_probe --output blend --steps 100 \
  --context 8 --horizon 5 --out runs/rgb-learnability
```

This deliberately **trains and scores on the same one trajectory** to answer a narrow engineering question: can the model move away from its initialization and overfit simple moving RGB content? A failed overfit gate is actionable evidence of a training, optimization, readout or architecture problem. Passing it is not object permanence, generalization, or benchmark success.

Compare the existing residual-plus-clipping head (`--output residual`) against the existing non-saturating blend variant (`--output blend`) without deleting either model. Track per-milestone MSE, frame variance and clip-rail frequency. Save the selected checkpoint and literal settings; selection is **in-sample** by design.

**Promotion gate:** A repeatable reduction relative to the step-zero model and persistence, plus visible moving-object forecasts on the fixed clip. Only then increase to held-out scene diversity and longer horizons. Report any shortfall rather than claiming a successful reproduction.

This PR introduces no novel model, no multi-seed generalization statistics and no expensive training during CI.
