# NCA growth reference exercise

Reference: Mordvintsev, Randazzo, Niklasson and Levin (2020), [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/). We want to **match published behavior**, not claim a new cellular architecture.

```bash
python -m experiments.reference_growth_exercise --steps 200 --ticks 16
python -m unittest tests.test_reference_growth_nca -v
```

This first PR introduces a small self-contained **reference-inspired** JAX growth rule trained on a generated ring, plus a damaged-state continuation control. It records initial/final training MSE and damage recovery. It does **not** use the original Distill sprite, asynchronous fire masks, alive-mask mechanics, batch-pool damage curriculum or paper-equivalent scoring. Therefore it is **not yet a faithful reproduction** and cannot be reported as matching Mordvintsev et al.

**Next reference-parity exercise:** pin upstream commit/license, reproduce the original target and training schedule without changing its meaning, measure match to published plots/recovery behavior, then port only the smallest reusable pieces to JAX. Keep the upstream protocol and our extension separately documented. This experimental runner is useful as a cheap trainability control for local growth dynamics and can be expanded without touching the RGB video predictor.
