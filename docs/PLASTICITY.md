# PR 8 — Local plasticity and homeostasis, deliberately separate

This experiment is a tiny recurrent ring of local cells with two neighbor connection weights per cell. Compare fixed connections, unconstrained Hebbian correlation updates, and Hebbian updates with homeostatic decay under the same traveling sensory stimulus and reversed sequence.

```bash
python -m experiments.plasticity --steps 150 --out runs/plasticity.json
```

Reference at 120 steps: fixed final maximum weight = 0.15, Hebbian = 0.362, and homeostatic = 0.0415 (mean over two test seeds). These are numbers **from this toy**, not biological measurements, and do not establish improved memory, object persistence, functional specialization or conscious perception. Homeostatic decay can also drive useful weights toward zero; a lower norm alone is not success. This experiment can guide the later controlled addition of online adaptation to a trained perceptual network.

No hidden weights are updated in the predictive mirror by this PR, and no cross-repository code sharing or new framework is introduced.
