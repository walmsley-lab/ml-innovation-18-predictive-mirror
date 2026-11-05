# Comparable JAX compute profile

**Objective:** report *compilation*, post-compile training, and warmed
inference separately for each architecture, at **identical batch size,
image dimensions, context length, forecast horizon, cellular ticks and
number of training updates**. Track parameter count, repetitions,
per-step variance and the hardware/JAX backend.

```bash
# Tiny smoke:
python -m experiments.comparable_compute --models cnn \
  --seeds 11 --train-steps 1 --warmup 1 --repetitions 2
# Full comparative probe (runs real JAX kernels but does NOT
# establish whether the trained models predict useful images):
python -m experiments.comparable_compute \
  --models cnn transport hierarchical nca nca_blend routing \
  --seeds 11 23 41 --train-steps 20 --warmup 3 --repetitions 20
python -m unittest tests.test_comparable_compute -v
```

Timing contract:

- `compile_ms.training_step` measures a separate `jax.jit(...).lower(...).compile()`.
- `training_steady_step_ms` is measured after compilation, with JAX outputs
  blocked each step; `training_steady_total_ms` sums measured updates.
- `compile_ms.inference_rollout` measures the compilation of one
  **full context-to-horizon rollout**.
- `inference_rollout_ms` includes `warmup` untimed executions followed by
  `repetitions` separately synchronized full-rollout executions. It
  reports mean, median, standard deviation, 5th and 95th percentiles,
  minima and maxima.
- Runtime metadata records Python/OS/CPU count, JAX/jaxlib/NumPy versions,
  accelerator backend and model-visible device type.

The model family, context/horizon, batch size and device choice materially
change the costs; `equal SGD updates` **does not** mean equal FLOPs, accuracy,
training examples or parameter count. JAX compilation cache effects and
competing host load must be controlled across repeated sessions. The runner
profiles on the reproducible linear world and saves model-specific parameter
counts. It does **not** re-label historical timing numbers as if they came
from this protocol or promise hardware-independent performance.
