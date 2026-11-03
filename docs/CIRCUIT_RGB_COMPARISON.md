# Hardware-constrained cellular rule: RGB forecasting scout

**Question:** Does an idealized, local, two-state primitive contribute
useful visual prediction compared with removing coupling, nonlinear feedback,
slow memory, or quantizing to 4 bits?

```bash
python -m unittest tests.test_circuit_rgb_comparison -v
python -m experiments.circuit_rgb_comparison --seed 11 --size 32
# Optional frozen model comparison on the same held-out scenes:
python -m experiments.circuit_rgb_comparison --checkpoint runs/rgb-intro/seed-11-curriculum.npz
```

- Portable circuit rule follows NCA [minimal circuit PR](https://github.com/walmsley-lab/ml-innovation-19-neuro-cellular-autonoma/pull/2);
  copied as a *small standalone NumPy function*, not a runtime dependency.
  One two-state circuit is instantiated for each RGB channel/pixel.
- All arms use the same exact world generator and context and independently
  fit only four linear readout coefficients (fast state, slow state, current
  pixel, intercept), against one-step training targets. Autoregressive rollouts
  never use future frames; score +1/+3/+5/+10 on disjoint synthetic test scenes.
- Fixed coefficients and ridge regularization, no chosen validation checkpoint;
  held-out validation is separately reported. Persistence and constant-velocity
  baselines are mandatory. An **optional** frozen RGB NCA checkpoint adds a
  model comparison if supplied; this is not a compute/parameter-matched study.
- Full/no_coupling/no_feedback/no_slow_memory/quantized_4bit ablations isolate
  questions but do not fully match active state dimensions (disabled memories
  leave a dormant channel). Report this when interpreting the result.

**Evidence boundaries:** The ports use periodic edges, a perfect pixel-local
drive, ideal simultaneous updates, and a digital *per-pixel* head. None
represents a realistic wired graphene sheet, ADC/DAC implementation, measured
device response, or energy advantage. Four linear head coefficients are shared,
but two analog state readings per RGB channel/pixel is a broad physical
observation boundary, not a demonstrated narrow-terminal readout. No claim
of hardware efficiency until the Physical Computing feasibility gate.

**Next:** Try restricting observation to a small number of row/column terminal
values; compare to delay lines and pixel-history models at equal readout/state
budgets, using a device-grounded primitive contract. Do not fold this into the
existing oscillatory/harmonic experiments.
