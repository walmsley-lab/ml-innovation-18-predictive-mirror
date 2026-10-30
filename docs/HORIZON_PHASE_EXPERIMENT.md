# Separate experimental lineage: long-horizon forecasts with phase oscillators

The original webcam predictor (one-step), the Phase Ablation (#30), and this
multi-horizon experiment are **three distinct implementations/runs**. No prior
checkpoints, results or old training scripts are overwritten or modified.

## Hypothesis

Training directly through autoregressive 1..H-frame rollouts might help forecast
longer than one frame. Coupled phase oscillators might retain useful temporal
structure over those missing observations. **Neither hypothesis is assumed true.**

This experiment reuses the core recurrent modules, but has an independent
trainer, evaluator, checkpoints and live display. Three separate arms share
initial weights, 100-step update budget, observed video minibatches, temporal
holdout and learning objective:
- `plain`: original persistence-residual recurrence.
- `phase_uncoupled`: same trainable weights, an extra phase channel but no neighbor coupling.
- `phase_coupled`: same extra phase state plus local oscillator coupling.

The phase arms have extra **nontrainable state/operations** and so the comparison
is compute-matched by parameter count and steps, not runtime. Nothing here is
a full 3D model, discovered object identity or human visual cortex.

## Training and validation

```bash
python -m experiments.horizon_phase --input recordings/webcam.npz \
  --steps 100 --context 8 --horizon 10 --out runs/horizon-phase
cat runs/horizon-phase/metrics.json
```

Long-horizon loss only receives eight observed context frames and trains
against the next ten frames. Each new prediction is fed back to forecast the
next step: **future ground-truth frames are never fed into the rollout**.

Hold out the final 20% of the same recording, with a 10% chronological embargo
and nonoverlapping windows; still **not independent validation**. Report errors
at +1,+3,+5,+10 frames versus last-frame persistence and pixel-difference
extrapolation, including oracle motion-region errors and counts.
The number of real milliseconds per frame is uncertain if camera timestamps
are irregular; metrics contain measured median capture interval if available.

## Optional live demo (requires opencv-python)

```bash
python -m experiments.horizon_phase_live --arm phase_coupled \
  --parameters runs/horizon-phase/phase_coupled.npz --size 32 --fps 10
```

Displays current observation, persistence and independent predictions at several
future horizons. Labels in seconds assume the requested FPS, not confirmed
sensor exposure intervals. Forecasts are generated before the next application
camera read; driver buffering has not been independently characterized.

## Future performance gates, in separate follow-up experiments

1. Independent second-recording validation and repeated seed confidence intervals.
2. Beat persistence **and** meaningful motion baselines on changing regions at each horizon.
3. Stress test occlusion, direction reversals, camera motion and out-of-distribution poses.
4. Record latency, actual capture intervals and model footprint on laptop CPU.
5. Explore explicit depth, learned object correspondences and active sensing as
   *distinct* models before preregistering a reconciliation architecture.

Do not claim oscillators confer multi-second object permanence without those gates.
