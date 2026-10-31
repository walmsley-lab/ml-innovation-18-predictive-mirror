# Side-by-side forecast comparisons (standalone viewer)

All three separately trained long-horizon models and last-frame persistence see
**exactly the same observed input frames**. This viewer does not alter their
weights or the existing training/architecture implementations. It shows a
five-column grid: future truth (replay only), persistence, plain recurrent,
phase oscillator without neighbor coupling, and coupled phase oscillator.
Each row is a future horizon, normally +1/+3/+5/+10 frames.

## Replay first (no camera or IR illumination needed)

```bash
python -m experiments.compare_horizons --source replay \
  --input recordings/webcam.npz \
  --parameters-dir runs/horizon-phase --size 32 --fps 10
```

The default replay uses **only the final 20%** of the recording, including its
observed context, and displays ground truth only **after all forecasts are
computed**. Future truth is used strictly for scoring and visualization.
A single recording's holdout is not an independent experiment. Replay latency
includes device compilation during the first window.

## Live synchronized webcam (HP device 2 on the tested laptop)

```bash
python -m experiments.compare_horizons --source live --camera 2 \
  --parameters-dir runs/horizon-phase --size 32 --fps 10
```

Each arm's +1 through +10 forecast derives from the **same camera history**,
computed before the next application-level camera read. Future columns are
never mistaken for an actually observed future. When new frames arrive, prior
forecasts are scored at the correct lead, so metrics appear with a delay.
Actual sensor capture/exposure timing and driver buffers remain unverified.
One complete panel pass runs the 3 models; on slow CPUs it can exceed 100 ms.
The code reports aggregate median model inference wall time rather than
claiming a fixed realtime frame rate.

You can save a headless evaluation result without opening a camera or UI:

```bash
python -m experiments.compare_horizons --source replay \
  --parameters-dir runs/horizon-phase --no-display \
  --max-frames 20 --out runs/horizon-comparison.json
```

Both modes report total-frame and changed-pixel MSE for every model and
future horizon in `runs/horizon-comparison.json`. Changing-pixel masks are
oracle evaluation masks, not inference inputs. Pixelwise persistence is the
baseline; an optical-flow motion predictor is a separate future control.
No trained models or local camera recordings are committed by the PR.

## Important: oscillators do not control the laptop's camera LEDs

The current oscillator is a **numerical phase angle per 32x32 pixel**, with
a uniform natural frequency and optional local Kuramoto-like coupling.
Phases start at zero across all pixels. Without meaningful phase differences,
neighbor coupling has little effect. This is not a traveling wave, a hierarchy
of independently entrained rhythms, an articulated object oscillator, or a
hardware strobe controller.

The red lights flanking some HP webcams are *hardware infrared emitters*
(consult the exact laptop guide to confirm). Camera operation can activate
hardware LEDs regardless of the model architecture. Replay mode avoids
opening the camera altogether.

**Separate next experiment:** Compare heterogeneous oscillator frequencies,
learned or observation-driven entrainment, propagation/delays, and traveling
waves to K=0 and plain recurrent controls with identical video and metrics.
No benefits are asserted without direct measurement.
