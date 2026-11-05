# Camera characteristics and capture robustness

```bash
python -m experiments.camera_robustness
python -m unittest tests.test_camera_robustness -v
# Optional device-only inspection (OpenCV required):
python -m experiments.camera_robustness --real-camera --camera-index 0 --lock
```

The CPU pipeline injects isolated/burst missing frames (1–10%),
illumination gain, white balance and approximate row-time skew.
Each perturbation uses **past observations only** in the persistence
baseline. Reports stratify sensor-frame error by exposure transitions
and missing observations; these are distinct from physical motion errors.

Real-camera capture is optional/lazy. It attempts exposure and
white-balance locking **where the driver supports it**, records
monotonic per-capture timestamps, exposure, gain and WB readings,
and leaves unavailable values as null. OpenCV `set()` success
does not prove settings truly stayed fixed; inspect raw variation
on the chosen camera.

Use measured target minus forecast-commit and display latency for
effective future lead. Nominal frame counts alone do not certify a
one-second live prediction. Synthetic 100-ms timestamps are **not**
substitutes for measured acquisition timestamps. Rolling-shutter
skew depends on hardware readout; the simulation is a stress test,
not a calibrated device model.

Do not infer objects from metadata fields or leak future frames.
