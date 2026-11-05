# Classical tracking, coasting and causal transport controls

```bash
python -m experiments.classical_transport_controls --seeds 11 23 41
python -m unittest tests.test_classical_transport -v
```

These controls use RGB history only; no future RGB, positions, IDs or
evaluator visibility. Past-frame translation is estimated by image
matching, and a **backward gather** is distinct from a **forward splat**.
The latter returns occupancy weights, making holes/duplication measurable.

Two deliberately simple object filters use a known synthetic color
palette: **coast** from local consecutive detections and **appearance
re-identification** from the last two valid color observations, even
through a gap. This is a strong *simple* baseline for controlled toy
scenes but not a benchmark for unconstrained natural-video re-ID.
Report misses separately from localization distance.

The runner reports per-frame MSE and joint failure counts with the
coast baseline, so we can distinguish reduced severity from new
failure modes. Initial thresholds are protocol constants and must be
frozen against development data before confirmation. Do not imply
that transport supplies content that was never visible in history.
