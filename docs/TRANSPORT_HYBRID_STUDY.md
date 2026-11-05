# Transport versus appearance synthesis

The repository already has `src.rgb_motion_transport`: it learns a spatial offset selection and an optional RGB correction. This PR **reuses** that idea instead of creating a second renderer or world simulator.

The new `rgb_transport_blend` arm is a bounded convex combination of a transported observation and a synthesized color estimate. Two separately trained arms, `transport` and `bounded_hybrid`, have matched initialization and optimization budgets; only synthesis differs.

```bash
python -m unittest tests.test_rgb_transport_blend -v
python -m experiments.transport_hybrid_study --steps 100
```

The output is a lightweight **validation training comparison**: the point is to test whether synthesizing missing appearance helps beyond relocating observed pixels, without adding clip-rail artifacts. This is not a validated occlusion result or a performance win. Inspect moving-object galleries and compare to a strong image-derived velocity baseline before selecting a winner.

Next: evaluate the trained arms with the frozen paired-history scorer and cost accounting. Require comparable input privileges and disclose whether flow or other state was externally supplied.
