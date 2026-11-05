# Layered advection and visibility truth

```bash
python -m experiments.layered_advection_benchmark
python -m unittest tests.test_layered_advection_world -v
```

This independent 2D world supplies *continuous-textured* foreground
layers, a textured background, front/back depth order, fractional-pixel
motion and exposure-only change. It preserves both source-to-target
flow (dy,dx) and **two different visibility masks**:
whether source pixels survive at t+1 and whether target pixels have a
corresponding visible pixel at t. Newly disoccluded target content is
not recoverable by transporting only the latest RGB frame.

All identity, flow, depth, and visibility fields are explicitly
**evaluator-only**. The predictor receives `model_inputs(record,context)`
which returns past RGB alone. In a separately labeled oracle substitution,
privileged flow/visibility may be passed to assess rendering ceilings.

Flow is expressed in **pixels per frame**. Visibility is tested via
nearest-pixel backprojection for small fractional motions; this is a
raster contract, not perfect subpixel geometry. Expect some aliasing
near boundaries. The generator does not represent real rolling shutter
or an industry-scale photorealistic video distribution.

Roadmap 57 compares direct RGB, history transport and residual synthesis
using this truth. Do not score a "future-occlusion" field as an online
attention feature.
