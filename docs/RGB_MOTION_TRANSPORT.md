# Motion transport with cellular-style residual correction

A small, optional RGB forecaster learns one of nine local spatial offsets per
pixel using a softmax mixture. A separate learned residual can correct the
transported image. Compare residual-only, transport-only and their combination
under identical train/validation/test RGB scenes and update budgets.

```bash
python -m experiments.rgb_motion_transport --steps 80 --seeds 11 23 41
python -m unittest tests.test_rgb_motion_transport -v
```

This is a **candidate mechanism**, not a replacement for the base NCA and
not yet a recurrent hidden-state model. Its residual branch is feedforward
and its transport samples are bounded to 0 and +/-2 pixel offsets;
conclusions cannot be generalized to learned optical flow, long-range
transport or 3D movement. All variants keep identical parameter storage,
but inactive branches and active computation are not fully matched.
Compare against persistence/velocity and event-specific metrics before
deciding whether transport should become part of the recurrent NCA.
