# Transport, visibility and synthesis—matched input comparison

```bash
python -m experiments.layered_output_comparison --seeds 11 23 41
python -m unittest tests.test_layered_output_heads -v
```

Three independently trained heads share the same source frames
(`t-2,t-1,t`), initial parameters and training budgets:

- **Direct RGB:** residual prediction from the observed latest frame.
- **Transport:** move color from the two most recent source views using
  estimated flow; learned mixing gives the short-history content path.
- **Transport + synthesis:** same transport plus a learned bounded residual
  whose inputs are **past frames**, not future target pixels.

The layered-world evaluator supplies source/target visibility, object IDs
and known flow, **never exposed to model rollout or gradient training**.
Its four held-out interventions swap oracle flow and/or oracle target
source-visibility. The 2×2 helps diagnose coupled errors; it does not
assume flow, visibility and synthesis have independent additive effects.

All methods train with the same RGB reconstruction/prediction objective,
but active parameter use and FLOPs differ. Report per-seed rollout MSE,
motion-region MSE, sharpness, visual holes/ghosting and action timestamp
correctness before claiming usefulness. Additional photometric controls
and longer-horizon error galleries remain mandatory for promotion.
