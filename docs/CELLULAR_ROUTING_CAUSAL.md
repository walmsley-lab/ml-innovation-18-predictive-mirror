# Frozen causal routing probes

This follow-up to soft cellular references does **not** introduce another
model family. It tests a saved checkpoint under learned routing, uniform
routing, spatially shuffled routing and a key-parameter knockout. Examine
forecast sign accuracy *and* the fraction of receiving-cell weight landing
on the true source.

```bash
python -m experiments.cellular_references --steps 60 --seeds 11
python -m experiments.cellular_routing_probe \
  --checkpoint runs/cellular-references/seed-11-learned.npz
python -m unittest tests.test_cellular_routing_probe -v
```

Source locations are computed from evaluator-only pulse labels. The learned
network already knows the supplied destination mask; therefore **routing
to the source is not autonomous object discovery**. This probe complements,
but cannot replace, matched retraining ablations and hidden-state interventions.
