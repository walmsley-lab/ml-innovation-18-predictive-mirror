# Functional memory and intervention sensitivity

The repository already contains hidden-state intervention and harmonic probes. This PR provides a short, common, **frozen checkpoint** test that does not silently call a dead model a scientific null.

```bash
python -m experiments.functional_state_campaign \
  --checkpoint runs/rgb-nca-1s/parameters.npz --episodes 6
# For a trained non-saturating head, explicitly select --output blend
```

For the *same observed RGB contexts* and parameters, compare intact, reset-hidden, spatially shifted hidden and image-shock arms. Future frames enter scoring only. The companion `forecast_diagnostics.collapse_report` labels each horizon as `live`, `dead`, or `insensitive` based on known controls.

**Positive-control warning:** an image shock is deliberately extreme and may fail to move a saturated recurrent model, while full memory reset can be ineffectual for a truly memoryless predictor. This means the measurement is uninformative, not that memory or communication is unnecessary. On a responsive model, evaluate both prediction changes and *task-performance changes*. A forecast change alone establishes causal influence on output, not a learned perceptual representation.

Keep spatial-mode suppression, shuffles and energy-matched nulls as separate experiments; only add them after this basic assay reports live responses on the target horizons. No re-training or favorable mechanism outcome is claimed here.
