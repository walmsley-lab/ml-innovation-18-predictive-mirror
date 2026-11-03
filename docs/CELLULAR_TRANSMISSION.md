# Pretrain cellular signal transmission before visual prediction

This small, independent JAX experiment asks whether shared local cells
can transmit a signed input pulse from one source cell to a destination
identified by an input query mask, then preserve the message for several
updates.

```bash
python -m experiments.cellular_transmission --steps 80 --seeds 11 23 41
python -m unittest tests.test_cellular_transmission -v
```

- 16x16 default grid, shared 3x3 learned transition, eight hidden channels.
- The signed source pulse occurs only at tick zero. The destination
  location is provided by a query mask; the **target sign is withheld**.
- Train/validation source-destination distances are 2..4 Manhattan cells;
  held-out distances are 5..7.
- Compare the same initialized rule with and without lateral messages.
  Readout and cell weights are learned end-to-end under the same updates.
- Report test sign accuracy, amplitude, error, validation-selected step and
  independent model seeds. An untrained step-zero checkpoint is eligible.

**This is a competence probe, not proof of useful communication**:
a successful rule might merely diffuse a signed signal. Stability at
longer horizons, distractor rejection, routing under obstacles, and
transfer to RGB forecasting are independent follow-up experiments.
No separate per-cell weights, biological specialization, or learned
references are claimed. The existing visual models remain untouched.
