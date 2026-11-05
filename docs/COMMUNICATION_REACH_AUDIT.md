# Communication reach versus displacement and compute

```bash
python -m experiments.communication_reach_audit --seeds 11 23 41
python -m unittest tests.test_communication_reach_audit -v
```

Single-site **causal perturbations** measure change magnitude by radial
distance, after fixed cellular ticks and prediction horizons. Separate
measurements for RGB output, isolated hidden-state updates, learned
flow-routing weights and fast/slow-sheet feedback prevent treating all
paths as equally local.

The theoretical `k*r` bound is only for local-cell message passing.
An image gather/splat accesses distant content directly; its flow
estimator must still compute a useful displacement. A large observed
response is not by itself evidence that the model uses the signal
for accurate prediction.

The memory experiment reports a conservative activation-size lower
bound and Python-traced host allocations versus BPTT horizon.
**It does not measure GPU device peak HBM**. The model probes use random
weights with an intentionally nonzero NCA RGB readout; substantive
claims require repeating with trained frozen checkpoints and paired
held-out displacement strata.

For confirmatory work, score absolute errors, local misses, joint
failures and persistence-relative skill with a frozen denominator
floor. Compare lesioning fast/slow feedback and any encoder skips at
equal parameter/compute budgets.
