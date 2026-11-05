# Capabilities and actual costs, not hypothetical efficiency

```bash
python -m experiments.accuracy_cost_audit --repeats 15
python -m unittest tests.test_accuracy_cost_frontier -v
```

The first mode runs actual synchronized inference for the existing RGB NCA and motion-transport models on the **same** synthetic input size and reports median/p90 latency. This is a device-specific timing measurement, not an accuracy result or energy measurement.

A second mode consumes real, same-protocol model quality/cost rows with `--input measured-results.json`, and computes Pareto dominance: a model is dominated if another is no worse on success and every selected cost, and strictly better on at least one. Keep incompatible tasks, hardware, inference lengths and input privileges out of one frontier.

Energy (joules) and memory traffic require actual device-specific measurement or clearly labeled estimates. Leave unavailable values null. The goal is to find good performance-versus-cost tradeoffs as models learn—not to use FLOPs as a substitute for physical efficiency.

Reuse the repository's existing analytical communication-accounting work when making claims about local versus remote links. This PR adds no infrastructure framework and no assertion of NCA efficiency.
