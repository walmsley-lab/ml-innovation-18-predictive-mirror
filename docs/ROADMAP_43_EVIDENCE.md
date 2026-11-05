# Roadmap 43 — provenance-first cross-architecture evidence

Run `python -m experiments.roadmap_43_evidence runs/roadmap-37/metrics.json runs/roadmap-38/metrics.json ...`. It accepts implemented roadmap metrics (36–42), normalizes numeric outcomes without changing their units or model input privileges and writes a portable, hash-linked evidence summary.

**No global leaderboard.** Group by experiment, protocol, input representation, domain, horizon, metric and unit. Only matching paired independent seeds produce exploratory paired-bootstrap differences. Results with one or no independent seed stay descriptive. A given skeleton with oracle joint observations may not be declared better than raw RGB on its lower position error; a 4-bit *simulated* circuit may not be declared energy-efficient without physical substrate measurements.

Retain whole-run calibration, preprocessing version, timestamp semantics, hardware conditions, train budgets and checkpoint digests in upstream manifests. A future canonical evidence schema should make these fields machine-required and add Pareto frontiers only when measured cost is comparable. This experiment is a deliberately small migration-free reader, not a plugin registry or data warehouse.
