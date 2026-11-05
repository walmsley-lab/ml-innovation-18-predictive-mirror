# Roadmap 38: learned communication topology versus rewired null

Use `python -m experiments.roadmap_38_graph_topology`. This is the existing graph-skeleton NCA trained with (a) kinematic graph, (b) degree-matched rewire, (c) disabled messages. Pair initial weights, train data, update counts and held-out actor seeds. Retain last-seen and velocity controls, checkpoint hashes, per-lead visible/hidden/reappearance errors and training time.

**Input privilege:** joint coordinates and visibility are supplied, not derived from RGB. Hidden truth is evaluator-only. Rewiring degrees does not preserve all graph geometry. A graph-harmonic pattern is not proven causally useful by a graph error difference; trained frozen-state mode interventions remain a separate evidence gate.
