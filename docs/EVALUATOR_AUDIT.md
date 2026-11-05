# Evaluator and observability audit

Run `python -m experiments.evaluator_audit` before using synthetic reappearance results.

The audit does **not** test any learned checkpoint. It verifies the RGB-only input boundary, opposing-history pairs with identical last observed frame, perfect future as an **evaluator-only positive control**, black/missing predictions as a negative control, and both barrier and camera-exit return events.

`src.guaranteed_reappearance.model_inputs(record)` exposes **only the observed prefix** rather than a record containing the unseen future. Experimental runners should also explicitly slice context and never import evaluator-only positions, visibility, colors or events into inference. The oracle truth is used exclusively to test scoring correctness.

If identical complete observed histories have different possible futures, the task is not deterministically identifiable. Such cases must be labeled **ambiguous**, not treated as an inevitable model failure.

The RGB event evaluator also has known-case tests for threshold sensitivity, spatial shifts and false detections. Passing these controls proves the scoring contract works on these cases, **not** that model predictions are good, robust across domains, or free from every simulator prior. Before promoting a learned-model finding, test training/test generator separation and check that any world-specific detector priors are declared.
