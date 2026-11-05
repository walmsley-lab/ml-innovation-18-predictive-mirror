# Evaluator and observability audit

Run `python -m experiments.evaluator_audit` before using synthetic reappearance results.

The audit does **not** test any learned checkpoint. It verifies the RGB-only input boundary, opposing-history pairs with identical last observed frame, perfect future as an **evaluator-only positive control**, black/missing predictions as a negative control, and both barrier and camera-exit return events.

`src.guaranteed_reappearance.model_inputs(record)` exposes **only the observed prefix** rather than a record containing the unseen future. Experimental runners should also explicitly slice context and never import evaluator-only positions, visibility, colors or events into inference. The oracle truth is used exclusively to test scoring correctness.

If identical complete observed histories have different possible futures, the task is not deterministically identifiable. Such cases must be labeled **ambiguous**, not treated as an inevitable model failure.

The RGB event evaluator also has known-case tests for threshold sensitivity, spatial shifts and false detections. Passing these controls proves the scoring contract works on these cases, **not** that model predictions are good, robust across domains, or free from every simulator prior. Before promoting a learned-model finding, test training/test generator separation and check that any world-specific detector priors are declared.


## Identifiability and scorer calibration (research amendment)

Split evaluation into at least three regimes: (1) histories that differ before the last observation and imply different futures; (2) *identical complete observed prefixes* with distinct possible future events, which are unidentifiable to a deterministic RGB-only model; and (3) histories with identical future events as a null control. The `identifiable()` helper categorizes these paired cases but does not estimate a posterior distribution. If the future is ambiguous, compare probabilistic predictions using proper scores rather than forcing one deterministic exit to be called uniquely correct.

For color/foreground metrics, record whether segmentation uses a known synthetic palette and whether truth labels only enter evaluator code. A hand-coded palette-aware detector is a **task-specific measurement instrument**, not a general learned object detector. Distinguish evaluator oracle positive controls from trained-model performance.
