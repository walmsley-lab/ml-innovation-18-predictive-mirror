# Recoverable motion memory versus genuinely ambiguous futures

Use `python -m experiments.future_identifiability_audit` to verify three **distinct information regimes**. This is an evaluator / world-design audit, not a learned model result.

| Regime | Same final observation? | Same full observed history? | Different future? | Appropriate inference |
| --- | --- | --- | --- | --- |
| `history_identifiable` | yes | **no** | yes | past observations distinguish the future; memory, Kalman and learned methods can compete |
| `unobserved_turn_ambiguous` | yes | **yes** | yes | unobserved random turn at last fully hidden object position; a deterministic RGB predictor cannot uniquely know the outcome |
| `same_future_null` | yes | yes | **no** | expected null condition, no contrast to infer |

The unobserved-turn case is a deliberate **counterfactual stochastic extension**: after the last observed occluded object frame, an unobserved ± direction decision changes its continuation. Its two futures come from the matched paired world at the same hidden center, speed and radius. Do not claim this stochastic event exists in the original deterministic benchmark.

For two equally likely possible future videos under identical full observations, the conditional pixelwise mean is the Bayes-optimal deterministic **MSE** forecast, with a strictly positive unavoidable error whenever futures differ. `ambiguity_bounds()` computes that diagnostic lower bound and the less accurate single-branch guess, with an oracle-only zero-error reference. A visually averaged video can be MSE-optimal while being physically implausible; in later probabilistic experiments prefer proper distributional scores and event calibration over point-MSE alone.

A successful paired-history memory test is **not** object identity, uncertainty quantification, or a unique NCA contribution. Good linear/Kalman/history-based baselines can succeed too. Train/test adaptation to the ambiguous regime would be a separate experiment and must use held-out stochastic branch decisions.

No new NCA architecture, model training or performance claim is added in this PR.
