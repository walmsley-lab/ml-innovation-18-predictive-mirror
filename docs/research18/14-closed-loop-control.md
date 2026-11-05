# PR 14 — Closed-loop interception calibration

Causal, one-shot intercept from partially occluded trajectory history.
Implements a reactive last-seen floor, causal finite-difference velocity
control, and a hit-rate/error evaluator. This is **not yet a learned-policy
comparison** and does not model active camera controls.

Following PR 10/11/12/13: evaluate frozen world states under identical action
budgets, add controlled camera information-gathering actions, and report
task-anchored sensing benefit rather than just entropy reduction.
