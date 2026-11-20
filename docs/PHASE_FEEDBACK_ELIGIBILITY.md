# Phase-conditioned feedback eligibility (synthetic probe)

**Question:** Does timing a shared top-down feedback pathway by receiving
phase improve a recurrent update relative to phase-blind feedback?

Run:

```bash
python -m experiments.phase_feedback_eligibility --seeds 11 23 41 \
  --samples 6000 --out runs/phase-feedback-eligibility.json
python -m unittest tests.test_phase_feedback_eligibility -v
```

This is an **analytic, fixed-gate mechanism probe**, not a trained recurrent
visual predictor or a demonstration of biological phase communication.
The one-step state update is `0.65 * sensory + gate * feedback`.
The synthetic generator explicitly makes feedback noise lower near phase zero.
Thus any advantage of the phase gate on this positive control is expected by
construction, not evidence that a model discovered useful timing.

Arms: phase cosine gate; phase-shuffled cosine gate (distribution-matched);
fixed 0.5 gate; validation-tuned phase-blind constant gain (41 candidates);
sensory-only. The second condition removes the phase/reliability relationship,
providing a negative control. All arms share the same input samples, feedback
pathway and recurrent update. Validation and test seeds are disjoint. Report
per-seed classification error and the negative-control result; no pooling with
RGB forecasting MSE or claims of learned phase routing.

**Next decision gate:** Train a phase controller or oscillator end-to-end on
occlusion/feedback tasks, then compare with a parameter/compute-matched learned
non-phase gate and randomized timing. Reject the added phase mechanism if it
does not improve held-out tasks or if the advantage disappears under
phase-blind reliability. This initial probe only checks the evaluation design.
