# Regulating plasticity under changing visual dynamics

**Question:** Can a perceptual cellular sheet adapt faster to unexpected changes
without losing competence when familiar dynamics return?

This is an engineering experiment inspired by neuromodulation and homeostasis,
**not** a model of DMT, psychedelic experience, consciousness, or a claim of a
biologically faithful neuromodulator.

## Smallest useful test

Reuse the NumPy \`DevelopmentalSheet\` and its existing spatially local
experience-dependent connections. Generate one continuous RGB scene with a
moving object and static landmark:

1. **Familiar:** fixed motion, learn a predictive state.
2. **Unexpected reversal:** change motion without resetting the object or model.
3. **Return:** restore the original motion and test retention/adaptation.

The observed frame is never used to score its own issued prediction. After
scoring \`forecast[t]\` against \`frame[t]\`, its resulting error may set the
plasticity rate for the **subsequent learning update**. All models see the same
frames, in the same order, from the same initial parameters.

The regulator has three bounded modes:

- **Stabilize**: baseline local plasticity, tracking a slowly moving error
  baseline.
- **Explore**: after two consecutive unusually large errors, increase the
  plasticity rate for a short window.
- **Consolidate**: reduce plasticity after exploration to protect recent changes.

These are explicit heuristics, not learned chemicals or evidence that the model
is discovering alternate realities.

Compare **adaptive**, **fixed-rate**, **matched constant-rate** (same sum of
commanded rates), and **time-shifted adaptive schedule** (same rate distribution,
misaligned in time). The latter is an offline diagnostic; it is not an online
controller. Matched commanded rates are not matched weight-update magnitudes,
because the existing local rule also scales by surprise.

## Run

\`\`\`bash
python -m unittest tests.test_regulated_plasticity -v
python -m experiments.regulated_plasticity \
  --seeds 11 23 41 --segment-frames 40 \
  --size 32 --sheet-size 8 \
  --out runs/regulated-plasticity
\`\`\`

Inspect \`runs/regulated-plasticity/metrics.json\` and the per-seed connection
checkpoints. Each regime records mean, early-third and late-third prequential
MSE. Lower is better; \`shift_recovery_delta\` and
\`return_recovery_delta\` quantify early-to-late improvements, and
\`retention_gap\` compares the first part of the returned regime with the last
part of the familiar regime. Per-frame error and rate curves, mode occupancy,
and connection change are retained. Report all seeds, not just the best run.

## Promotion and caveats

Do not promote the adaptive controller for merely entering an explore mode,
changing weights, or producing coherent internal dynamics. It must show a
repeatable reduction in shift-period error **and** retain/recover previous
predictive ability, versus both fixed and budget-matched controls.

This is a **small 2D diagnostic** with a synthetic, externally switched motion
law. It does not establish object permanence, full world-model learning,
real-webcam generalization, or a computational benefit. Abrupt novelty
detection is confounded with ordinary predictive difficulty, and the simple
sheet may lack the capacity to exploit additional plasticity. If all arms are
equal, that is a useful null result: inspect model sensitivity to the
connections before adding complexity.

Only after a reproducible effect, repeat the same protocol on held-out scene
families and the existing 3D/camera world. No paid Akash deployment is needed
for this experiment.
