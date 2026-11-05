# Separate reappearance, identity, and irreducible uncertainty

```bash
python -m experiments.identity_uncertainty_audit
python -m unittest tests.test_identity_uncertainty_world -v
```

The existing paired-history benchmark asks whether motion **history** improves return prediction. This PR adds a different question: can an RGB observer determine **which object** emerged when two indistinguishable objects could have swapped identities behind an occluder?

The `identical` condition deliberately constructs two hidden identity assignments with exactly identical full RGB videos. No image-based predictor can distinguish them. A fair evaluator must report image-level identity as undefined in this regime rather than reward a fortunate hidden-state guess. The `distinct` color condition is an elementary observable identity control, not evidence of general tracking.

A uniform probability over the two hidden assignments provides a simple calibrated baseline; binary Brier and log scores are available for future probabilistic models. The audit is **not** a learned uncertainty model, multi-object tracking achievement or active-gaze policy.

Next: introduce a viewpoint/action that reveals the previously hidden identity, compare passive vs active observation, and keep its action privileges explicit. Do not introduce stochastic NCAs until a basic classical belief tracker and observation policy have been evaluated.


## External reference tasks (not our novelty)

- [IntPhys](https://intphys.cognitive-ml.fr/benchmark/description.html) already tests object permanence with matched possible/impossible video events and parametric occlusion duration/object movement. Our paired-history task does **not** invent occlusion reasoning.
- [Physion](https://physion-benchmark.github.io/) evaluates visual physical prediction over realistic 3D scenes, including object contact. It is a useful external competence/transfer reference, not numerically comparable with this 32x32 identity-observability audit.
- [IntPhys 2](https://arxiv.org/abs/2506.09849) expands violation-of-expectation physics principles, including permanence and continuity.

The relevant narrower question is how *observable history*, trained recurrent state and explicit identity alternatives interact under a controlled input budget. Avoid claiming a general object-permanence benchmark is novel.
