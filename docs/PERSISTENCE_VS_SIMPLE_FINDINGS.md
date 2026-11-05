# Do persistent / cellular representations beat simpler predictors?

**Question.** Across object reappearance, unseen motion, viewpoint change, and
**computational cost** — not aggregate image MSE — do persistent representations
and cellular dynamics give a measurable advantage over simpler predictors, or do
simpler approaches remain sufficient?

**Verdict (at these tested CPU budgets): no demonstrated benefit from additional complexity.**
Plain persistence beats the RGB-inferred Gaussian predictor on viewpoint change;
cheap CNN / transport match or beat the expensive cellular NCA variants on the
perceptual ladder; cellular dynamics add nothing over static/velocity Gaussian
controls; and the one cellular arm with a flattering headline number
(`nca_blend`) is a **degenerate uniform-output artifact**, not a win. The
cellular/structured models cost **2–10× more compute** for, at best, a modest
and metric-dependent localization edge.

Runners: `experiments/rgb_world_ladder_training.py`
([protocol](RGB_WORLD_LADDER_TRAINING.md)) and
`experiments/rgb_gaussian_nca.py` ([protocol](RGB_GAUSSIAN_NCA.md)). Budgets:
perceptual 6 families × 2 curricula × 3 seeds × 150 steps, tested on 7 held-out
domains; Gaussian 3 seeds × 80 steps, tested on `ood_camera`.

## Computational cost (the elevated axis)

| model | params | train wall-clock | inference / rollout |
|---|---|---|---|
| **cnn** | 1035 | **3.9 s** | **7.0 ms** |
| **transport** | 540 | **4.2 s** | 11.5 ms |
| hierarchical | 2288 | 9.7 s | 16.2 ms |
| nca (plain) | 2315 | 39.3 s | 34.4 ms |
| nca_blend | 2324 | 39.1 s | 35.4 ms |
| routing | 617 | 28.3 s | **73.0 ms** |

The cellular/structured models are **2.3–10× slower at inference** and up to
**10× slower to train** than the CNN. `routing` — which localizes the most
objects (below) — is the single most expensive arm, ~10× the CNN at inference.

## Viewpoint change

**`ood_camera` (Gaussian predictor), mean over seeds — persistence wins:**

| arm | MSE | visible-geometry MSE |
|---|---|---|
| **baseline: persistence** | **0.0113** | **0.1051** |
| gaussian: static | 0.0177 | 0.1821 |
| gaussian: velocity | 0.0201 | 0.2048 |
| gaussian: cellular | 0.0200 | 0.2040 |
| baseline: constant-velocity | 0.0209 | 0.1868 |

Plain persistence beats the RGB-inferred Gaussian predictor on both metrics, and
**cellular ≈ velocity ≈ worse than static** — the persistent/cellular machinery
buys nothing. One of three seeds selected step 0 (training never beat
initialization).

**`camera_pan` (perceptual ladder):** foreground recall ≈ 0.00 for every real
model. This is an unknown-action camera shift for RGB-only arms, so universal
failure is expected and is not a differentiator.

## Unseen motion & reappearance (perceptual ladder)

No model localizes at this budget. Every real model misses most objects
(missed 2.5–7.3 of ~7.3 visible; hard-detection location error 22–45 px on a
32-px frame). **The hard localization average includes a full-image diagonal penalty (45.25 px at 32×32) for each missed visible object; it is not the mean displacement of successfully detected centers.**

| model | visible-location MAE px | missed / visible objects |
|---|---|---|
| routing | 22.1 | 2.50 / 7.29 |
| hierarchical | 24.5 | 2.90 / 7.29 |
| cnn | 31.9 | 4.50 / 7.29 |
| nca (plain) | 34.6 | 5.07 / 7.29 |
| transport | 44.6 | 7.14 / 7.29 |
| nca_blend | 45.3 | 7.29 / 7.29 (misses all) |

The structured models (`routing`, `hierarchical`) localize marginally more
objects than `cnn` or plain `nca` — but it is a modest, metric-dependent edge
(soft-centroid scoring actually favours `transport`) at 2–10× the compute, not a
win. Plain `nca` is less competitive on this narrow measured protocol, not ruled out more generally.

**Reappearance is not directly measurable in this run.** The dedicated
return-after-invisibility probe detected **zero** qualifying events in every test
domain (`occlusion_long`, `combined`, `camera_pan`) at the +10 horizon, so no
reappearance conclusion is drawn; `occlusion_long` foreground recall is the only
available proxy and shows the same no-localization picture.

## The `nca_blend` artifact

`nca_blend` posts the best-looking aggregate numbers — foreground recall
**1.000**, lowest foreground MSE (~0.147) — but these are an artifact of a
**degenerate uniform-brightness collapse**:

* `predicted_mass_fraction = 0.0`, foreground precision ≈ 0.011 (chance),
* `missing_visible_predictions = 4 / 4` (localizes **zero** objects),
* identical values across every seed, curriculum and domain (input-independent
  constant output).

It paints the bright mean everywhere, which trivially "covers" every true
foreground pixel (recall 1.0) and lowers MSE. The anti-collapse blend head
escapes the **black** absorbing fixed point (as in the
[budget sweep](TRAINING_BUDGET_SWEEP.md)) only to settle on a **gray** one at
this budget. A clear illustration of why aggregate recall/MSE were deprioritized
here.

## Limits

Undertrained CPU regime (150 / 80 steps, 32-px / 24-px worlds), consistent with
the [budget sweep](TRAINING_BUDGET_SWEEP.md) finding that even 1200 steps did not
localize. The honest claim is therefore **"no measurable advantage at achievable
budgets,"** not "cellular dynamics can never help." Equal SGD steps and image
loss do not imply equal FLOPs or capacity; initializations are model-specific;
curriculum (linear vs mixed) showed small, inconsistent differences with no clear
winner. Checkpoints and per-domain reports are saved so a larger run can
re-score the same protocol. No result here is promoted to the README. These numbers are operator-reported: the original local metrics, checkpoints, exact launch command, source revision, and timing conditions have not been independently verified. See the [metric and provenance audit](CAMPAIGN_FINDINGS_20261104.md).
