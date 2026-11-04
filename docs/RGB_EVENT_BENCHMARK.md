# Frozen RGB event benchmark (pilot)

The normal RGB benchmark reports full-frame and thresholded motion MSE. This
independent evaluator adds **foreground error, visible-object localization,
missing detections, and return-after-invisibility** on the same frozen
synthetic episodes. The evaluator reads synthetic colors and visibility
only *after* the forecaster commits its predictions. It does not train or
modify an NCA.

Each visible object is now measured **three complementary ways** so a low score
has a traceable cause (see [the evaluator verification](#verify-the-evaluator-first)):

- **Hard localization** — a brightness+color *threshold* detector
  (`missing_visible_predictions`, `visible_location_mae_px`); a miss costs the
  full image diagonal. Sensitive to the threshold itself.
- **Foreground segmentation** — `foreground_iou` / `foreground_precision` /
  `foreground_recall` of the predicted moving-object mask vs the true one: how
  much of the object (rather than background) the forecast reproduces.
- **Soft, threshold-free localization** — a hue-based, intensity-robust
  saliency centroid (`soft_centroid_mae_px`) and its colour **mass**
  (`predicted_mass_fraction`, `soft_localized_objects`): *where does the object
  appear, even if its colour or intensity is imperfect, and how much of it is
  present at all.*

```bash
python -m experiments.rgb_event_benchmark --episodes 16 --balls 2
python -m experiments.rgb_event_benchmark --checkpoint runs/pilot/frozen-rgb-campaign/seed-11-nca.npz --episodes 16 --balls 2
# Side-by-side panels with true/predicted centroid overlays (PNGs need Pillow;
# a JSON manifest is always written):
python -m experiments.rgb_event_gallery --checkpoint runs/pilot/frozen-rgb-campaign/seed-11-nca.npz --episodes 4
python -m unittest tests.test_rgb_event_metrics tests.test_rgb_event_gallery -v
```

## Verify the evaluator first

`tests/test_rgb_event_metrics.py` pins the evaluator on known cases **before**
any model conclusion is drawn: a perfect forecast (zero error, IoU 1, mass 1,
no miss), a shifted object (both localization views report ~the shift), a
missing object (hard miss **and** ~zero soft mass — a genuine disappearance), a
globally dimmed object (hard detector misses, soft view keeps it — a
detection-threshold artifact), and a false detection (extra foreground lowers
precision). This is what lets us separate the two causes below.

## Why the frozen NCA "missed 25/25" objects — both, by horizon

Running the frozen campaign checkpoint (`seed-11-nca.npz`, 16 episodes) through
all three views shows the hard "every object missing" result is **two different
failures stacked by lead**:

| lead | hard miss | `loc_mae_px` | `foreground_recall` | `mass_fraction` | `soft_centroid_mae_px` |
|---|---|---|---|---|---|
| 1 | 26/26 | 45.3 (max) | 0.64 | 0.95 | **1.21** |
| 3 | 25/25 | 45.3 (max) | 0.21 | 1.17 | 3.51 |
| 5 | 25/25 | 45.3 (max) | 0.00 | **0.00** | — |
| 10 | 25/25 | 45.3 (max) | 0.00 | **0.00** | — |

- **Leads 1–3 — detection-threshold artifact.** The object *is* in the forecast:
  colour mass ≈1 and the soft centroid is 1.2 px from truth at lead 1 — as good
  as constant velocity (0.99 px). The 26/26 "miss" is the hard brightness+colour
  gate rejecting a slightly dim, soft disc, not an absent object.
- **Leads 5–10 — genuine disappearance.** Colour mass drops to 0 and no soft
  centroid exists: the open-loop forecast has collapsed to a near-black frame
  (the `clip(image + 0.20·tanh(...), 0, 1)` rollout reaches an input-independent
  all-black clip-rail fixed point), so there is truly nothing to detect.

The galleries (`episode-*.png`) make this visible: the predicted red centroid
dot sits inside the true green ring at +1f/+3f, then vanishes at +5f/+10f while
the true ring remains. **Takeaway: never read the hard `missing_visible_predictions`
alone — pair it with `predicted_mass_fraction` to tell disappearance from a
threshold.**

## Interpretation

- Foreground is thresholded from **target RGB**, hence is evaluator-only.
- A visible synthetic object is detected by matching its known palette
  color. A missing forecast receives an image-diagonal distance penalty.
  This tests simple colored-disc localization, **not object identification**.
- Return-after-invisibility may reflect either an off-screen re-entry or
  barrier occlusion. **Do not present it as a pure occlusion/reappearance score.**
- Missing events produce `null`, not fabricated zero errors.
- Synthetic seed separation from training and checkpoint selection must be
  checked externally; default test seeds are 81000+.
- This first contract does not yet establish OOD robustness, multi-run
  confidence intervals, geometry or spatial memory. Follow-on simulator
  and benchmark PRs should add unambiguous event metadata and stronger
  learned controls, without silently changing the v1 metric meaning.

The underlying `rgb_motion_world.episode` exposes state for diagnostics,
but `src.rgb_benchmark.frames` is the only data passed to the predictor.
