# RGB-only history controls for paired occlusion

Use `python -m experiments.history_baseline_eval` to compare four **independent** baselines on the paired-history world; this does not train an NCA or import previously published scores.

| Method | Inputs | Privilege |
| --- | --- | --- |
| Persistence | last observed RGB frame | RGB-only |
| RGB history linear | colored moving pixels estimated from observed RGB frames; fit straight trajectory | RGB-only; assumes one saturated moving blob / static scene |
| RGB history Kalman | same observed RGB detections; constant-velocity transition, explicit process and measurement noise | RGB-only; same declared structural assumptions |
| Known-prior linear reference | hardcoded generator color, disc radius, occluder geometry | **Known simulator primitives**, not a fair generic RGB-only comparator |

The two new methods do not access `COLORS`, event annotations, positions, velocities, hidden state, or future RGB. Their assumptions are **not** general object detection: they use a chroma threshold, size estimation and visually inferred static occluder. On textured scenes, multiple overlapping objects, or camera motion they may fail; that failure is a reason to strengthen the baseline, not evidence for an NCA.

Evaluation groups two opposite histories sharing the final frame into one statistical scene pair, reports event success (return timing +/-1 frame, <=3 px location, no premature visibility), image MSE and future-contrast error. No confidence interval is inferred from a three-pair smoke. Future comparative campaigns should integrate these `rgb_only` methods and include uncertainty reporting, select any tuned hyperparameters using validation only, and never select a comparator on final test.

The baseline runner is standalone to avoid changing the preexisting object-permanence campaign during parallel research PRs. The frozen campaign PR will integrate its observation contract explicitly.


## Interpretation from prior-art review

A Kalman filter has an explicit uncertainty covariance; multi-object filters with track association can preserve identities. The current single-object `kalman_history()` is deliberately narrower and does **not** implement multi-object tracking, identity assignment or calibrated video uncertainty. Do not claim that Kalman methods inherently lack these capabilities.

An autoregressive per-pixel AR(1)/AR(2) baseline is a possible later diagnostic, but on translating colored objects it can be a weak and potentially misleading comparator. Before adding it, establish whether existing RGB history regression and Kalman filters pass the frozen history-contrast tests. This PR keeps a small fair baseline suite and preserves modern learned reference models for the following comparison stage.

The learned-model benchmark must test both matched training resources and an adequately trained reference model. Moving-MNIST published MSE values are not numerically comparable with this normalized RGB synthetic task.
