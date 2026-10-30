# Evaluation correction: true barrier occlusion, not off-screen motion

The simulator now exposes two **evaluation-only** truth fields: `in_view` and `occluded` (the object is within the camera's image bounds but hidden by the wall). Neither field is passed through `batch()` or `curriculum_batch()` to the learning model.

A reappearance event counts only when the agent previously saw the object, the immediately preceding frame was hidden **by the wall**, and the target frame contains a visible object. A dot leaving and re-entering the camera's field of view is a different experiment and is **excluded**. Report `reappearance_count` and `occluded_frame_count`; a missing event means the location MAE is `null`, never a perfect score.

The evaluation matrix now contains all four combinations: moving/still object × moving/still camera. In the stationary-object/moving-camera case, the sensor is active, but the object remains fixed in world coordinates; this is crucial for separating external object motion from ego-motion.

**Comparability warning:** This changes the meaning of reappearance MAE. Do **not** combine scores produced before and after this PR in one table. Rerun all model/baseline evaluations on the same frozen holdout seeds after merging it. This is a measurement correction, not evidence that any hypothesis succeeded.
