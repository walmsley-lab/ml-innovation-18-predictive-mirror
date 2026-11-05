# First campaign: metric and provenance audit (November 2026)

This note supplements the [canonical numerical findings](PERSISTENCE_VS_SIMPLE_FINDINGS.md); numerical tables are not duplicated.

## Clarify localization units

The hard `visible_location_mae_px` metric uses Euclidean pixel distances for detected colored objects but assigns **the entire image diagonal** to every missed visible object. On 32×32 frames this is `sqrt(32²+32²) = 45.25 px`. The reported mean 22–45 px is therefore a mixture of distances and missed-object penalties, **not** a displacement measured only on successful localizations. Always present `missing_visible_predictions / visible_objects` with it.

The soft `soft_centroid_mae_px` is a hue-saliency centroid error only for predictions having sufficient correct-color mass. A forecast without an object produces a **null** soft error; this is not zero displacement. Always show `predicted_mass_fraction` and `soft_localized_objects` beside it.

The uniform-brightness blend-head NCA illustrates two non-equivalent metrics: brightness-based foreground recall can be 1.0 for an image painted bright everywhere, while color-specific predicted mass can be effectively zero. Foreground precision, hue mass, spatial variance and scene dependence are all necessary to reject the false win.

## Reappearance not measured

The earlier evaluator counts a re-entry only if an object is visible at a selected forecast lead and invisible at the immediately preceding frame. Its sampled leads may omit genuine transitions, and the generic `occlusion_long` world does not certify a complete visible → fully occluded → returned sequence within a scored window. Zero scored re-entry events therefore do not support an object-permanence conclusion. Guarantee event coverage, separate partial occlusion from camera exit and score each forecast frame.

## Provenance limits

The campaign numbers came from the operator's report of the 150-step perceptual and 80-step Gaussian runs. The original local `scratchpad/campaign/{perceptual,gaussian}/metrics.json`, checkpoints and example predictions have **not been independently inspected or imported** here. Exact command, git SHA, selected checkpoint hashes, per-seed records, hardware, synchronization and timing repetitions cannot be certified from the transcript. A new archive tool may verify them **only when those original files are supplied**; replayed examples must be labeled as replay, not original campaign output.

Gaussian rendering, oracle-state dynamics, RGB-inferred slots and active view evaluation are now **implemented in the codebase**; this does not demonstrate that they outperform persistence or learn object permanence.

The appropriate conclusion is **no demonstrated cellular/Gaussian advantage in the evaluated small, short-trained configurations**. It does not mean simple models already solve the task, or that recurrence cannot help under richer conditions.
