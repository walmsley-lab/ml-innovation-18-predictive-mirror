"""Event-aware, evaluator-only metrics for frozen RGB forecasts.

The predictor receives images alone; object visibility and color IDs are
consulted exclusively here for evaluation. This is a simple shape/color
diagnostic, not a learned object detector.

Three complementary views of the same forecast, so a low score can be traced
to a cause rather than lumped into one number:

* **Hard localization** (`_center`) -- a brightness+color *threshold* detector.
  It answers "did a correctly coloured disc appear?" and assigns the full image
  diagonal when nothing passes the threshold. Sensitive to the threshold itself.
* **Foreground segmentation** (`foreground_segmentation`) -- IoU / precision /
  recall of the predicted moving-object mask against the true object mask, so we
  see how much of the object (vs background) the forecast reproduces.
* **Soft, threshold-free localization** (`soft_localization`) -- an
  intensity-and-colour-weighted centroid plus the total predicted "mass". Mass
  near zero means the object is genuinely absent from the forecast
  (disappearance); mass present but below the hard gate means the detector
  threshold, not the forecast, produced the miss. This is what separates a
  genuine disappearance from a detection-threshold artifact.
"""
import numpy as np

from src.rgb_motion_world import COLORS

FOREGROUND_BRIGHTNESS = 0.30
# Soft saliency is hue-based and therefore robust to intensity: a dim disc of
# the right colour still counts. HUE_FLOOR is the cosine-similarity below which
# a pixel's colour direction is treated as "not this object" (it rejects the
# grey occluder and the other discs); DARK_FLOOR removes the near-black
# background, whose colour direction is undefined.
HUE_FLOOR = 0.9
DARK_FLOOR = 0.05
# A predicted object "mass" below this fraction of the true object's mass is
# treated as absent for the soft localization error (but still reported raw).
MASS_PRESENT_FRACTION = 0.10


def _center(image, color, threshold=0.28):
    # The synthetic renderer uses saturated, flat-color discs.
    # Background and occluder are excluded by the brightness threshold.
    bright = image.max(axis=-1) > FOREGROUND_BRIGHTNESS
    matching = np.linalg.norm(image - color, axis=-1) < threshold
    yy, xx = np.nonzero(bright & matching)
    if not len(xx):
        return None
    return np.array([yy.mean(), xx.mean()], np.float64)


def object_saliency(image, color, hue_floor=HUE_FLOOR, dark_floor=DARK_FLOOR):
    """Threshold-free, intensity-robust per-pixel weight in [0,1] for a colour.

    The weight is the colour *direction* match (cosine similarity to the object
    colour, rescaled from ``hue_floor``..1 to 0..1), gated only to drop the
    near-black background whose direction is undefined. Because it uses the
    colour direction and not its magnitude, a dim disc of the right hue scores
    the same as a bright one -- so the soft view answers "where does the object
    appear, even if its intensity is wrong?" rather than "is it bright enough?".
    """
    image = np.asarray(image, np.float64)
    color = np.asarray(color, np.float64)
    norms = np.linalg.norm(image, axis=-1)
    color_norm = np.linalg.norm(color)
    cosine = (image @ color) / (norms * color_norm + 1e-9)
    weight = np.clip((cosine - hue_floor) / (1.0 - hue_floor), 0.0, 1.0)
    return np.where(norms > dark_floor, weight, 0.0)


def soft_localization(image, color, hue_floor=HUE_FLOOR, dark_floor=DARK_FLOOR):
    """Return ``(mass, centroid)`` for one object's soft saliency field.

    ``mass`` is the total saliency (threshold-free analogue of "how much of the
    object is present"); ``centroid`` is the saliency-weighted ``[row, col]`` or
    ``None`` when there is effectively no mass.
    """
    saliency = object_saliency(image, color, hue_floor, dark_floor)
    mass = float(saliency.sum())
    if mass <= 1e-9:
        return mass, None
    rows, cols = np.indices(saliency.shape[:2], dtype=np.float64)
    return mass, np.array([(saliency * rows).sum() / mass,
                           (saliency * cols).sum() / mass], np.float64)


def foreground_segmentation(prediction, target, brightness=FOREGROUND_BRIGHTNESS):
    """IoU / precision / recall of predicted vs true moving-object foreground.

    Foreground = pixels brighter than ``brightness`` (the discs, not the black
    background or the dim occluder). ``None`` where a ratio is undefined (no true
    or no predicted foreground).
    """
    true_fg = np.asarray(target).max(axis=-1) > brightness
    pred_fg = np.asarray(prediction).max(axis=-1) > brightness
    intersection = float(np.logical_and(true_fg, pred_fg).sum())
    union = float(np.logical_or(true_fg, pred_fg).sum())
    true_count = float(true_fg.sum())
    pred_count = float(pred_fg.sum())
    return {
        "iou": intersection / union if union > 0 else None,
        "precision": intersection / pred_count if pred_count > 0 else None,
        "recall": intersection / true_count if true_count > 0 else None,
        "true_foreground_pixels": int(true_count),
        "predicted_foreground_pixels": int(pred_count),
    }


def _mean(values):
    return float(np.mean(values)) if values else None


def score(forecasts, records, context, leads=(1, 3, 5, 10), threshold=0.03):
    """Score [episode,lead,height,width,RGB] on synthetic world records.

    Only records' future frames and visibility are used for scoring.
    A missing visible object receives the full image-diagonal penalty.
    Unobservable/occluded object positions are not guessed or scored.

    Each lead row reports, side by side: whole-frame/foreground/motion MSE, the
    *hard* threshold detector (`missing_visible_predictions`,
    `visible_location_mae_px`), foreground segmentation (`foreground_iou/…`), and
    *soft* threshold-free localization (`soft_centroid_mae_px`,
    `predicted_mass_fraction`, `soft_localized_objects`). Comparing the hard and
    soft views tells whether a miss is a genuine disappearance (mass ~0) or a
    detection-threshold artifact (mass present, centroid close).
    """
    forecasts = np.asarray(forecasts)
    if not records or forecasts.ndim != 5 or forecasts.shape[-1] != 3:
        raise ValueError("Need RGB forecasts and at least one world record")
    if len(records) != len(forecasts) or context < 2:
        raise ValueError("Inconsistent episode count or context")
    if not leads or min(leads) < 1 or max(leads) > forecasts.shape[1]:
        raise ValueError("Requested lead is outside the forecast horizon")
    diagonal = float(np.hypot(*forecasts.shape[2:4]))
    rows = {}
    for lead in leads:
        all_squared, fg_errors, motion_errors, location_errors, returns = [], [], [], [], []
        ious, precisions, recalls = [], [], []
        soft_errors, mass_fractions, return_soft_errors, return_mass = [], [], [], []
        missed, visible_targets, foreground_count, moving_count = 0, 0, 0, 0
        soft_localized = 0
        for i, record in enumerate(records):
            frames = record["frames"]
            visible = record["visible"]
            target_index = context + lead - 1
            if target_index >= len(frames) or len(visible) != len(frames):
                raise ValueError("Record does not cover requested horizon")
            target, anchor = frames[target_index], frames[context - 1]
            prediction = forecasts[i, lead - 1]
            if prediction.shape != target.shape:
                raise ValueError("Forecast/image shape mismatch")
            squared = ((prediction - target) ** 2).mean(axis=-1)
            all_squared.append(float(squared.mean()))
            foreground = target.max(axis=-1) > FOREGROUND_BRIGHTNESS
            moving = np.abs(target - anchor).mean(axis=-1) > threshold
            fg_errors.extend(squared[foreground].tolist())
            motion_errors.extend(squared[moving].tolist())
            foreground_count += int(foreground.sum())
            moving_count += int(moving.sum())
            segmentation = foreground_segmentation(prediction, target)
            for key, bucket in (("iou", ious), ("precision", precisions),
                                ("recall", recalls)):
                if segmentation[key] is not None:
                    bucket.append(segmentation[key])
            if len(visible[target_index]) > len(COLORS):
                raise ValueError("Synthetic palette cannot identify >4 objects")
            for obj, is_visible in enumerate(visible[target_index]):
                if not is_visible:
                    continue
                true_center = _center(target, COLORS[obj])
                if true_center is None:
                    continue  # Source's visible flag can include fully overlapped discs.
                visible_targets += 1
                found = _center(prediction, COLORS[obj])
                if found is None:
                    missed += 1
                    error = diagonal
                else:
                    error = float(np.linalg.norm(found - true_center))
                location_errors.append(error)
                # Soft, threshold-free localization for the same object. The
                # reference is the object's *soft* centroid in the true frame, so
                # a perfect forecast scores exactly zero regardless of the hard
                # detector's threshold.
                true_mass, true_soft_center = soft_localization(target, COLORS[obj])
                pred_mass, pred_center = soft_localization(prediction, COLORS[obj])
                fraction = pred_mass / true_mass if true_mass > 1e-9 else 0.0
                mass_fractions.append(fraction)
                is_reentry = (target_index > 0
                              and not visible[target_index - 1, obj])
                if (pred_center is not None and true_soft_center is not None
                        and fraction >= MASS_PRESENT_FRACTION):
                    soft_localized += 1
                    soft_error = float(np.linalg.norm(pred_center - true_soft_center))
                    soft_errors.append(soft_error)
                else:
                    soft_error = None
                if is_reentry:
                    # Hard re-entry error (back-compat) and soft re-entry view.
                    returns.append(error)
                    return_mass.append(fraction)
                    if soft_error is not None:
                        return_soft_errors.append(soft_error)
        rows[str(lead)] = {
            "mse": float(np.mean(all_squared)),
            "foreground_mse": _mean(fg_errors),
            "motion_mse": _mean(motion_errors),
            "foreground_pixels": foreground_count,
            "motion_pixels": moving_count,
            # Hard threshold detector (threshold-sensitive).
            "visible_location_mae_px": _mean(location_errors),
            "visible_objects": visible_targets,
            "missing_visible_predictions": missed,
            # Foreground segmentation of the moving object vs background.
            "foreground_iou": _mean(ious),
            "foreground_precision": _mean(precisions),
            "foreground_recall": _mean(recalls),
            # Soft, threshold-free localization (disappearance vs threshold).
            "soft_centroid_mae_px": _mean(soft_errors),
            "soft_localized_objects": soft_localized,
            "predicted_mass_fraction": _mean(mass_fractions),
            # Object persistence through (dis)appearance; hard + soft re-entry.
            "return_after_invisibility_mae_px": _mean(returns),
            "return_events": len(returns),
            "reappearance_soft_mae_px": _mean(return_soft_errors),
            "reappearance_mass_fraction": _mean(return_mass),
        }
    return rows
