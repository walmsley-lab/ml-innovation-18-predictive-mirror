"""Event-aware, evaluator-only metrics for frozen RGB forecasts.

The predictor receives images alone; object visibility and color IDs are
consulted exclusively here for evaluation. This is a simple shape/color
diagnostic, not a learned object detector.
"""
import numpy as np

from src.rgb_motion_world import COLORS


def _center(image, color, threshold=0.28):
    # The synthetic renderer uses saturated, flat-color discs.
    # Background and occluder are excluded by the brightness threshold.
    bright = image.max(axis=-1) > 0.30
    matching = np.linalg.norm(image - color, axis=-1) < threshold
    yy, xx = np.nonzero(bright & matching)
    if not len(xx):
        return None
    return np.array([yy.mean(), xx.mean()], np.float64)


def score(forecasts, records, context, leads=(1, 3, 5, 10), threshold=0.03):
    """Score [episode,lead,height,width,RGB] on synthetic world records.

    Only records' future frames and visibility are used for scoring.
    A missing visible object receives the full image-diagonal penalty.
    Unobservable/occluded object positions are not guessed or scored.
    """
    forecasts = np.asarray(forecasts)
    if not records or forecasts.ndim != 5 or forecasts.shape[-1] != 3:
        raise ValueError("Need RGB forecasts and at least one world record")
    if len(records) != len(forecasts) or context < 2:
        raise ValueError("Inconsistent episode count or context")
    if not leads or min(leads) < 1 or max(leads) > forecasts.shape[1]:
        raise ValueError("Requested lead is outside the forecast horizon")
    rows = {}
    for lead in leads:
        all_squared, fg_errors, motion_errors, location_errors, returns = [], [], [], [], []
        missed, visible_targets, foreground_count, moving_count = 0, 0, 0, 0
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
            foreground = target.max(axis=-1) > 0.30
            moving = np.abs(target - anchor).mean(axis=-1) > threshold
            fg_errors.extend(squared[foreground].tolist())
            motion_errors.extend(squared[moving].tolist())
            foreground_count += int(foreground.sum())
            moving_count += int(moving.sum())
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
                    error = float(np.hypot(*target.shape[:2]))
                else:
                    error = float(np.linalg.norm(found - true_center))
                location_errors.append(error)
                # Not necessarily an occlusion: a re-entry may be off-screen.
                if target_index > 0 and not visible[target_index - 1, obj]:
                    returns.append(error)
        rows[str(lead)] = {
            "mse": float(np.mean(all_squared)),
            "foreground_mse": float(np.mean(fg_errors)) if fg_errors else None,
            "motion_mse": float(np.mean(motion_errors)) if motion_errors else None,
            "foreground_pixels": foreground_count,
            "motion_pixels": moving_count,
            "visible_location_mae_px": float(np.mean(location_errors)) if location_errors else None,
            "visible_objects": visible_targets,
            "missing_visible_predictions": missed,
            "return_after_invisibility_mae_px": float(np.mean(returns)) if returns else None,
            "return_events": len(returns),
        }
    return rows
