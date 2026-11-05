"""Shared diagnostics that separate a genuine null from a dead measurement.

Several experiments in this lab compare an intervention arm against a control
and report the change in open-loop RGB forecasts. When that change is ~0 there
are two very different explanations:

* **Genuine null** -- the forecast still depends on its inputs and responds to
  a known positive-control perturbation, yet the *specific* manipulation under
  test (a transferred core, a suppressed harmonic subspace, a feedback term)
  truly does not change held-out predictions.
* **Dead measurement** -- the open-loop forecast has collapsed to an
  input-independent fixed point (for this model family, an all-zero frame
  pinned against the ``clip(x,0,1)`` rail). Then *no* intervention can change
  anything, so a ~0 delta is uninformative rather than evidence of a null.

These pure functions make that distinction explicit and machine-readable:

* :func:`clip_saturation` -- are the forecast pixels pinned to the clip rails?
* :func:`input_independence` -- do genuinely different inputs yield the same
  forecast? (the direct collapse detector)
* :func:`response` -- by how much does a *known* perturbation move the forecast?
  (the positive control; the caller supplies baseline and perturbed rollouts)
* :func:`measurement_verdict` -- per-lead ``"live"``/``"dead"`` label derived
  from the positive control, so a reader can tell whether a null is meaningful.

No model family or world is assumed here; callers pass finished forecast arrays
shaped ``[batch, horizon, height, width, channels]`` and pick the leads.
"""
import numpy as np

# A per-lead positive-control response (mean absolute forecast change, in pixel
# units on a [0,1] intensity scale) below this is treated as "the measurement
# cannot see anything here". 1e-3 is well under any real forecast signal in
# these sparse worlds (mean pixel intensity ~0.04) yet far above float noise.
DEAD_RESPONSE_FLOOR = 1e-3
# A forecast is "collapsed" at a lead when nearly every pixel is on the clip
# rail AND different scenes yield almost the same frame: an input-independent
# fixed point that no upstream intervention can move.
COLLAPSE_SATURATION = 0.99
COLLAPSE_INDEPENDENCE = 1e-3


def _leads_in_range(forecasts, leads):
    horizon = np.asarray(forecasts).shape[1]
    chosen = tuple(int(lead) for lead in leads)
    if not chosen or min(chosen) < 1 or max(chosen) > horizon:
        raise ValueError("Leads must lie within the forecast horizon")
    return chosen


def clip_saturation(forecasts, leads, low=0.0, high=1.0, eps=1e-6):
    """Fraction of forecast pixels pinned to the clip rails, per lead.

    A value near 1.0 means almost every pixel sits on the ``low``/``high`` rail,
    so upstream state changes cannot propagate through the ``clip`` nonlinearity.
    """
    forecasts = np.asarray(forecasts)
    if forecasts.ndim != 5:
        raise ValueError("Expected forecasts [batch,horizon,H,W,channels]")
    return {str(lead): float(np.mean(
        (forecasts[:, lead - 1] <= low + eps)
        | (forecasts[:, lead - 1] >= high - eps)))
        for lead in _leads_in_range(forecasts, leads)}


def input_independence(forecasts, leads):
    """Mean absolute deviation of the forecast across the batch, per lead.

    Each sample in the batch comes from a different observed scene, so a healthy
    forecaster produces visibly different frames and this is clearly > 0. A value
    at (or extremely close to) 0 means every scene maps to the *same* predicted
    frame: an input-independent fixed point. Trivially 0 for a single sample.
    """
    forecasts = np.asarray(forecasts)
    if forecasts.ndim != 5:
        raise ValueError("Expected forecasts [batch,horizon,H,W,channels]")
    out = {}
    for lead in _leads_in_range(forecasts, leads):
        frame = forecasts[:, lead - 1]
        out[str(lead)] = float(np.abs(
            frame - frame.mean(axis=0, keepdims=True)).mean())
    return out


def response(baseline, variants, leads):
    """Positive-control response: mean |forecast change| vs baseline, per lead.

    ``baseline`` and each array in ``variants`` are forecasts
    ``[batch,horizon,H,W,channels]`` produced from the *same* contexts, differing
    only by a known perturbation the caller applied (wiped memory, scaled readout,
    counterfactual actions, a stronger feedback gain...). If even a large, known
    perturbation moves the forecast by ~0 at some lead, the measurement is dead
    there regardless of what any scientific intervention reports.
    """
    baseline = np.asarray(baseline)
    leads = _leads_in_range(baseline, leads)
    out = {}
    for name, variant in variants.items():
        variant = np.asarray(variant)
        if variant.shape != baseline.shape:
            raise ValueError(f"Variant '{name}' shape differs from baseline")
        out[name] = {str(lead): float(np.abs(
            variant[:, lead - 1] - baseline[:, lead - 1]).mean())
            for lead in leads}
    return out


def measurement_verdict(response_by_variant, leads, floor=DEAD_RESPONSE_FLOOR,
                        independence=None, saturation=None):
    """Classify each lead so a reader can tell what a ~0 result means.

    Three labels:

    * ``"live"`` -- at least one known perturbation moves the forecast by at
      least ``floor`` here, so the measurement works and a ~0 scientific result
      is a genuine null.
    * ``"dead"`` -- the forecast has collapsed to an input-independent clip-rail
      fixed point (needs ``independence`` and ``saturation``) and the positive
      control cannot move it; any ~0 result here is uninformative.
    * ``"insensitive"`` -- the forecast still depends on its inputs, but even the
      positive control barely moves it, so the control is too weak to certify
      the measurement at this lead; interpret with care.

    Without ``independence``/``saturation`` the label is the two-way
    ``"live"``/``"dead"`` split on ``floor`` alone.
    """
    if not response_by_variant:
        raise ValueError("Need at least one positive-control variant")
    leads = tuple(int(lead) for lead in leads)
    verdict = {}
    for lead in leads:
        key = str(lead)
        best = max(variant[key] for variant in response_by_variant.values())
        if best >= floor:
            verdict[key] = "live"
        elif independence is None or saturation is None:
            verdict[key] = "dead"
        elif (saturation[key] >= COLLAPSE_SATURATION
                and independence[key] <= COLLAPSE_INDEPENDENCE):
            verdict[key] = "dead"
        else:
            verdict[key] = "insensitive"
    return verdict


def collapse_report(forecasts, leads, positive_control=None,
                    floor=DEAD_RESPONSE_FLOOR):
    """Bundle the collapse diagnostics (and optional positive control) as JSON.

    ``positive_control`` is a mapping ``{variant_name: forecasts}`` of known
    perturbations of the same scenes as ``forecasts``; when supplied, the report
    includes per-lead response magnitudes and a live/dead verdict.
    """
    saturation = clip_saturation(forecasts, leads)
    independence = input_independence(forecasts, leads)
    report = {
        "clip_rail_saturation": saturation,
        "input_independence": independence,
    }
    if positive_control:
        responses = response(forecasts, positive_control, leads)
        report["positive_control_response"] = responses
        report["measurement_verdict"] = measurement_verdict(
            responses, leads, floor, independence, saturation)
    return report


def perceptual_failure_report(forecasts, records, context, leads=(1, 3, 5, 10)):
    """Calibrate five forecast failure types against evaluator-only RGB truth.

    Additive diagnostic: does not modify the historical live/dead/insensitive
    positive-control verdict. Requires distinct scenes for an interpretable
    across-scene independence test; with one scene that flag is unavailable.
    A diagnostic flag is a *screening heuristic*, not a statistical test.
    """
    from src import rgb_event_metrics as events
    forecasts = np.asarray(forecasts, np.float32)
    if (forecasts.ndim != 5 or len(records) != len(forecasts)
            or forecasts.shape[-1] != 3 or context < 2):
        raise ValueError("Need [episodes,horizon,H,W,RGB] and matched truth")
    leads = _leads_in_range(forecasts, leads)
    metrics = events.score(forecasts, records, context, leads)
    rows = {}
    for lead in leads:
        key = str(lead)
        pred = forecasts[:, lead - 1]
        truth = np.stack([record["frames"][context + lead - 1]
                          for record in records])
        if pred.shape != truth.shape:
            raise ValueError("Truth/prediction image mismatch")
        brightness = float(pred.mean())
        spatial_sd = float(np.mean(pred.std(axis=(1, 2))))
        bright_pred = np.max(pred, axis=-1) > events.FOREGROUND_BRIGHTNESS
        bright_true = np.max(truth, axis=-1) > events.FOREGROUND_BRIGHTNESS
        pred_fraction = float(bright_pred.mean())
        true_fraction = float(bright_true.mean())
        across = (float(np.abs(pred - pred.mean(axis=0, keepdims=True)).mean())
                  if len(pred) > 1 else None)
        truth_spread = (float(np.abs(truth - truth.mean(axis=0, keepdims=True)).mean())
                        if len(truth) > 1 else None)
        can_check_independence = truth_spread is not None and truth_spread > 1e-3
        event = metrics[key]
        flags = []
        if brightness < .07 and spatial_sd < .02 and true_fraction > 0:
            flags.append("black")
        if brightness > .30 and spatial_sd < .03 and pred_fraction > .8:
            flags.append("bright")
        if (pred_fraction > max(.4, 4 * true_fraction)
                and event["foreground_precision"] is not None
                and event["foreground_precision"] < .30):
            flags.append("diffuse")
        if (event["soft_centroid_mae_px"] is not None
                and event["soft_centroid_mae_px"] > 3.0
                and (event["predicted_mass_fraction"] or 0) >= .1):
            flags.append("misplaced")
        if can_check_independence and across <= 1e-5:
            flags.append("input_independent")
        if (event["visible_objects"] > 0
                and event["missing_visible_predictions"] == event["visible_objects"]):
            flags.append("all_objects_hard_missed")
        mse = float(np.square(pred - truth).mean())
        if mse <= 1e-9 and not flags:
            flags.append("accurate")
        rows[key] = {
            "flags": flags, "pixel_mse": mse,
            "mean_intensity": brightness,
            "spatial_std": spatial_sd,
            "across_scene_deviation": across,
            "truth_across_scene_deviation": truth_spread,
            "input_independence_testable": bool(can_check_independence),
            "predicted_foreground_fraction": pred_fraction,
            "true_foreground_fraction": true_fraction,
            "foreground_precision": event["foreground_precision"],
            "foreground_recall": event["foreground_recall"],
            "hue_mass_ratio": event["predicted_mass_fraction"],
            "soft_centroid_error_px": event["soft_centroid_mae_px"],
            "hard_missing_objects": event["missing_visible_predictions"],
            "visible_objects": event["visible_objects"],
        }
    return {"per_lead": rows,
            "limits": "Heuristic thresholds for this synthetic RGB palette; "
             "never infer object identity or promotion from one flag. "
             "Hard missing-object counts incur the full-image-diagonal "
             "penalty in the separate event score. Across-scene dependence "
             "cannot be assessed with one scene or identical truth."}
