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
