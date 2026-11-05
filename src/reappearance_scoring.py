"""Event-complete per-frame scoring of guaranteed reappearance episodes.

Reference visibility and identity stay evaluator-only. Prediction is
committed before the target future frame is inspected.
"""
import numpy as np
from src.rgb_event_metrics import soft_localization

PRESENCE_MASS_THRESHOLD=.10


def score(forecasts, records, context):
    forecasts=np.asarray(forecasts)
    if (forecasts.ndim!=5 or forecasts.shape[-1]!=3 or not records
            or len(records)!=len(forecasts) or context<2):
        raise ValueError("Expected forecasts [N,horizon,H,W,3] and records")
    horizon=forecasts.shape[1]
    results=[]
    for predicted,record in zip(forecasts,records):
        true=record["frames"]
        reasons=record["visibility_reason"]
        if (len(true)<context+horizon or predicted.shape!=true[context:context+horizon].shape
                or record["context"]!=context):
            raise ValueError("Record/context/forecast mismatch")
        color=record["color"]
        # A fully visible *observed* frame calibrates a color-specific
        # visibility threshold; no future labels enter prediction.
        visible_mass=[soft_localization(true[t],color)[0]
                      for t in range(context) if reasons[t]=="visible"]
        if not visible_mass:
            raise ValueError("Missing visible anchor before disappearance")
        reference_mass=max(visible_mass)
        if reference_mass<=0:
            raise ValueError("Absent reference object")
        timeline=[]
        for lead in range(1,horizon+1):
            frame=context+lead-1
            mass,center=soft_localization(predicted[lead-1],color)
            true_mass,true_center=soft_localization(true[frame],color)
            fraction=float(mass/reference_mass)
            present=bool(fraction>=PRESENCE_MASS_THRESHOLD)
            truth_present=bool(reasons[frame] in ("visible","partial"))
            error=(float(np.linalg.norm(center-true_center))
                   if present and truth_present and center is not None
                   and true_center is not None else None)
            timeline.append({
                "frame":frame,"lead":lead,"truth_visibility":str(reasons[frame]),
                "truth_present":truth_present,"predicted_present":present,
                "predicted_mass_ratio":fraction,
                "soft_centroid_error_px":error,
                "missed_visible":bool(truth_present and not present),
                "false_visible_during_hidden":bool(
                    not truth_present and present)})
        event_scores=[]
        for event in record["events"]:
            t=int(event["frame"])
            if t<context or t>=context+horizon:
                continue
            hidden=max(int(event["hidden_start"]),context)
            samples=[row for row in timeline if row["frame"]>=hidden]
            first=next((row["frame"] for row in samples
                        if row["predicted_present"]),None)
            on_return=timeline[t-context]
            event_scores.append({
                "kind":event["kind"],"true_return_frame":t,
                "hidden_start":hidden,
                "predicted_return_frame":first,
                "timing_error_frames":None if first is None else first-t,
                "reappearance_centroid_error_px":on_return["soft_centroid_error_px"],
                "reappearance_mass_ratio":on_return["predicted_mass_ratio"],
                "missed_reappearance":bool(on_return["missed_visible"]),
                "premature_visible_frames":sum(
                    row["false_visible_during_hidden"]
                    for row in timeline if hidden<=row["frame"]<t),
            })
        if not event_scores:
            raise ValueError("No return event falls inside scored horizon")
        results.append({"seed":int(record["seed"]),"mode":record["mode"],
                        "per_frame":timeline,"events":event_scores})
    return {
        "protocol":"guaranteed-reappearance-v1",
        "episodes":len(results),
        "barrier_events":sum(e["kind"]=="barrier_return"
                             for x in results for e in x["events"]),
        "camera_exit_events":sum(e["kind"]=="camera_return"
                                 for x in results for e in x["events"]),
        "results":results,
        "limits":"Color-specific soft-mass threshold; no learned object "
                 "identities. A null centroid/timing means undefined or "
                 "missed, never numerical zero. The events are guaranteed "
                 "by explicit synthetic world construction and scored at "
                 "every frame, not only sparse leads.",
    }
