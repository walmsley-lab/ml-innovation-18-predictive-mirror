"""Pure temporal matching of issued predictions against later observations."""
import numpy as np


def issue(pending, index, stamp_ns, current, forecast):
    """Append a prediction made using only observations through index."""
    pending.append((int(index), int(stamp_ns),
                    np.asarray(current).copy(), np.asarray(forecast).copy()))
    return pending


def resolve(pending, index, stamp_ns, actual, lead):
    """Only score predictions whose *specified* future frame arrived now."""
    remaining, pairs = [], []
    for issued_at, emitted_ns, last, forecast in pending:
        age = index-issued_at
        if age == lead:
            if lead < 1 or lead > len(forecast):
                raise ValueError("Missing forecast for requested lead")
            pairs.append({"issued_at": issued_at, "target_index": index,
                          "observed": last, "prediction": forecast[lead-1],
                          "actual": np.asarray(actual).copy(),
                          "elapsed_ms": (stamp_ns-emitted_ns)/1e6})
        elif age < lead:
            remaining.append((issued_at,emitted_ns,last,forecast))
        # Predictions that missed the target are expired, not misaligned.
    return remaining,pairs


def metrics(pairs):
    if not pairs:
        return {"scored":0,"model_mse":None,"persistence_mse":None,
                "median_observed_horizon_ms":None}
    model=[float(np.square(p["prediction"]-p["actual"]).mean()) for p in pairs]
    hold=[float(np.square(p["observed"]-p["actual"]).mean()) for p in pairs]
    return {"scored":len(pairs),"model_mse":float(np.mean(model)),
            "persistence_mse":float(np.mean(hold)),
            "median_observed_horizon_ms":float(np.median(
                [p["elapsed_ms"] for p in pairs]))}
