"""Frozen, observation-only evaluation of open-loop RGB forecasts."""
import numpy as np
from src import baselines, rgb_motion_world

LEADS = (1, 3, 5, 10)


def scenes(first_seed, count, context=8, horizon=10, size=32, **world):
    if count < 1 or context < 2 or horizon < 1:
        raise ValueError("Invalid scenes/context/horizon")
    return [rgb_motion_world.episode(first_seed + i, steps=context+horizon,
                                    size=size, **world) for i in range(count)]


def frames(records):
    return np.stack([record["frames"] for record in records])


def score(forecasts, observed, context, leads=LEADS, threshold=.03):
    forecasts, observed = np.asarray(forecasts), np.asarray(observed)
    if (forecasts.ndim != 5 or observed.ndim != 5 or context < 2
            or forecasts.shape[0] != observed.shape[0]
            or forecasts.shape[2:] != observed.shape[2:]
            or observed.shape[1] < context + forecasts.shape[1]):
        raise ValueError("Inconsistent RGB frames/forecasts")
    if not leads or min(leads) < 1 or max(leads) > forecasts.shape[1]:
        raise ValueError("Leads must be within forecast horizon")
    anchor = observed[:, context-1]
    rows = {}
    for h in leads:
        truth = observed[:, context+h-1]
        error = np.square(forecasts[:, h-1] - truth).mean(axis=-1)
        changed = np.abs(truth-anchor).mean(axis=-1) > threshold
        rows[str(h)] = {
            "mse": float(error.mean()),
            "motion_mse": float(error[changed].mean()) if changed.any() else None,
            "motion_pixels": int(changed.sum()),
            "episodes": int(len(observed)),
        }
    return rows


def controls(contexts, horizon, optical_flow=False):
    """Baselines never receive future RGB frames or world metadata."""
    if contexts.ndim != 5 or contexts.shape[1] < 2 or contexts.shape[-1] != 3:
        raise ValueError("Expected RGB [B,C,H,W,3]")
    methods = {"persistence": baselines.persistence_forecast,
               "constant_velocity": baselines.constant_velocity_forecast}
    if optical_flow:
        methods["optical_flow"] = baselines.optical_flow_forecast
    return {name: np.stack([method(x, horizon) for x in contexts])
            for name, method in methods.items()}


def compare(predictions, observations, context, leads, optical_flow=False):
    horizon = predictions.shape[1]
    all_predictions = {"nca": predictions, **controls(observations[:, :context],
                                                      horizon, optical_flow)}
    return {name: score(values, observations, context, leads)
            for name, values in all_predictions.items()}
