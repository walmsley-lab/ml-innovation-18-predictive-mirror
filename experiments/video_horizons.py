"""Causal multi-horizon evaluation of the real-video recurrent predictor.

Roll out each future frame from the model's *own* previous prediction.
Held-out future observations are used solely for scoring.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import video_residual


def rollout(params, history, count):
    """[history,H,W,1] -> [count,H,W,1]; no future observations."""
    if count < 1 or history.shape[0] < 2:
        raise ValueError("Need at least two observed frames and a positive horizon")
    size = history.shape[1]
    hidden = jnp.zeros((1, size, size, params["local"].shape[-1]), jnp.float32)
    zero = jnp.zeros((1, 1), jnp.float32)

    def warm(state, frame):
        updated, _ = video_residual.step(params, state, frame[None], zero)
        return updated, None

    hidden, _ = jax.lax.scan(warm, hidden, history[:-1])

    def advance(state, _):
        hidden, frame = state
        updated, prediction = video_residual.step(params, hidden, frame, zero)
        return (updated, prediction), prediction[0]

    _, forecasts = jax.lax.scan(advance, (hidden, history[-1:]), None, length=count)
    return forecasts


def compare(params, frames, history=8, horizons=(1, 3, 5, 10), motion_threshold=0.03):
    """Nonoverlapping windows from the final 20% of a chronological recording."""
    if frames.ndim != 4 or frames.shape[-1] != 1:
        raise ValueError("Expected [T,H,W,1] frames")
    horizons = tuple(sorted(set(map(int, horizons))))
    if not horizons or horizons[0] <= 0 or history < 2:
        raise ValueError("Positive horizons and history >= 2 required")
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError("Frames must be finite in [0,1]")
    farthest = horizons[-1]
    start = int(0.8 * len(frames))
    stride = history + farthest
    starts = range(start, len(frames) - stride + 1, stride)
    if not len(starts):
        raise ValueError("Not enough untouched validation frames for requested horizons")
    totals = {h: {"model": [], "persistence": [], "velocity": [], "model_motion": [],
                  "persistence_motion": [], "motion_pixels": 0} for h in horizons}
    for offset in starts:
        observed = jnp.asarray(frames[offset:offset + history])
        prediction = np.asarray(rollout(params, observed, farthest))
        last = frames[offset + history - 1]
        previous = frames[offset + history - 2]
        for h in horizons:
            truth = frames[offset + history + h - 1]
            predicted = prediction[h - 1]
            persistence = last
            velocity = np.clip(last + h * (last - previous), 0.0, 1.0)
            mask = np.abs(truth - last) > motion_threshold
            slot = totals[h]
            slot["model"].append(float(np.mean((predicted - truth) ** 2)))
            slot["persistence"].append(float(np.mean((persistence - truth) ** 2)))
            slot["velocity"].append(float(np.mean((velocity - truth) ** 2)))
            slot["motion_pixels"] += int(mask.sum())
            if mask.any():
                slot["model_motion"].extend(np.square(predicted - truth)[mask].tolist())
                slot["persistence_motion"].extend(np.square(persistence - truth)[mask].tolist())
    return {
        "architecture": "video_residual",
        "mode": "autoregressive_no_future_observations",
        "validation_start_frame": start,
        "nonoverlapping_windows": len(starts),
        "history": history,
        "motion_threshold": motion_threshold,
        "horizons": {str(h): {
            "model_mse": float(np.mean(totals[h]["model"])),
            "persistence_mse": float(np.mean(totals[h]["persistence"])),
            "velocity_mse": float(np.mean(totals[h]["velocity"])),
            "motion_region_model_mse": (float(np.mean(totals[h]["model_motion"]))
                                         if totals[h]["motion_pixels"] else None),
            "motion_region_persistence_mse": (float(np.mean(totals[h]["persistence_motion"]))
                                               if totals[h]["motion_pixels"] else None),
            "motion_pixels": totals[h]["motion_pixels"],
        } for h in horizons},
        "limitation": "One recording, chronological holdout; exploratory, not independent replication",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="recordings/webcam.npz")
    p.add_argument("--parameters", default="runs/webcam-100/parameters.npz")
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--out", default="runs/video-horizons.json")
    args = p.parse_args()
    with np.load(args.input, allow_pickle=False) as f:
        frames = f["frames"].astype(np.float32)
    with np.load(args.parameters, allow_pickle=False) as f:
        params = {key: jnp.asarray(f[key]) for key in f.files}
    result = compare(params, frames, args.history, args.horizons)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
