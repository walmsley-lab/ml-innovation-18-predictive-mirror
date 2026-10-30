"""Evaluate recurrent and image-only baselines on disjoint synthetic episodes."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.baselines import persistence, velocity
from src.model import predict
from src.world import sequence


REGIMES = {
    "still_object_still_camera": dict(object_motion=False, camera=False),
    "moving_object_still_camera": dict(object_motion=True, camera=False),
    "moving_object_moving_camera": dict(object_motion=True, camera=True),
}


def score(forecast, truth, visible_mask):
    """Scores use simulator visible-object masks only after predictions are made."""
    mask = visible_mask[1:]
    diff = (forecast - truth[1:]) ** 2
    return {
        "mse": float(np.mean(diff)),
        "visible_object_pixel_mse": float(np.mean(diff[mask])) if np.any(mask) else None,
    }


def evaluate(params, seeds, steps=12):
    results = {}
    for regime, options in REGIMES.items():
        buckets = {key: [] for key in ("model", "persistence", "velocity")}
        for seed in seeds:
            frames, actions, hidden = sequence(int(seed), steps, **options)
            forecasts = {
                "model": np.asarray(predict(params, frames, actions)),
                "persistence": persistence(frames),
                "velocity": velocity(frames, actions),
            }
            for name, forecast in forecasts.items():
                buckets[name].append(score(forecast, frames, hidden["visible_mask"]))
        results[regime] = {name: {
            metric: float(np.mean([v[metric] for v in values if v[metric] is not None]))
            if any(v[metric] is not None for v in values) else None
            for metric in ("mse", "visible_object_pixel_mse")}
            for name, values in buckets.items()}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", default="runs/pilot/parameters.npz")
    parser.add_argument("--output", default="runs/evaluation.json")
    parser.add_argument("--episodes", type=int, default=32)
    args = parser.parse_args()
    with np.load(args.parameters, allow_pickle=False) as file:
        params = {key: jnp.asarray(file[key]) for key in file.files}
    results = evaluate(params, range(30_000_000, 30_000_000 + args.episodes))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
