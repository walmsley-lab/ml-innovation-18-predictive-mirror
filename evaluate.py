"""Evaluate recurrent and image-only baselines on disjoint synthetic episodes."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.baselines import persistence, velocity
from src import model as recurrent, feedforward, multisheet, multisheet_no_feedback
from src.world import sequence


REGIMES = {
    "still_object_still_camera": dict(object_motion=False, camera=False),
    "moving_object_still_camera": dict(object_motion=True, camera=False),
    "still_object_moving_camera": dict(object_motion=False, camera=True),
    "moving_object_moving_camera": dict(object_motion=True, camera=True),
}


def score(forecast, truth, visible_mask, occluded):
    """Score actual barrier occlusion, never mere sensor-field disappearance."""
    mask = visible_mask[1:]
    diff = (forecast - truth[1:]) ** 2
    reappearance_errors = []
    had_object = bool(visible_mask[0].any())
    for t in range(1, len(visible_mask)):
        # Require actual barrier occlusion *and* a prior sighting. Off-screen
        # excursions are a different task and must not inflate this metric.
        if had_object and bool(occluded[t - 1]) and visible_mask[t].any():
            target_position = np.argwhere(visible_mask[t, ..., 0])[0]
            estimate = forecast[t - 1, ..., 0]
            if float(estimate.max()) < 0.5:
                reappearance_errors.append(float(truth.shape[2]))  # no detection: penalized miss
            else:
                predicted = np.array(np.unravel_index(np.argmax(estimate), estimate.shape))
                reappearance_errors.append(float(np.linalg.norm(predicted - target_position, ord=1)))
        had_object |= bool(visible_mask[t].any())
    return {
        "reappearance_sum_abs_error_px": float(sum(reappearance_errors)),
        "reappearance_count": len(reappearance_errors),
        "occluded_frame_count": int(np.count_nonzero(occluded)),
        "mse": float(np.mean(diff)),
        "visible_object_pixel_mse": float(np.mean(diff[mask])) if np.any(mask) else None,
    }


def evaluate(params, seeds, steps=12, architecture="recurrent"):
    model = {"recurrent": recurrent, "feedforward": feedforward, "multisheet": multisheet,
             "multisheet_no_feedback": multisheet_no_feedback}[architecture]
    results = {}
    for regime, options in REGIMES.items():
        buckets = {key: [] for key in ("model", "persistence", "velocity")}
        for seed in seeds:
            frames, actions, hidden = sequence(int(seed), steps, **options)
            forecasts = {
                "model": np.asarray(model.predict(params, frames, actions)),
                "persistence": persistence(frames),
                "velocity": velocity(frames, actions),
            }
            for name, forecast in forecasts.items():
                buckets[name].append(score(forecast, frames, hidden["visible_mask"],
                                           hidden["occluded"]))
        results[regime] = {}
        for name, values in buckets.items():
            n = sum(v["reappearance_count"] for v in values)
            results[regime][name] = {
                metric: float(np.mean([v[metric] for v in values if v[metric] is not None]))
                if any(v[metric] is not None for v in values) else None
                for metric in ("mse", "visible_object_pixel_mse")}
            results[regime][name]["reappearance_location_mae_px"] = (
                sum(v["reappearance_sum_abs_error_px"] for v in values) / n if n else None)
            results[regime][name]["reappearance_count"] = n
            results[regime][name]["occluded_frame_count"] = sum(
                v["occluded_frame_count"] for v in values)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", default="runs/pilot/parameters.npz")
    parser.add_argument("--output", default="runs/evaluation.json")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--architecture", choices=("recurrent", "feedforward", "multisheet", "multisheet_no_feedback"), default="recurrent")
    args = parser.parse_args()
    with np.load(args.parameters, allow_pickle=False) as file:
        params = {key: jnp.asarray(file[key]) for key in file.files}
    results = evaluate(params, range(30_000_000, 30_000_000 + args.episodes), architecture=args.architecture)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
