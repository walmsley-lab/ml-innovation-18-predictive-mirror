"""Held-out one-, three-, and five-pixel barrier-occlusion memory test."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from evaluate import score
from src import model, feedforward, multisheet, multisheet_no_feedback
from src.baselines import persistence, velocity
from src.world import sequence


NETWORKS = {"recurrent": model, "feedforward": feedforward,
            "multisheet": multisheet,
            "multisheet_no_feedback": multisheet_no_feedback}


def benchmark(params, architecture, widths, seeds, steps=20):
    result = {}
    for width in widths:
        records = {name: [] for name in ("model", "persistence", "velocity")}
        for seed in seeds:
            frames, actions, truth = sequence(int(seed), steps=steps, wall_width=width)
            forecasts = {
                "model": np.asarray(NETWORKS[architecture].predict(params, frames, actions)),
                "persistence": persistence(frames), "velocity": velocity(frames, actions)}
            for name, forecast in forecasts.items():
                records[name].append(score(forecast, frames, truth["visible_mask"],
                                           truth["occluded"]))
        result[str(width)] = {}
        for name, values in records.items():
            n = sum(v["reappearance_count"] for v in values)
            result[str(width)][name] = {
                "mse": float(np.mean([v["mse"] for v in values])),
                "reappearance_location_mae_px": (
                    sum(v["reappearance_sum_abs_error_px"] for v in values) / n if n else None),
                "reappearance_count": n,
                "occluded_frames": sum(v["occluded_frame_count"] for v in values),
                "episodes": len(values)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--architecture", choices=tuple(NETWORKS), default="recurrent")
    parser.add_argument("--widths", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", default="runs/occlusion-holdout.json")
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 2 or any(v not in (1, 3, 5) for v in args.widths):
        parser.error("Need positive episodes, >=2 steps and widths in {1,3,5}")
    with np.load(args.parameters, allow_pickle=False) as file:
        params = {k: jnp.asarray(file[k]) for k in file.files}
    results = benchmark(params, args.architecture, args.widths,
                        range(60_000_000, 60_000_000 + args.episodes), args.steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved benchmark to {out}")


if __name__ == "__main__":
    main()
