"""Run repeatable small training/evaluation matrices without a training framework."""

import argparse
import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from evaluate import evaluate
from train import train


def aggregate(rows):
    """Keep individual runs visible; summarize only identical configurations."""
    grouped = {}
    for row in rows:
        key = (row["architecture"], row["curriculum"])
        grouped.setdefault(key, []).append(row)
    result = {}
    for (architecture, curriculum), values in grouped.items():
        scores = [v["reappearance_location_mae_px"] for v in values
                  if v["reappearance_location_mae_px"] is not None]
        result[f"{architecture}:{curriculum}"] = {
            "seeds": [v["seed"] for v in values],
            "n": len(values),
            "moving_camera_reappearance_mae_mean_px": float(np.mean(scores)) if scores else None,
            "moving_camera_reappearance_mae_std_px": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
            "note": "descriptive only; no claim of significance",
        }
    return result


def run(output, seeds, architectures, curricula, steps, batch_size, episodes, require_gpu):
    """Train/evaluate each condition on untouched deterministic holdout episodes."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for architecture in architectures:
        for curriculum in curricula:
            for seed in seeds:
                name = f"{architecture}-{curriculum}-seed{seed}"
                directory = root / name
                train(seed=seed, steps=steps, batch_size=batch_size, output=directory,
                      require_gpu=require_gpu, curriculum=curriculum, architecture=architecture)
                with np.load(directory / "parameters.npz", allow_pickle=False) as archive:
                    params = {key: jnp.asarray(archive[key]) for key in archive.files}
                evaluation = evaluate(params, range(30_000_000, 30_000_000 + episodes),
                                      architecture=architecture)
                (directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
                moving = evaluation["moving_object_moving_camera"]["model"]
                rows.append({"architecture": architecture, "curriculum": curriculum,
                             "seed": seed, "test_mse": moving["mse"],
                             "reappearance_location_mae_px": moving["reappearance_location_mae_px"],
                             "reappearance_count": moving["reappearance_count"]})
                (root / "runs.json").write_text(json.dumps(rows, indent=2) + "\n")
    (root / "summary.json").write_text(json.dumps(aggregate(rows), indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.getenv("SWEEP_OUTPUT", "runs/sweep"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[31, 37, 41, 43, 47])
    parser.add_argument("--architectures", nargs="+", choices=("recurrent", "feedforward", "multisheet"),
                        default=["recurrent", "feedforward", "multisheet"])
    parser.add_argument("--curricula", nargs="+", choices=("baseline", "staged", "shuffled"),
                        default=["staged", "shuffled"])
    parser.add_argument("--steps", type=int, default=int(os.getenv("STEPS", "120")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "8")))
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--require-gpu", action="store_true", default=os.getenv("REQUIRE_GPU", "0") == "1")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.episodes < 1:
        parser.error("Training steps, batch size and evaluation episodes must be positive")
    run(args.output, args.seeds, args.architectures, args.curricula,
        args.steps, args.batch_size, args.episodes, args.require_gpu)


if __name__ == "__main__":
    main()
