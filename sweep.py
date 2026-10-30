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
        key = (row["architecture"], row["curriculum"], row.get("loss_mode", "plain"))
        grouped.setdefault(key, []).append(row)
    result = {}
    for (architecture, curriculum, loss_mode), values in grouped.items():
        scores = [v["reappearance_location_mae_px"] for v in values
                  if v["reappearance_location_mae_px"] is not None]
        result[f"{architecture}:{curriculum}:{loss_mode}"] = {
            "seeds": [v["seed"] for v in values],
            "n": len(values),
            "moving_camera_reappearance_mae_mean_px": float(np.mean(scores)) if scores else None,
            "moving_camera_reappearance_mae_std_px": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
            "note": "descriptive only; no claim of significance",
        }
    return result


def write_json(path, value):
    """Publish a complete JSON artifact via same-filesystem atomic rename."""
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(output, seeds, architectures, curricula, steps, batch_size, episodes,
        require_gpu, loss_mode="plain", resume=False, checkpoint_every=25):
    """Train/evaluate each condition on untouched deterministic holdout episodes."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for architecture in architectures:
        for curriculum in curricula:
            for seed in seeds:
                name = f"{architecture}-{curriculum}-seed{seed}"
                directory = root / name
                specification = {
                    "schema": 1,
                    "seed": seed,
                    "architecture": architecture,
                    "curriculum": curriculum,
                    "loss_mode": loss_mode,
                    "updates": steps,
                    "batch_size": batch_size,
                    "episodes": episodes,
                    "checkpoint_every": checkpoint_every,
                    "replay_fraction": 0.10,
                    "source_revision": os.getenv("SOURCE_REVISION", "unknown"),
                }
                manifest = directory / "spec.json"
                if manifest.exists():
                    if json.loads(manifest.read_text()) != specification:
                        raise ValueError(f"Run configuration changed: {manifest}")
                    if not resume:
                        raise FileExistsError(f"Existing run: {directory}; pass --resume")
                else:
                    if directory.exists() and any(directory.iterdir()):
                        raise FileExistsError(f"Untracked run output: {directory}")
                    directory.mkdir(parents=True, exist_ok=True)
                    write_json(manifest, specification)

                metrics_file = directory / "metrics.json"
                weights_file = directory / "parameters.npz"
                checkpoint_file = directory / "checkpoint.npz"
                if not (metrics_file.exists() and weights_file.exists()):
                    train(seed=seed, steps=steps, batch_size=batch_size, output=directory,
                          checkpoint_every=checkpoint_every,
                          resume=resume and checkpoint_file.exists(),
                          require_gpu=require_gpu, curriculum=curriculum,
                          architecture=architecture, loss_mode=loss_mode)

                evaluation_file = directory / "evaluation.json"
                if evaluation_file.exists() and resume:
                    evaluation = json.loads(evaluation_file.read_text())
                else:
                    with np.load(weights_file, allow_pickle=False) as archive:
                        params = {key: jnp.asarray(archive[key]) for key in archive.files}
                    evaluation = evaluate(params, range(30_000_000, 30_000_000 + episodes),
                                          architecture=architecture)
                    write_json(evaluation_file, evaluation)

                moving = evaluation["moving_object_moving_camera"]["model"]
                rows.append({"architecture": architecture, "curriculum": curriculum,
                             "loss_mode": loss_mode, "seed": seed,
                             "test_mse": moving["mse"],
                             "reappearance_location_mae_px": moving["reappearance_location_mae_px"],
                             "reappearance_count": moving["reappearance_count"]})
                write_json(root / "runs.json", rows)
    write_json(root / "summary.json", aggregate(rows))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.getenv("SWEEP_OUTPUT", "runs/sweep"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[31, 37, 41, 43, 47])
    parser.add_argument("--architectures", nargs="+", choices=("recurrent", "feedforward", "multisheet"),
                        default=["recurrent", "feedforward", "multisheet"])
    parser.add_argument("--curricula", nargs="+", choices=("baseline", "staged", "shuffled"),
                        default=["staged", "shuffled"])
    parser.add_argument("--loss-mode", choices=("plain", "contrast"), default=os.getenv("LOSS_MODE", "plain"))
    parser.add_argument("--resume", action="store_true", default=os.getenv("RESUME", "0") == "1")
    parser.add_argument("--checkpoint-every", type=int, default=int(os.getenv("CHECKPOINT_EVERY", "25")))
    parser.add_argument("--steps", type=int, default=int(os.getenv("STEPS", "120")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "8")))
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--require-gpu", action="store_true", default=os.getenv("REQUIRE_GPU", "0") == "1")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.episodes < 1 or args.checkpoint_every < 1:
        parser.error("Training steps, batch size and evaluation episodes must be positive")
    run(args.output, args.seeds, args.architectures, args.curricula,
        args.steps, args.batch_size, args.episodes, args.require_gpu, args.loss_mode,
        args.resume, args.checkpoint_every)


if __name__ == "__main__":
    main()
