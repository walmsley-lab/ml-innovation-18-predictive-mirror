"""Evaluate identical held-out tasks at each developmental learning milestone."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src import model, feedforward, multisheet, multisheet_no_feedback
from src.world import curriculum_batch


MODELS = {
    "recurrent": model, "feedforward": feedforward,
    "multisheet": multisheet, "multisheet_no_feedback": multisheet_no_feedback,
}


def read_checkpoint(path):
    with np.load(path, allow_pickle=False) as file:
        return {name: jnp.asarray(file[name]) for name in file.files}


def score_snapshot(params, architecture, seeds, stage):
    """Observed pixels only, independent episode seeds for every stage."""
    frames, actions = curriculum_batch(seeds, [stage] * len(seeds))
    y = np.stack([np.asarray(MODELS[architecture].predict(params, frame, action))
                  for frame, action in zip(frames, actions)])
    target = frames[:, 1:]
    positive = target > 0.9
    return {
        "mse": float(np.mean((y - target) ** 2)),
        "visible_object_mse": float(np.mean(((y - target) ** 2)[positive])) if positive.any() else None,
        "visible_object_pixels": int(positive.sum()),
    }


def evaluate_snapshots(root, architecture, seeds):
    snapshots = sorted(Path(root).glob("snapshots/step-*.npz"))
    if not snapshots:
        raise FileNotFoundError(f"No milestone snapshots: {root}; train with --save-milestones")
    table = {}
    for path in snapshots:
        params = read_checkpoint(path)
        table[path.stem] = {str(stage): score_snapshot(params, architecture, seeds, stage)
                            for stage in range(4)}
    return table


def compare(ordered, shuffled, architecture, seeds):
    """Results are diagnostic learning curves, not a test of transfer causation."""
    seeds = tuple(seeds)
    first = evaluate_snapshots(ordered, architecture, seeds)
    second = evaluate_snapshots(shuffled, architecture, seeds)
    if set(first) != set(second):
        raise ValueError("Snapshot steps must match across curricula")
    return {"staged": first, "shuffled": second, "test_seeds": list(seeds),
            "note": "Descriptive comparison; matched exposure and compute required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", required=True)
    parser.add_argument("--shuffled", required=True)
    parser.add_argument("--architecture", choices=tuple(MODELS), default="recurrent")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--out", default="runs/developmental-transfer.json")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    result = compare(args.staged, args.shuffled, args.architecture,
                     range(70_000_000, 70_000_000 + args.episodes))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
