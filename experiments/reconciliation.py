"""Post-hoc probes for whether distinct neural sheets jointly encode hidden motion."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.multisheet import trace
from src.world import sequence


def observations(params, seeds, feedback_strength=1.0):
    """Probe input is frozen internal activity; targets are evaluation-only truth."""
    fast_features, slow_features, targets, hidden_checks = [], [], [], []
    for seed in seeds:
        frames, actions, hidden = sequence(int(seed), steps=12, camera=True, occlusion=True)
        _, fast, slow = trace(params, frames, actions, feedback_strength)
        fast_features.extend(np.asarray(fast).mean(axis=1).reshape(12, -1))
        slow_features.extend(np.asarray(slow).mean(axis=1).reshape(12, -1))
        # Target is next-frame retinal x: a diagnostic predictive readout.
        targets.extend(hidden["world_x"][1:] - hidden["camera_x"][1:])
        hidden_checks.extend(~hidden["visible_mask"][1:].any(axis=(1, 2, 3)))
    return tuple(map(np.asarray, (fast_features, slow_features, targets, hidden_checks)))


def fit_probe(features, target, ridge=0.1):
    """Small linear ridge decoder, trained only after the perceptual network freezes."""
    inputs = np.column_stack((np.ones(len(features)), features.astype(np.float64)))
    gram = inputs.T @ inputs
    penalty = ridge * np.eye(gram.shape[0])
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, inputs.T @ target)


def decode(features, weights):
    return np.column_stack((np.ones(len(features)), features)) @ weights


def test_reconciliation(params, train_seeds, test_seeds):
    train = observations(params, train_seeds)
    test = observations(params, test_seeds)
    wa, wb = fit_probe(train[0], train[2]), fit_probe(train[1], train[2])
    a, b = decode(test[0], wa), decode(test[1], wb)
    truth, occluded = test[2], test[3]
    jointly_correct = (np.abs(a - truth) <= 1) & (np.abs(b - truth) <= 1)
    compatible = np.abs(a - b) <= 1
    outcomes = {
        "n_occluded": int(occluded.sum()),
        "fast_mae": float(np.mean(np.abs(a - truth))),
        "slow_mae": float(np.mean(np.abs(b - truth))),
        "compatible_fraction": float(np.mean(compatible[occluded])) if occluded.any() else None,
        "accurate_and_compatible_fraction": float(np.mean((compatible & jointly_correct)[occluded])) if occluded.any() else None,
    }
    # Retain intact probes: feedback-cut scores measure immediate representational disruption.
    cut = observations(params, test_seeds, feedback_strength=0.0)
    ca, cb = decode(cut[0], wa), decode(cut[1], wb)
    both = (np.abs(ca - truth) <= 1) & (np.abs(cb - truth) <= 1)
    outcomes["feedback_cut_accurate_and_compatible_fraction"] = (
        float(np.mean(((np.abs(ca - cb) <= 1) & both)[occluded])) if occluded.any() else None)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", default="runs/multisheet/parameters.npz")
    parser.add_argument("--output", default="runs/reconciliation.json")
    parser.add_argument("--train-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=16)
    args = parser.parse_args()
    with np.load(args.parameters, allow_pickle=False) as archive:
        params = {name: jnp.asarray(archive[name]) for name in archive.files}
    result = test_reconciliation(params, range(40_000_000, 40_000_000 + args.train_episodes),
                                 range(50_000_000, 50_000_000 + args.test_episodes))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
