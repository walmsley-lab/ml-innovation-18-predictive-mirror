"""Minimal local neural plasticity and homeostasis experiment (H4 toy)."""

import argparse
import json
from pathlib import Path

import numpy as np


RULES = ("fixed", "hebbian", "homeostatic")


def update(activity, weights, stimulus, rule="fixed", rate=0.04, decay=0.025):
    """Locally connected recurrent cells; plasticity touches neighbor weights only."""
    if rule not in RULES:
        raise ValueError(f"Unknown rule: {rule}")
    neighbors = np.stack([np.roll(activity, 1), np.roll(activity, -1)], axis=-1)
    next_activity = np.tanh(0.25 * activity + np.sum(weights * neighbors, axis=-1) + stimulus)
    if rule == "fixed":
        next_weights = weights.copy()
    else:
        local_correlation = next_activity[:, None] * neighbors
        next_weights = weights + rate * local_correlation
        if rule == "homeostatic":
            next_weights -= decay * weights
    return next_activity, next_weights


def simulate(seed=42, cells=16, steps=150, rule="fixed"):
    """The same controlled sensory-remapping sequence for each update rule."""
    rng = np.random.default_rng(seed)
    activity = rng.uniform(0.2, 0.5, cells)
    weights = np.full((cells, 2), 0.15)
    weight_norms, active_fraction = [], []
    for t in range(steps):
        stimulus = np.zeros(cells)
        loc = t % cells if t < steps // 2 else (steps - t - 1) % cells
        stimulus[loc] = 0.65
        activity, weights = update(activity, weights, stimulus, rule)
        weight_norms.append(float(np.max(np.abs(weights))))
        active_fraction.append(float(np.mean(np.abs(activity) > 0.95)))
    return {"rule": rule, "seed": seed, "steps": steps,
            "max_weight_abs": weight_norms[-1],
            "max_weight_abs_trajectory": weight_norms,
            "fraction_near_saturation": float(np.mean(active_fraction[-20:]))}


def compare(seeds=(1, 2, 3), steps=150):
    results = {}
    for rule in RULES:
        runs = [simulate(seed, steps=steps, rule=rule) for seed in seeds]
        results[rule] = {
            "mean_final_max_weight": float(np.mean([r["max_weight_abs"] for r in runs])),
            "mean_near_saturation": float(np.mean([r["fraction_near_saturation"] for r in runs])),
            "runs": runs,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/plasticity.json")
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()
    result = compare(steps=args.steps)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print({k: v["mean_final_max_weight"] for k, v in result.items()})


if __name__ == "__main__":
    main()
