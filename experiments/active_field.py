"""Minimal embodied predictive field: deterministic gaze and local adaptation controls."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src import model, plastic_field


ACTIONS = (-1, 0, 1)


def reset(seed, size=16):
    rng = np.random.default_rng(seed)
    width = size + 6
    return {"size": size, "width": width, "x": int(rng.integers(3, width-3)),
            "y": int(rng.integers(3, size-3)), "direction": int(rng.choice((-1, 1))),
            "view": 3, "wall": width // 2}


def observe(world):
    size = world["size"]
    frame = np.zeros((size, size, 1), np.float32)
    wall = world["wall"] - world["view"]
    if 0 <= wall < size:
        frame[:, wall, 0] = 0.35
    x = world["x"] - world["view"]
    visible = 0 <= x < size and world["x"] != world["wall"]
    if visible:
        frame[world["y"], x, 0] = 1
    return frame, visible


def advance(world, action):
    """Physics uses hidden state; the sensing policy never receives it."""
    if action not in ACTIONS:
        raise ValueError("Invalid gaze action")
    next_world = dict(world)
    next_world["view"] = int(np.clip(world["view"] + action, 0, world["width"] - world["size"]))
    next_world["x"] += world["direction"]
    if next_world["x"] < 1 or next_world["x"] >= world["width"] - 1:
        next_world["direction"] *= -1
        next_world["x"] += 2 * next_world["direction"]
    return next_world


def gaze_action(params, state, observed):
    """Greedy predicted-pixel uncertainty; no future pixels or simulator truth."""
    scored = []
    for action in ACTIONS:
        _, y = plastic_field.step(params, state, jnp.asarray(observed)[None],
                                  jnp.asarray([[action]], jnp.float32), plasticity_rate=0.0)
        p = np.asarray(y[0])
        uncertainty = float(np.mean(p * (1 - p)))
        scored.append((uncertainty, -abs(action), action))
    return max(scored)[2]


def episode(params, seed, policy="fixed", plasticity_rate=0.0, steps=12):
    """One observation per step; reset hidden state for every independent episode."""
    if policy not in ("fixed", "random", "model"):
        raise ValueError("Unknown gaze policy")
    world = reset(seed)
    rng = np.random.default_rng(seed + 900)
    state = plastic_field.initial(world["size"])
    errors, object_visible = [], []
    observed, _ = observe(world)
    for _ in range(steps):
        action = (0 if policy == "fixed" else
                  int(rng.choice(ACTIONS)) if policy == "random" else
                  gaze_action(params, state, observed))
        state, forecast = plastic_field.step(
            params, state, jnp.asarray(observed)[None],
            jnp.asarray([[action]], jnp.float32), plasticity_rate=plasticity_rate)
        world = advance(world, action)
        actual, visible = observe(world)
        errors.append(float(np.mean((np.asarray(forecast[0]) - actual) ** 2)))
        object_visible.append(bool(visible))
        observed = actual
    return {"prediction_mse": float(np.mean(errors)),
            "visible_fraction": float(np.mean(object_visible)),
            "final_gain_l1": float(jnp.mean(jnp.abs(state[1])))}


def compare(params, seeds, steps=12, plasticity_rate=0.05):
    result = {}
    for policy in ("fixed", "random", "model"):
        for label, rate in (("fixed", 0.0), ("plastic", plasticity_rate)):
            outcomes = [episode(params, int(seed), policy, rate, steps) for seed in seeds]
            result[f"{policy}:{label}"] = {
                "episodes": len(outcomes),
                "observation_budget": steps,
                "mean_prediction_mse": float(np.mean([r["prediction_mse"] for r in outcomes])),
                "mean_visible_fraction": float(np.mean([r["visible_fraction"] for r in outcomes])),
                "mean_final_gain_l1": float(np.mean([r["final_gain_l1"] for r in outcomes])),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", required=True, help="Trained recurrent parameters.npz")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--plasticity-rate", type=float, default=0.05)
    parser.add_argument("--out", default="runs/active-plasticity.json")
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1 or not 0 <= args.plasticity_rate <= 0.1:
        parser.error("Positive episodes/steps and plasticity_rate in [0,0.1] required")
    with np.load(args.parameters, allow_pickle=False) as archive:
        params = {key: jnp.asarray(archive[key]) for key in archive.files}
    result = compare(params, range(80_000_000, 80_000_000 + args.episodes),
                     args.steps, args.plasticity_rate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
