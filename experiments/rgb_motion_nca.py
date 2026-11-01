"""Step 5: test whether the RGB NCA learns simple ball motion.

The overfit arm trains on one rendered trajectory. The generalization arm uses
that same checkpoint on unseen trajectories from the same controllable world.
Only rendered frames enter training and inference; world truth is evaluation
metadata and is never passed to the NCA.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import baselines, predictive_rgb_nca as nca, rgb_motion_world
from train import adamw


def _examples(seeds, steps, size, context, horizon, balls=1):
    length = context + horizon
    if steps < length:
        raise ValueError("steps must cover context plus forecast horizon")
    episodes = [rgb_motion_world.episode(seed, steps=steps, size=size,
                                         balls=balls)
                for seed in seeds]
    return np.stack([item["frames"][:length] for item in episodes])


def _baseline_scores(context, truth):
    persistence = baselines.persistence_forecast(context, len(truth))
    velocity = baselines.constant_velocity_forecast(context, len(truth))
    return {
        "persistence_mse": float(np.mean(np.square(persistence - truth))),
        "constant_velocity_mse": float(np.mean(np.square(velocity - truth))),
    }


def evaluate(params, examples, context, horizon, ticks):
    predictions = np.asarray(nca.rollout(
        params, jnp.asarray(examples[:, :context]), horizon, ticks))
    truth = examples[:, context:context + horizon]
    rows = []
    for index in range(len(examples)):
        row = {"model_mse": float(np.mean(np.square(predictions[index] - truth[index])))}
        row.update(_baseline_scores(examples[index, :context], truth[index]))
        rows.append(row)
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in rows[0]
    } | {
        "per_episode": rows,
        "model_beats_persistence": bool(
            np.mean([row["model_mse"] for row in rows]) <
            np.mean([row["persistence_mse"] for row in rows])),
        "model_beats_constant_velocity": bool(
            np.mean([row["model_mse"] for row in rows]) <
            np.mean([row["constant_velocity_mse"] for row in rows])),
    }


def fit(seed=41, steps=50, training_steps=100, size=24, context=4,
        horizon=3, ticks=2, channels=8, generalization_seeds=(101, 102, 103),
        training_seeds=None,
        output="runs/rgb-motion-nca"):
    if min(steps, training_steps, size, context, horizon, ticks, channels) < 1:
        raise ValueError("All dimensions and training steps must be positive")
    if training_seeds is None:
        training_seeds = (seed, seed + 1, seed + 2, seed + 3, seed + 4)
    training_seeds = tuple(map(int, training_seeds))
    train = _examples(training_seeds, steps, size, context, horizon)
    heldout = _examples(generalization_seeds, steps, size, context, horizon)
    params = nca.init(jax.random.PRNGKey(seed), channels=channels)
    moments = jax.tree.map(jnp.zeros_like, params)
    variances = jax.tree.map(jnp.zeros_like, params)

    @jax.jit
    def update(p, m, v, batch, step):
        error, gradients = jax.value_and_grad(
            lambda weights: nca.loss(weights, batch, context, horizon, ticks))(p)
        norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(gradients)))
        gradients = jax.tree.map(lambda g: g / jnp.maximum(1.0, norm), gradients)
        p, m, v = adamw(p, gradients, m, v, step)
        return p, m, v, error

    history = []
    batch = jnp.asarray(train)
    for step in range(1, training_steps + 1):
        params, moments, variances, error = update(
            params, moments, variances, batch, step)
        if step == 1 or step == training_steps or step % max(1, training_steps // 5) == 0:
            history.append({"step": step, "training_loss": float(error)})

    overfit = evaluate(params, train, context, horizon, ticks)
    generalization = evaluate(params, heldout, context, horizon, ticks)
    result = {
        "experiment": "rgb_motion_nca_step5",
        "architecture": "predictive_rgb_nca",
        "training_seed": seed,
        "training_seeds": list(training_seeds),
        "generalization_seeds": list(map(int, generalization_seeds)),
        "world": {"size": size, "balls": 1, "occlusion": False,
                  "camera_motion": False},
        "context_frames": context, "forecast_horizon_frames": horizon,
        "nca_ticks": ticks, "channels": channels,
        "training_steps": training_steps,
        "training_history": history,
        "overfit": overfit,
        "generalization": generalization,
        "success_criteria": {
            "overfit_beats_persistence": overfit["model_beats_persistence"],
            "generalization_beats_persistence": generalization["model_beats_persistence"],
            "generalization_beats_constant_velocity": generalization[
                "model_beats_constant_velocity"],
        },
        "note": "Rendered RGB frames train the model; positions and velocities are evaluation-only.",
    }
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "parameters.npz",
                        **jax.tree.map(np.asarray, params))
    (target / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--generalization-seeds", type=int, nargs="+",
                        default=(101, 102, 103))
    parser.add_argument("--training-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--out", default="runs/rgb-motion-nca")
    args = parser.parse_args()
    print(json.dumps(fit(args.seed, args.steps, args.training_steps, args.size,
                         args.context, args.horizon, args.ticks, args.channels,
                         tuple(args.generalization_seeds), args.training_seeds,
                         args.out), indent=2))


if __name__ == "__main__":
    main()
