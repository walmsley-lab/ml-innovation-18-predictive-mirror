"""Tiny-sequence NCA learnability gate: overfit before scaling the world.

This is intentionally a capability-training experiment, not a benchmark of
generalization or a mechanism for hiding a failed model behind more metrics.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import rgb_grounded_nca as model
from src import rgb_motion_distribution as world
from src import experiment_harness as harness


def evaluate(params, sample, context, horizon, ticks, output):
    forecast = np.asarray(model.rollout(
        params, jnp.asarray(sample[:, :context]), horizon, ticks, output))
    truth = sample[:, context:context + horizon]
    mse = float(np.mean((forecast - truth) ** 2))
    return {
        "mse": mse,
        "forecast_spread": float(np.std(forecast)),
        "clip_rail_fraction": float(np.mean((forecast <= 1e-6) | (forecast >= 1 - 1e-6))),
        "last_frame_mse": float(np.mean((np.repeat(
            sample[:, context - 1:context], horizon, axis=1) - truth) ** 2))
    }


def run(out="runs/rgb-learnability", seed=11, steps=100,
        context=8, horizon=5, size=32, channels=8, ticks=2,
        output="blend", foreground_weight=6.0):
    if (steps < 1 or context < 2 or horizon < 1 or ticks < 1
            or size < 16 or channels < 1 or output not in model.OUTPUT_MODES
            or foreground_weight < 0):
        raise ValueError("Invalid learnability configuration")
    frames, world_type = world.episode(seed + 31000, frames=context+horizon, size=size)
    # Exactly ONE repeated training trajectory: this is an overfitting test,
    # not disjoint-scene validation or a result about generalization.
    sample = frames[None]
    key = jax.random.PRNGKey(seed)
    params = model.init(key, channels=channels, output=output)
    moments = jax.tree.map(jnp.zeros_like, params)
    variance = jax.tree.map(jnp.zeros_like, params)
    update = harness.clipped_adamw_update(
        lambda p, batch: model.loss(
            p, batch, context, horizon, ticks, output, foreground_weight))
    milestones = sorted({0, 1, steps, *[int(steps * n / 4) for n in (1,2,3)]})
    history = []
    best = (float("inf"), 0, params)
    for step in range(steps + 1):
        if step:
            params, moments, variance, train_loss = update(
                params, moments, variance, (jnp.asarray(sample),), step)
            if not np.isfinite(float(train_loss)):
                raise FloatingPointError("Non-finite loss at step " + str(step))
        if step in milestones:
            quality = evaluate(params, sample, context, horizon, ticks, output)
            quality["step"] = step
            history.append(quality)
            if quality["mse"] < best[0]:
                best = (quality["mse"], step, params)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "best-parameters.npz"
    np.savez_compressed(path, **jax.tree.map(np.asarray, best[2]))
    result = {
        "purpose": "one-trajectory in-sample learnability (no test/generalization claim)",
        "seed": seed, "steps": steps, "context": context, "horizon": horizon,
        "model": "rgb_grounded_nca", "output": output,
        "foreground_weight": foreground_weight, "world_type": world_type,
        "best_step": best[1], "best_mse": best[0],
        "baseline_mse": history[0]["last_frame_mse"],
        "in_sample_mse_better_than_persistence": bool(best[0] < history[0]["last_frame_mse"]),
        "learning_curve": history,
        "checkpoint": path.name
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/rgb-learnability")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--ticks", type=int, default=2)
    p.add_argument("--output", choices=model.OUTPUT_MODES, default="blend")
    p.add_argument("--foreground-weight", type=float, default=6.0)
    a = p.parse_args()
    r = run(a.out,a.seed,a.steps,a.context,a.horizon,a.size,a.channels,
            a.ticks,a.output,a.foreground_weight)
    print(json.dumps({key:r[key] for key in (
        "best_step","best_mse","baseline_mse","in_sample_mse_better_than_persistence")}, indent=2))


if __name__ == "__main__":
    main()
