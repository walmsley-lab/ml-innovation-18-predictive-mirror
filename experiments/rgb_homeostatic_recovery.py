"""Matched-seed RGB-NCA local recovery, shuffled recovery, and plain control."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import predictive_rgb_nca as base, rgb_benchmark as benchmark
from src import rgb_homeostatic_nca as model
from train import adamw


def run(out="runs/rgb-homeostatic-recovery", seeds=(11, 23, 41),
        steps=60, context=8, horizon=10, size=32, ticks=2, channels=12,
        batch_size=2, strength=.15, decay=.9):
    if (not seeds or steps < 1 or context < 2 or horizon < 1
            or ticks < 1 or channels < 1 or batch_size < 1
            or not 0 <= strength <= 1 or not 0 <= decay < 1):
        raise ValueError("Invalid experiment settings")
    # Different scene IDs for training, model selection and final test.
    train = benchmark.frames(benchmark.scenes(
        31000, 12, context, horizon, size))
    validation = benchmark.frames(benchmark.scenes(
        61000, 4, context, horizon, size))
    test = benchmark.frames(benchmark.scenes(
        71000, 4, context, horizon, size))
    leads = tuple(sorted({horizon, *(k for k in benchmark.LEADS if k <= horizon)}))
    baselines = {name: benchmark.score(value, test, context, leads)
                 for name, value in benchmark.controls(
                     test[:, :context], horizon).items()}

    def validation_mse(params, mode):
        predicted = np.asarray(model.rollout(
            params, jnp.asarray(validation[:, :context]),
            horizon, ticks, mode, strength, decay))
        return float(np.square(predicted - validation[:, context:]).mean())

    def make_update(mode):
        @jax.jit
        def update(params, m, v, examples, step):
            error, grad = jax.value_and_grad(
                lambda p: model.loss(p, examples, context, horizon, ticks,
                                     mode, strength, decay))(params)
            norm = jnp.sqrt(sum(jnp.square(g).sum()
                                for g in jax.tree.leaves(grad)))
            grad = jax.tree.map(lambda g: g / jnp.maximum(norm, 1), grad)
            params, m, v = adamw(params, grad, m, v, step)
            return params, m, v, error
        return update

    updates = {mode: make_update(mode) for mode in model.MODES}
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        initial = base.init(jax.random.PRNGKey(int(seed)), channels)
        zeros = lambda: jax.tree.map(jnp.zeros_like, initial)
        arms = {}
        for mode in model.MODES:
            val = validation_mse(initial, mode)
            arms[mode] = {"params": initial, "m": zeros(), "v": zeros(),
                          "best": initial, "best_step": 0, "val": val,
                          "step_zero_val": val}
        rng = np.random.default_rng(int(seed) + 1000)
        for step in range(1, steps + 1):
            # Exactly the same examples, order, steps and base initialization.
            examples = jnp.asarray(
                train[rng.integers(len(train), size=batch_size)])
            for mode, arm in arms.items():
                arm["params"], arm["m"], arm["v"], _ = updates[mode](
                    arm["params"], arm["m"], arm["v"], examples, step)
            if step == steps or step % max(1, steps // 5) == 0:
                for mode, arm in arms.items():
                    val = validation_mse(arm["params"], mode)
                    if val < arm["val"]:
                        arm["best"], arm["val"], arm["best_step"] = (
                            arm["params"], val, step)
        seed_results = {"seed": int(seed), "arms": {}}
        for mode, arm in arms.items():
            predictions, activity = model.rollout_and_diagnostics(
                arm["best"], jnp.asarray(test[:, :context]), horizon,
                ticks, mode, strength, decay)
            score = benchmark.score(np.asarray(predictions), test,
                                    context, leads)
            seed_results["arms"][mode] = {
                "best_step": int(arm["best_step"]),
                "validation_mse": arm["val"],
                "step_zero_validation_mse": arm["step_zero_val"],
                "test": score,
                "activity": {key: float(value) for key, value in activity.items()},
            }
            np.savez_compressed(output / f"seed-{seed}-{mode}.npz",
                                **jax.tree.map(np.asarray, arm["best"]))
        results.append(seed_results)
    base_params = base.init(jax.random.PRNGKey(0), channels)
    report = {
        "experiment": "rgb_nca_homeostatic_recovery_ablation",
        "seeds": list(map(int, seeds)), "steps": steps,
        "batch_size": batch_size, "context": context, "horizon": horizon,
        "ticks": ticks, "size": size, "channels": channels,
        "regulation_strength": strength, "decay": decay,
        "train_scene_start": 31000, "validation_scene_start": 61000,
        "test_scene_start": 71000,
        "trainable_parameters_per_arm": int(sum(
            np.size(np.asarray(p)) for p in jax.tree.leaves(base_params))),
        "additional_recovery_scalars_per_episode_per_arm": {
            "plain": 0, "local_recovery": size * size * channels,
            "shuffled_recovery": size * size * channels},
        "controls": baselines, "results": results,
        "limits": "Extra fixed-size recurrent recovery state and compute for "
                  "recovery arms, but no new trainable parameters. This is a "
                  "toy activity-dependent negative feedback, not real chemical "
                  "homeostasis. Report test performance vs constant velocity; "
                  "higher internal stability alone is not predictive value. "
                  "No direct harmonic forcing or phase oscillator.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/rgb-homeostatic-recovery")
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 41])
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--ticks", type=int, default=2)
    p.add_argument("--channels", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--strength", type=float, default=.15)
    p.add_argument("--decay", type=float, default=.9)
    a = p.parse_args()
    r = run(a.out, a.seeds, a.steps, a.context, a.horizon, a.size,
            a.ticks, a.channels, a.batch_size, a.strength, a.decay)
    print(json.dumps({"seeds": r["seeds"], "arms": list(model.MODES)}, indent=2))


if __name__ == "__main__":
    main()
