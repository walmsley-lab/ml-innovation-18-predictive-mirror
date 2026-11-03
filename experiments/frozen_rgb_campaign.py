"""Paired same-world NCA/CNN/transport training and frozen evaluation.

A CPU smoke validates mechanics; substantial multi-seed runs are separate
research work. No results are silently promoted to the README.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import predictive_rgb_nca as nca
from src import rgb_motion_transport as transport
from src import tiny_cnn_video as cnn
from src import rgb_benchmark as bench
from train import adamw

MODELS = ("nca", "cnn", "transport")


def specification(name, key, width, context, horizon):
    if name == "nca":
        return (nca.init(key, width),
                lambda p, x: nca.loss(p, x, context, horizon, 2),
                lambda p, x: nca.rollout(p, x, horizon, 2))
    if name == "cnn":
        return (cnn.init(key, width),
                lambda p, x: cnn.loss(p, x, context, horizon),
                lambda p, x: cnn.rollout(p, x, horizon))
    if name == "transport":
        return (transport.init(key, width),
                lambda p, x: transport.loss(p, x, context, horizon, "combined"),
                lambda p, x: transport.rollout(p, x, horizon, "combined"))
    raise ValueError("Unknown model")


def paired_interval(a, b, truth, lead, seed=90, draws=400):
    """Resample *whole episodes*, not dependent pixels."""
    left = np.square(a[:, lead-1] - truth[:, lead-1]).mean((1, 2, 3))
    right = np.square(b[:, lead-1] - truth[:, lead-1]).mean((1, 2, 3))
    delta = left-right
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(delta), (draws, len(delta)))
    low, high = np.quantile(delta[choices].mean(axis=1), [.025, .975])
    return {"mean_delta_mse": float(delta.mean()),
            "episode_bootstrap_95pct": [float(low), float(high)]}


def run(out="runs/frozen-rgb-campaign", seeds=(11, 23, 41), steps=100,
        context=8, horizon=10, size=32, width=8, batch_size=2,
        train_count=48, val_count=12, test_count=24):
    if (not seeds or steps < 1 or context < 2 or horizon < 1
            or min(width, batch_size, train_count, val_count, test_count) < 1):
        raise ValueError("Invalid experiment budget")
    # World seeds are disjoint and identical for all trained architectures.
    world = {"balls": 2, "occlusion": True, "camera_motion": False}
    train = bench.frames(bench.scenes(31000, train_count, context, horizon, size, **world))
    val = bench.frames(bench.scenes(61000, val_count, context, horizon, size, **world))
    test = bench.frames(bench.scenes(71000, test_count, context, horizon, size, **world))
    context_test = jnp.asarray(test[:, :context])
    control = bench.controls(test[:, :context], horizon)
    leads = tuple(sorted({horizon, *(h for h in bench.LEADS if h <= horizon)}))
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed) + 900)
        sample_ids = rng.integers(len(train), size=(steps, batch_size))
        cases = {}
        for name in MODELS:
            initial, objective, predictor = specification(
                name, jax.random.PRNGKey(int(seed)), width, context, horizon)
            params = best = initial
            m = jax.tree.map(jnp.zeros_like, initial)
            v = jax.tree.map(jnp.zeros_like, initial)
            def evaluate(p):
                pred = np.asarray(predictor(p, jnp.asarray(val[:, :context])))
                return float(np.square(pred-val[:, context:]).mean())
            best_val, selected = evaluate(initial), 0
            @jax.jit
            def step(p, mm, vv, batch, index):
                value, grad = jax.value_and_grad(objective)(p, batch)
                norm = jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grad)))
                grad = jax.tree.map(lambda g: g/jnp.maximum(norm, 1.), grad)
                p, mm, vv = adamw(p, grad, mm, vv, index)
                return p, mm, vv, value
            start = time.perf_counter()
            for index, ids in enumerate(sample_ids, 1):
                params, m, v, _ = step(params, m, v, jnp.asarray(train[ids]), index)
                if index == steps or index % max(1, steps//5) == 0:
                    value = evaluate(params)
                    if value < best_val:
                        best, best_val, selected = params, value, index
            # Explicitly synchronize device work for the elapsed time.
            jax.block_until_ready(jax.tree.leaves(params)[0])
            elapsed = time.perf_counter() - start
            prediction = np.asarray(predictor(best, context_test))
            cases[name] = {
                "selected_step": selected, "validation_mse": best_val,
                "train_seconds_including_jit": elapsed,
                "parameters": int(sum(x.size for x in jax.tree.leaves(best))),
                "test": bench.score(prediction, test, context, leads),
                "paired_vs_velocity": {
                    str(h): paired_interval(prediction, control["constant_velocity"],
                                            test[:, context:], h, int(seed))
                    for h in leads
                },
            }
            np.savez_compressed(folder/f"seed-{seed}-{name}.npz",
                                **jax.tree.map(np.asarray, best))
        runs.append({"seed": int(seed), "models": cases})
    report = {
        "protocol": "frozen-rgb-campaign-v1",
        "main_metric": "mse, lower is better", "model_seeds": list(map(int, seeds)),
        "steps_per_model": steps, "batch_size": batch_size,
        "context": context, "horizon": horizon, "width": width,
        "world": world, "train_seed_root": 31000, "val_seed_root": 61000,
        "test_seed_root": 71000, "episodes": {
            "train": train_count, "validation": val_count, "test": test_count},
        "baselines": {name: bench.score(pred, test, context, leads)
                      for name, pred in control.items()},
        "runs": runs,
        "limitations": "Identical train episodes and SGD update counts, not equal "
            "FLOPs, recurrent unroll cost or losses. Bootstrap intervals resample "
            "test episodes conditional on the trained seed; three-seed replication "
            "and independent webcam validation remain essential."
    }
    (folder/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/frozen-rgb-campaign")
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 41])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    a = p.parse_args()
    result = run(a.out, a.seeds, a.steps, a.context, a.horizon,
                 a.size, a.width, a.batch_size)
    print(json.dumps({"protocol": result["protocol"], "runs": len(result["runs"])}, indent=2))


if __name__ == "__main__":
    main()
