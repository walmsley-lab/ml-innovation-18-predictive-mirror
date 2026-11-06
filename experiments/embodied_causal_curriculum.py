"""Paired training curriculum: spring -> active blind palpation -> held-out tests.

The model sees symbolic motor programs, known movement axes, present sensors,
and modality masks. Future sensors, interpreted motor values, body state, and
shape labels are supervision/evaluation only. No paid deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import embodied_arithmetic_world as world
from src import embodied_arithmetic_model as model
from train import adamw

ARMS = ("paired", "desynchronized_touch")
STRESSES = ("available", "no_touch", "no_proprio", "no_vibration",
            "no_vision", "touch_time_shifted")
TENSOR_KEYS = ("observed", "tree", "axes", "stage", "target",
               "shape", "motor_label")


def prepare(data, ids, rng, mode, training=True, stress="available"):
    """Make one causal batch; time-shifted sensory input is a negative control."""
    if mode not in ARMS or stress not in STRESSES:
        raise ValueError("Unknown mode or sensor stress")
    values = {key: np.array(data[key][ids], copy=True) for key in TENSOR_KEYS}
    blind = data["blind"][ids]
    b, timesteps = values["observed"].shape[:2]
    mask = np.ones((b, timesteps, 4), np.float32)
    mask[..., 0] = (~blind)[:, None]
    if training:
        dropout = rng.random((b, timesteps, 3)) < .10
        mask[..., 1:] = (~dropout).astype(np.float32)
    visual = values["observed"].shape[-1] - 4
    if (mode == "desynchronized_touch" and training) or \
            stress == "touch_time_shifted":
        for row in range(b):
            shift = int(rng.integers(1, timesteps))
            values["observed"][row, :, visual+2:] = np.roll(
                values["observed"][row, :, visual+2:], shift, axis=0)
    if stress == "no_touch":
        mask[..., 2] = 0.
    elif stress == "no_proprio":
        mask[..., 1] = 0.
    elif stress == "no_vibration":
        mask[..., 3] = 0.
    elif stress == "no_vision":
        mask[..., 0] = 0.
    return (values["observed"], values["tree"], values["axes"],
            values["stage"], mask, values["target"], values["shape"],
            values["motor_label"])


def _tensor(batch):
    return tuple(jnp.asarray(item) for item in batch)


def _unvisited(data, size):
    """Evaluator-only mask for shape pixels not touched by the trajectory."""
    proprio = data["observed"][..., size*size*3:size*size*3+2]
    pixel = (proprio + 1.) * .5 * (size - 1)
    y, x = np.mgrid[:size, :size]
    visited = np.zeros((len(pixel), size, size), bool)
    for t in range(pixel.shape[1]):
        visited |= ((x[None] - pixel[:, t, 0, None, None])**2 +
                    (y[None] - pixel[:, t, 1, None, None])**2 < 1.5**2)
    return ~visited


def evaluate(params, data, size, prior, mode="paired", stress="available"):
    batch = prepare(data, np.arange(len(data["observed"])),
                    np.random.default_rng(14713), mode, False, stress)
    forecast, shape, commands = model.rollout(
        params, *_tensor(batch[:5]), size)
    forecast, shape, commands = map(np.asarray, (forecast, shape, commands))
    future = batch[5]
    v = size * size * 3
    mse = lambda a, b: float(np.mean(np.square(a - b)))
    result = {
        "vision_mse": mse(forecast[..., :v], future[..., :v]),
        "proprio_mse": mse(forecast[..., v:v+2], future[..., v:v+2]),
        "touch_mse": mse(forecast[..., v+2], future[..., v+2]),
        "vibration_mse": mse(forecast[..., v+3], future[..., v+3]),
        "command_semantics_mae": float(np.abs(commands - batch[7]).mean()),
        "observed_vision_fraction": float(batch[4][..., 0].mean())
    }
    if data["stage"][0, 0, 1] == 1:
        unvisited = _unvisited(data, size)
        truth = data["shape"][:, -1]
        predicted = shape[:, -1]
        if not np.any(unvisited):
            raise AssertionError("No unprobed contour cells remain")
        result["unprobed_shape_mse"] = float(
            np.square(predicted - truth)[unvisited].mean())
        result["unprobed_shape_prior_mse"] = float(
            np.square(np.broadcast_to(prior, truth.shape) - truth)[unvisited].mean())
    return result


def run(out="runs/embodied-causal", seeds=(11, 23, 41),
        spring_steps=40, touch_steps=80, steps_per_episode=24,
        size=8, width=24, batch_size=4, train_count=48,
        val_count=12, test_count=16):
    if (not seeds or len(seeds) != len(set(seeds)) or
            min(spring_steps, touch_steps, batch_size, train_count,
                val_count, test_count) < 1 or
            steps_per_episode < 4 or size < 8 or width < 4):
        raise ValueError("Invalid curriculum budget")
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)

    def data(start, count, stage, grammar="small", stiffness="usual",
             blind="mixed"):
        return world.dataset(start, count, stage, steps_per_episode,
                             size, grammar, stiffness, blind)

    train_spring = data(31000, train_count, "spring", blind="no")
    train_touch = data(32000, train_count, "touch", blind="mixed")
    val_spring = data(61000, val_count, "spring", blind="no")
    val_touch = data(62000, val_count, "touch", blind="mixed")
    test = {
        "spring_familiar": data(91000, test_count, "spring", blind="no"),
        "spring_composed": data(92000, test_count, "spring",
                                grammar="composed", blind="no"),
        "spring_stiff": data(93000, test_count, "spring",
                             stiffness="stiff", blind="no"),
        "touch_sighted": data(94000, test_count, "touch", blind="no"),
        "touch_blind": data(95000, test_count, "touch", blind="yes"),
        "touch_blind_composed": data(96000, test_count, "touch",
                                      grammar="composed", blind="yes"),
    }
    prior = train_touch["shape"][:, 0].mean(axis=0)
    results = []

    @jax.jit
    def update(p, m, v, inputs, index):
        observed, tree, axes, stage, mask, target, shape, motor = inputs
        objective = lambda q: model.loss(
            q, observed, tree, axes, stage, mask, target, shape, motor, size)
        value, gradients = jax.value_and_grad(objective)(p)
        gradient_norm = jnp.sqrt(sum(jnp.sum(g*g) for g in
                                     jax.tree.leaves(gradients)))
        gradients = jax.tree.map(
            lambda g: g / jnp.maximum(1., gradient_norm), gradients)
        p, m, v = adamw(p, gradients, m, v, index, rate=.0015)
        return p, m, v, value

    def validation_loss(p):
        parts = []
        for dataset in (val_spring, val_touch):
            batch = prepare(dataset, np.arange(len(dataset["observed"])),
                            np.random.default_rng(17), "paired", False)
            parts.append(float(model.loss(p, *_tensor(batch), size)))
        return .3 * parts[0] + .7 * parts[1]

    for seed in seeds:
        rng = np.random.default_rng(int(seed) + 7000)
        # The exact batch indices/dropout are identical for both arms.
        curriculum = []
        for step in range(spring_steps + touch_steps):
            if step < spring_steps:
                source = train_spring
            else:
                source = train_touch if rng.random() < .7 else train_spring
            ids = rng.integers(0, train_count, batch_size)
            curriculum.append((source, ids, int(rng.integers(1, 2**31))))
        initial = model.init(jax.random.PRNGKey(int(seed)), size, width)
        arms = {name: {"p": initial, "m": jax.tree.map(jnp.zeros_like, initial),
                       "v": jax.tree.map(jnp.zeros_like, initial),
                       "best": initial, "best_score": validation_loss(initial),
                       "best_step": 0, "history": []}
                for name in ARMS}
        for step, (source, ids, rng_seed) in enumerate(curriculum, 1):
            for name in ARMS:
                arm = arms[name]
                batch = prepare(source, ids, np.random.default_rng(rng_seed),
                                name, True)
                arm["p"], arm["m"], arm["v"], loss = update(
                    arm["p"], arm["m"], arm["v"], _tensor(batch), step)
                if step % max(1, touch_steps // 4) == 0 or \
                        step == len(curriculum):
                    score = validation_loss(arm["p"])
                    arm["history"].append({"step": step,
                                           "validation_loss": score,
                                           "batch_loss": float(loss)})
                    if score < arm["best_score"]:
                        arm["best"] = arm["p"]
                        arm["best_score"] = score
                        arm["best_step"] = step
        reports = {}
        for name, arm in arms.items():
            checkpoint = folder / f"seed-{seed}-{name}.npz"
            np.savez_compressed(checkpoint, **jax.tree.map(np.asarray,
                                                          arm["best"]))
            datasets = {}
            for case, data_set in test.items():
                stresses = ("available", "no_touch", "no_proprio",
                            "no_vibration", "touch_time_shifted") \
                    if case.startswith("touch") else ("available",)
                datasets[case] = {
                    stress: evaluate(arm["best"], data_set, size, prior,
                                     name, stress)
                    for stress in stresses}
            reports[name] = {"selected_step": arm["best_step"],
                             "validation_loss": arm["best_score"],
                             "history": arm["history"],
                             "checkpoint_sha256": hashlib.sha256(
                                 checkpoint.read_bytes()).hexdigest(),
                             "test": datasets}
        results.append({"seed": int(seed), "arms": reports})
    document = {
        "protocol": "embodied-arithmetic-and-active-touch-v1",
        "seeds": list(map(int, seeds)), "stage_1_spring_updates": spring_steps,
        "stage_2_mixed_updates": touch_steps,
        "sequence_length": steps_per_episode, "size": size, "width": width,
        "batch_size": batch_size, "train_episodes_per_stage": train_count,
        "validation_episodes_per_stage": val_count,
        "test_episodes_per_case": test_count,
        "arm_names": list(ARMS),
        "train_seed_roots": [31000, 32000],
        "validation_seed_roots": [61000, 62000],
        "test_seed_roots": list(range(91000, 96001, 1000)),
        "controls": "Matched initializations, minibatch draws and update budgets; "
                    "one arm destroys within-episode temporal alignment of tactile "
                    "and vibration inputs, with supervision unchanged.",
        "evaluation": "Future observation and contour labels are never inputs. "
                      "Blind-touch reconstruction is scored on unprobed pixels "
                      "against a train-set mean-shape prior.",
        "limitations": "Tiny hand-designed mechanics and fivefold disc; action grammar "
                       "is known syntax with supervised numeric consequences, not "
                       "natural language. No real body, learned policy, calibration "
                       "of subjective experience, sentience, or qualia. "
                       "Equal training steps do not prove matched runtime.",
        "results": results
    }
    (folder / "metrics.json").write_text(json.dumps(document, indent=2) + "\n")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/embodied-causal")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 41])
    parser.add_argument("--spring-steps", type=int, default=40)
    parser.add_argument("--touch-steps", type=int, default=80)
    parser.add_argument("--steps-per-episode", type=int, default=24)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--val-count", type=int, default=12)
    parser.add_argument("--test-count", type=int, default=16)
    a = parser.parse_args()
    report = run(a.out, a.seeds, a.spring_steps, a.touch_steps,
                 a.steps_per_episode, a.size, a.width, a.batch_size,
                 a.train_count, a.val_count, a.test_count)
    print(json.dumps({"protocol": report["protocol"],
                      "seeds": report["seeds"], "out": a.out}, indent=2))


if __name__ == "__main__":
    main()
