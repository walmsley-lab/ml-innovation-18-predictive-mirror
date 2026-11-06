"""Train a shared local state-transition rule on arbitrary connected geometries.

Every arm gets the same geometry, contact actions and sparse sensor streams.
Only observed sensors supervise training. Test includes unseen topologies, edge
lesions, action misalignment, and a no-communication architecture ablation.
"""
import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import geometry_automata as world
from src import geometry_harmonics as harmonics
from src import geometry_perceiver as model
from train import adamw

ARMS = ("messages", "no_messages")


def data(seed, nodes, kind, medium="spring"):
    if medium == "spring":
        return world.episode(seed, nodes=nodes, kind=kind, steps=18)
    if medium == "harmonic":
        return harmonics.sensor_record(seed, nodes=nodes, kind=kind, steps=18)
    raise ValueError("Unknown geometry medium")


def test_score(params, sample, use_messages, stress="normal"):
    obs = np.array(sample["observed"], copy=True)
    masks = np.array(sample["sensor_mask"], copy=True)
    force = np.array(sample["forces"], copy=True)
    graph = np.array(sample["adjacency"], copy=True)
    if stress == "shifted_actions":
        force = np.roll(force, 3, axis=0)
    elif stress == "cut_graph":
        graph[:] = 0.
    elif stress == "no_sensors":
        masks[:] = 0.
        obs[:] = 0.
    elif stress != "normal":
        raise ValueError("Unknown stress")
    predictions = np.asarray(model.predict(
        params, jnp.asarray(obs), jnp.asarray(masks),
        jnp.asarray(force), jnp.asarray(graph), use_messages))
    # Evaluator truth is not fed back into inference, including during stresses.
    truth = sample["truth"][1:]
    sensed = np.broadcast_to(sample["sensor_mask"][1:], truth.shape).astype(bool)
    errors = (predictions - truth)**2
    baseline = (sample["observed"][:-1] - truth)**2
    result = {
        "sensed_mse": float(errors[sensed].mean()),
        "persistence_mse": float(baseline[sensed].mean()),
        "all_sites_evaluator_mse": float(errors.mean()),
        "steps": int(len(predictions)),
    }
    return result


def run(out="runs/geometry-automata", seeds=(11, 23, 41),
        steps=50, channels=12, medium="spring"):
    if (not seeds or len(set(seeds)) != len(seeds) or steps < 1 or
            channels < 2 or medium not in ("spring", "harmonic")):
        raise ValueError("Invalid experiment settings")
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    # Entire generated worlds are disjoint, not just randomly split time slices.
    train = [data(31000+i, (10, 13, 16)[i % 3],
                  ("irregular", "branch", "ring")[i % 3], medium) for i in range(30)]
    valid = [data(61000+i, (10, 13, 16)[i % 3],
                  ("ring", "irregular", "branch")[i % 3], medium) for i in range(6)]
    test = {
        "new_irregular_21": data(91011, 21, "irregular", medium),
        "new_branch_19": data(91023, 19, "branch", medium),
        "new_ring_23": data(91041, 23, "ring", medium),
    }

    def tensors(sample):
        return tuple(jnp.asarray(sample[key]) for key in
                     ("observed", "sensor_mask", "forces", "adjacency"))

    @partial(jax.jit, static_argnames=("use_messages",))
    def update(p, m, v, sample, count, use_messages):
        objective = lambda q: model.loss(q, *sample, use_messages)
        score, gradients = jax.value_and_grad(objective)(p)
        norm = jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(gradients)))
        gradients = jax.tree.map(lambda g: g / jnp.maximum(1., norm), gradients)
        p, m, v = adamw(p, gradients, m, v, count, rate=.002)
        return p, m, v, score

    def validation(p, use_messages):
        return float(np.mean([float(model.loss(p, *tensors(item), use_messages))
                              for item in valid]))

    reports = []
    for seed in seeds:
        start = model.init(jax.random.PRNGKey(int(seed)), channels)
        rng = np.random.default_rng(int(seed) + 2900)
        indices = rng.integers(len(train), size=steps)
        arms = {}
        for arm in ARMS:
            p = start
            m = jax.tree.map(jnp.zeros_like, start)
            v = jax.tree.map(jnp.zeros_like, start)
            allowed = arm == "messages"
            best = p
            best_val = validation(p, allowed)
            best_step = 0
            curve = []
            for index, example_id in enumerate(indices, 1):
                p, m, v, value = update(p, m, v, tensors(train[int(example_id)]),
                                         index, allowed)
                if index % max(1, steps // 4) == 0 or index == steps:
                    val = validation(p, allowed)
                    curve.append({"step": index, "train_loss": float(value),
                                  "validation_loss": val})
                    if val < best_val:
                        best, best_val, best_step = p, val, index
            path = destination / f"seed-{seed}-{arm}.npz"
            np.savez_compressed(path, **jax.tree.map(np.asarray, best))
            scores = {}
            for name, record in test.items():
                scores[name] = {stress: test_score(best, record, allowed, stress)
                                for stress in ("normal", "shifted_actions",
                                               "cut_graph", "no_sensors")}
            arms[arm] = {"validation_loss": best_val,
                         "selected_step": best_step, "learning_curve": curve,
                         "checkpoint": path.name,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "test": scores,
                         "parameter_count": int(sum(np.size(x) for x in
                                                    jax.tree.leaves(best)))}
        reports.append({"seed": int(seed), "arms": arms})
    result = {
        "experiment": "arbitrary_geometry_to_local_perception_v1",
        "seeds": list(map(int, seeds)), "steps": steps, "channels": channels,
        "medium": medium,
        "train_seed_range": [31000, 31029], "validation_seed_range": [61000, 61005],
        "test_worlds": {name: {"seed": record["seed"],
                               "kind": record["geometry_kind"],
                               "nodes": len(record["coords"]),
                               "sensed_nodes": int(record["sensor_mask"][0].sum())}
                        for name, record in test.items()},
        "reports": reports,
        "limits": "Known arbitrary topology and outgoing local motor force; sparse "
                  "displacement sensors only, with future observations held out. "
                  "The mechanical simulator, action sequence and sensor placements "
                  "are hand-coded. No learned actions, self-model, real-world "
                  "physical embodiment or subjective experience. "
                  "Paired weight/step budgets do not guarantee matched computation. "
                  "Graph-cut stress also alters model normalization/readout inputs, "
                  "so score changes are a pathway intervention, not mechanistic proof."
    }
    (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/geometry-automata")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 41])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--medium", choices=("spring", "harmonic"), default="spring")
    args = parser.parse_args()
    report = run(args.out, tuple(args.seeds), args.steps, args.channels,
                 args.medium)
    print(json.dumps({"seeds": report["seeds"],
                      "test_worlds": report["test_worlds"], "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
