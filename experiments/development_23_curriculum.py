"""Roadmap 23: paired staged-vs-mixed RGB NCA training with identical exposure.

Only order changes: both arms consume the exact same training episodes,
the same replay events, architecture, initialization and update budget.
"""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as nca, rgb_world_ladder as worlds
from train import adamw

DOMAINS = ("linear", "accelerating", "occlusion_long", "combined")


def schedules(steps, seed):
    if steps < 4:
        raise ValueError("At least four updates required")
    replay = tuple(i for i in range(steps) if (i + 1) % 10 == 0)
    main_count = steps - len(replay)
    ids = np.arange(main_count)
    staged = ids.copy()
    mixed = np.random.default_rng(seed).permutation(ids)
    def inject(order):
        output, j, k = [], 0, 0
        for step in range(steps):
            if step in replay:
                output.append(("replay", k))
                k += 1
            else:
                output.append(("main", int(order[j])))
                j += 1
        return output
    return {"staged": inject(staged), "mixed": inject(mixed)}


def episodes(steps, context, horizon, size):
    main_count = steps - steps // 10
    splits = np.array_split(np.arange(main_count), len(DOMAINS))
    main = {}
    for group, indexes in enumerate(splits):
        for j in indexes:
            main[int(j)] = worlds.episode(31000+int(j), context+horizon,
                                          size, domain=DOMAINS[group])["frames"]
    replay = [worlds.episode(51000+j, context+horizon, size,
                             domain="linear")["frames"] for j in range(steps//10)]
    validation = np.stack([worlds.episode(61000+i, context+horizon, size,
                                         domain=d)["frames"] for i, d in enumerate(DOMAINS)])
    test = np.stack([worlds.episode(91000+i, context+horizon, size,
                                   domain=d)["frames"] for i, d in enumerate(DOMAINS)])
    return {"main": main, "replay": replay}, validation, test


def score(params, frames, context, horizon, ticks):
    p = np.asarray(nca.rollout(params, jnp.asarray(frames[:, :context]), horizon, ticks))
    truth = frames[:, context:context+horizon]
    return float(np.square(p-truth).mean())


def run(out="runs/development-23", seeds=(11, 23, 41), steps=40,
        context=4, horizon=3, size=20, channels=6, ticks=1):
    if not seeds or context < 2 or horizon < 1 or channels < 1 or ticks < 1:
        raise ValueError("Invalid experiment")
    sequences, val, test = episodes(steps, context, horizon, size)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        orders = schedules(steps, int(seed))
        initial = nca.init(jax.random.PRNGKey(int(seed)), channels)
        for arm, order in orders.items():
            params = initial
            m = jax.tree.map(jnp.zeros_like, initial)
            v = jax.tree.map(jnp.zeros_like, initial)
            history = [{"step": 0, "validation_mse": score(initial, val, context, horizon, ticks)}]
            best = initial
            best_val = history[0]["validation_mse"]
            best_step = 0
            for step, (family, index) in enumerate(order, 1):
                sample = jnp.asarray(sequences[family][index][None])
                objective = lambda p: nca.loss(p, sample, context, horizon, ticks)
                _, grads = jax.value_and_grad(objective)(params)
                norm = jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(grads)))
                grads = jax.tree.map(lambda g: g/jnp.maximum(norm, 1.), grads)
                params, m, v = adamw(params, grads, m, v, step)
                if step == steps or step % max(1, steps//4) == 0:
                    metric = score(params, val, context, horizon, ticks)
                    history.append({"step": step, "validation_mse": metric})
                    if metric < best_val:
                        best, best_val, best_step = params, metric, step
            name = f"seed-{seed}-{arm}"
            np.savez_compressed(folder/f"{name}.npz",
                                **jax.tree.map(np.asarray, best))
            rows.append({"seed": int(seed), "arm": arm, "best_step": best_step,
                         "validation_mse": best_val,
                         "selected_test_mse": score(best, test, context, horizon, ticks),
                         "final_test_mse": score(params, test, context, horizon, ticks),
                         "history": history})
    report = {"roadmap_id": 23, "seed_list": list(map(int, seeds)), "steps": steps,
              "context": context, "horizon": horizon, "domains": list(DOMAINS),
              "replay_fraction": (steps//10)/steps, "training_episode_multiset_equal": True,
              "replay_updates_identical": True, "results": rows,
              "limits": "Same RGB episodes and update counts in different order, "
              "not equal wall-clock FLOPs. Preliminary tiny NCA campaign; "
              "training data never includes object or camera truth. "
              "No generalization benefit claimed without replicated held-out results."}
    (folder/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/development-23")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seeds", nargs="+", type=int, default=[11,23,41])
    a = p.parse_args()
    print(json.dumps(run(a.out, a.seeds, a.steps)["results"], indent=2))


if __name__ == "__main__":
    main()
